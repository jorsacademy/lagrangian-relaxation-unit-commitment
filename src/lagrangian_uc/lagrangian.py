from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import ThermalUnit, UnitCommitmentInstance
from .master import UCSolution, recover_primal
from .patterns import CommitmentPattern, enumerate_commitment_patterns, pattern_fixed_cost


@dataclass(frozen=True)
class UnitSubproblemResult:
    pattern_index: int
    pattern: CommitmentPattern
    generation: np.ndarray
    value: float


@dataclass(frozen=True)
class DualEvaluation:
    multipliers: np.ndarray
    value: float
    total_generation: np.ndarray
    subgradient: np.ndarray
    unit_results: tuple[UnitSubproblemResult, ...]


@dataclass(frozen=True)
class LRIteration:
    iteration: int
    dual_value: float
    best_lower_bound: float
    primal_value: float
    best_upper_bound: float
    subgradient_norm: float
    step_size: float


@dataclass(frozen=True)
class LagrangianResult:
    best_lower_bound: float
    best_upper_bound: float
    best_multipliers: np.ndarray
    best_primal: UCSolution
    history: tuple[LRIteration, ...]

    @property
    def absolute_gap(self) -> float:
        return max(0.0, self.best_upper_bound - self.best_lower_bound)

    @property
    def relative_gap(self) -> float:
        scale = max(1.0, abs(self.best_upper_bound))
        return self.absolute_gap / scale


def solve_unit_subproblem(
    unit: ThermalUnit,
    horizon: int,
    multipliers: np.ndarray,
) -> UnitSubproblemResult:
    if multipliers.shape != (horizon,):
        raise ValueError("multipliers have the wrong shape")
    patterns = enumerate_commitment_patterns(unit, horizon)
    best_value = np.inf
    best_index = -1
    best_generation: np.ndarray | None = None

    for k, pattern in enumerate(patterns):
        on = np.asarray(pattern.on, dtype=float)
        generation = np.zeros(horizon, dtype=float)
        for t in range(horizon):
            if on[t] < 0.5:
                continue
            reduced_marginal = unit.marginal_cost - multipliers[t]
            generation[t] = unit.p_max if reduced_marginal < 0.0 else unit.p_min
        value = pattern_fixed_cost(unit, pattern) + float(
            np.dot(unit.marginal_cost - multipliers, generation)
        )
        if value < best_value - 1e-10:
            best_value = value
            best_index = k
            best_generation = generation

    if best_generation is None or best_index < 0:
        raise RuntimeError("unit subproblem produced no feasible commitment pattern")
    return UnitSubproblemResult(
        pattern_index=best_index,
        pattern=patterns[best_index],
        generation=best_generation,
        value=float(best_value),
    )


def evaluate_dual(
    instance: UnitCommitmentInstance,
    multipliers: np.ndarray,
) -> DualEvaluation:
    if multipliers.shape != (instance.horizon,):
        raise ValueError("multipliers have the wrong shape")
    unit_results = tuple(
        solve_unit_subproblem(unit, instance.horizon, multipliers) for unit in instance.units
    )
    total_generation = np.sum([result.generation for result in unit_results], axis=0)
    value = float(np.dot(multipliers, instance.demand) + sum(r.value for r in unit_results))
    subgradient = np.asarray(instance.demand - total_generation, dtype=float)
    return DualEvaluation(
        multipliers=np.asarray(multipliers, dtype=float),
        value=value,
        total_generation=np.asarray(total_generation, dtype=float),
        subgradient=subgradient,
        unit_results=unit_results,
    )


def run_lagrangian_relaxation(
    instance: UnitCommitmentInstance,
    iterations: int = 120,
    theta: float = 1.6,
    stall_limit: int = 12,
    min_theta: float = 0.05,
) -> LagrangianResult:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not (0.0 < theta <= 2.0):
        raise ValueError("theta must be in (0, 2]")
    if stall_limit < 1 or min_theta <= 0:
        raise ValueError("invalid step-control parameters")

    multipliers = np.full(
        instance.horizon,
        np.median([unit.marginal_cost for unit in instance.units]),
        dtype=float,
    )

    initial_dual = evaluate_dual(instance, multipliers)
    initial_targets = tuple(result.pattern_index for result in initial_dual.unit_results)
    best_primal = recover_primal(instance, initial_targets)
    best_upper = best_primal.objective
    best_lower = -np.inf
    best_multipliers = multipliers.copy()
    history: list[LRIteration] = []
    no_improvement = 0
    current_theta = theta

    for iteration in range(1, iterations + 1):
        dual = evaluate_dual(instance, multipliers)
        if dual.value > best_lower + 1e-9:
            best_lower = dual.value
            best_multipliers = multipliers.copy()
            no_improvement = 0
        else:
            no_improvement += 1

        targets = tuple(result.pattern_index for result in dual.unit_results)
        primal = recover_primal(instance, targets)
        if primal.objective < best_upper - 1e-9:
            best_upper = primal.objective
            best_primal = primal

        norm_sq = float(np.dot(dual.subgradient, dual.subgradient))
        if norm_sq <= 1e-12:
            step_size = 0.0
        else:
            gap_for_step = max(0.0, best_upper - dual.value)
            step_size = current_theta * gap_for_step / norm_sq

        history.append(
            LRIteration(
                iteration=iteration,
                dual_value=dual.value,
                best_lower_bound=best_lower,
                primal_value=primal.objective,
                best_upper_bound=best_upper,
                subgradient_norm=float(np.sqrt(norm_sq)),
                step_size=float(step_size),
            )
        )

        if norm_sq <= 1e-12 or best_upper - best_lower <= 1e-8:
            break

        multipliers = multipliers + step_size * dual.subgradient

        if no_improvement >= stall_limit:
            current_theta = max(min_theta, 0.5 * current_theta)
            no_improvement = 0

    return LagrangianResult(
        best_lower_bound=float(best_lower),
        best_upper_bound=float(best_upper),
        best_multipliers=best_multipliers,
        best_primal=best_primal,
        history=tuple(history),
    )
