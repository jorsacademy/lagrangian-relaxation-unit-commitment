from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .data import UnitCommitmentInstance
from .patterns import CommitmentPattern, enumerate_commitment_patterns, pattern_fixed_cost


@dataclass(frozen=True)
class UCSolution:
    objective: float
    commitment: np.ndarray
    generation: np.ndarray
    startups: np.ndarray
    pattern_indices: tuple[int, ...]
    mip_nodes: int


def _build_pattern_library(
    instance: UnitCommitmentInstance,
) -> tuple[tuple[CommitmentPattern, ...], ...]:
    return tuple(
        enumerate_commitment_patterns(unit, instance.horizon) for unit in instance.units
    )


def _dispatch_fixed_commitment(
    instance: UnitCommitmentInstance, commitment: np.ndarray
) -> tuple[float, np.ndarray]:
    n_units = len(instance.units)
    horizon = instance.horizon
    if commitment.shape != (n_units, horizon):
        raise ValueError("commitment has the wrong shape")

    generation = np.zeros((n_units, horizon), dtype=float)
    variable_cost = 0.0
    for t in range(horizon):
        p_min = np.asarray(
            [instance.units[i].p_min * commitment[i, t] for i in range(n_units)], dtype=float
        )
        p_max = np.asarray(
            [instance.units[i].p_max * commitment[i, t] for i in range(n_units)], dtype=float
        )
        if p_min.sum() > instance.demand[t] + 1e-9 or p_max.sum() < instance.demand[t] - 1e-9:
            raise RuntimeError("fixed commitment cannot meet demand")
        remaining = float(instance.demand[t] - p_min.sum())
        order = np.argsort([unit.marginal_cost for unit in instance.units])
        p = p_min.copy()
        for i in order:
            headroom = p_max[i] - p[i]
            addition = min(headroom, remaining)
            p[i] += addition
            remaining -= addition
            if remaining <= 1e-10:
                break
        if remaining > 1e-8:
            raise RuntimeError("economic dispatch failed to satisfy demand")
        generation[:, t] = p
        variable_cost += sum(instance.units[i].marginal_cost * p[i] for i in range(n_units))
    return float(variable_cost), generation


def solve_exact_uc(instance: UnitCommitmentInstance) -> UCSolution:
    libraries = _build_pattern_library(instance)
    n_units = len(instance.units)
    horizon = instance.horizon

    offsets: list[int] = []
    count = 0
    for patterns in libraries:
        offsets.append(count)
        count += len(patterns)
    n_pattern_vars = count
    n_generation_vars = n_units * horizon
    total_vars = n_pattern_vars + n_generation_vars

    c = np.zeros(total_vars, dtype=float)
    integrality = np.zeros(total_vars, dtype=int)
    integrality[:n_pattern_vars] = 1
    ub = np.full(total_vars, np.inf, dtype=float)
    ub[:n_pattern_vars] = 1.0

    for i, patterns in enumerate(libraries):
        for k, pattern in enumerate(patterns):
            c[offsets[i] + k] = pattern_fixed_cost(instance.units[i], pattern)
    for i, unit in enumerate(instance.units):
        for t in range(horizon):
            c[n_pattern_vars + i * horizon + t] = unit.marginal_cost

    rows: list[np.ndarray] = []
    lb: list[float] = []
    row_ub: list[float] = []

    for i, patterns in enumerate(libraries):
        row = np.zeros(total_vars, dtype=float)
        row[offsets[i] : offsets[i] + len(patterns)] = 1.0
        rows.append(row)
        lb.append(1.0)
        row_ub.append(1.0)

    for t in range(horizon):
        row = np.zeros(total_vars, dtype=float)
        for i in range(n_units):
            row[n_pattern_vars + i * horizon + t] = 1.0
        rows.append(row)
        lb.append(float(instance.demand[t]))
        row_ub.append(float(instance.demand[t]))

    for i, unit in enumerate(instance.units):
        patterns = libraries[i]
        for t in range(horizon):
            gen_index = n_pattern_vars + i * horizon + t

            lower_row = np.zeros(total_vars, dtype=float)
            lower_row[gen_index] = 1.0
            for k, pattern in enumerate(patterns):
                lower_row[offsets[i] + k] -= unit.p_min * pattern.on[t]
            rows.append(lower_row)
            lb.append(0.0)
            row_ub.append(np.inf)

            upper_row = np.zeros(total_vars, dtype=float)
            upper_row[gen_index] = 1.0
            for k, pattern in enumerate(patterns):
                upper_row[offsets[i] + k] -= unit.p_max * pattern.on[t]
            rows.append(upper_row)
            lb.append(-np.inf)
            row_ub.append(0.0)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(np.zeros(total_vars), ub),
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lb), np.asarray(row_ub)),
        options={"presolve": True},
    )
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"exact UC solve failed: {result.message}")

    pattern_indices: list[int] = []
    commitment = np.zeros((n_units, horizon), dtype=int)
    startups = np.zeros((n_units, horizon), dtype=int)
    for i, patterns in enumerate(libraries):
        local = result.x[offsets[i] : offsets[i] + len(patterns)]
        k = int(np.argmax(local))
        pattern_indices.append(k)
        commitment[i] = np.asarray(patterns[k].on, dtype=int)
        startups[i] = np.asarray(patterns[k].startups, dtype=int)

    generation = result.x[n_pattern_vars:].reshape(n_units, horizon)
    return UCSolution(
        objective=float(result.fun),
        commitment=commitment,
        generation=np.asarray(generation, dtype=float),
        startups=startups,
        pattern_indices=tuple(pattern_indices),
        mip_nodes=int(getattr(result, "mip_node_count", 0) or 0),
    )


def recover_primal(
    instance: UnitCommitmentInstance,
    target_pattern_indices: tuple[int, ...],
) -> UCSolution:
    libraries = _build_pattern_library(instance)
    if len(target_pattern_indices) != len(instance.units):
        raise ValueError("target pattern vector has the wrong length")

    offsets: list[int] = []
    count = 0
    for patterns in libraries:
        offsets.append(count)
        count += len(patterns)

    c = np.zeros(count, dtype=float)
    integrality = np.ones(count, dtype=int)
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []

    for i, patterns in enumerate(libraries):
        target = patterns[target_pattern_indices[i]]
        for k, pattern in enumerate(patterns):
            distance = sum(abs(a - b) for a, b in zip(pattern.on, target.on, strict=True))
            c[offsets[i] + k] = float(distance)
        row = np.zeros(count, dtype=float)
        row[offsets[i] : offsets[i] + len(patterns)] = 1.0
        rows.append(row)
        lb.append(1.0)
        ub.append(1.0)

    for t in range(instance.horizon):
        max_row = np.zeros(count, dtype=float)
        min_row = np.zeros(count, dtype=float)
        for i, unit in enumerate(instance.units):
            for k, pattern in enumerate(libraries[i]):
                max_row[offsets[i] + k] += unit.p_max * pattern.on[t]
                min_row[offsets[i] + k] += unit.p_min * pattern.on[t]
        rows.append(max_row)
        lb.append(float(instance.demand[t]))
        ub.append(np.inf)
        rows.append(min_row)
        lb.append(-np.inf)
        ub.append(float(instance.demand[t]))

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(np.zeros(count), np.ones(count)),
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lb), np.asarray(ub)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"primal recovery failed: {result.message}")

    pattern_indices: list[int] = []
    commitment = np.zeros((len(instance.units), instance.horizon), dtype=int)
    startups = np.zeros_like(commitment)
    fixed_cost = 0.0
    for i, patterns in enumerate(libraries):
        local = result.x[offsets[i] : offsets[i] + len(patterns)]
        k = int(np.argmax(local))
        pattern_indices.append(k)
        commitment[i] = np.asarray(patterns[k].on, dtype=int)
        startups[i] = np.asarray(patterns[k].startups, dtype=int)
        fixed_cost += pattern_fixed_cost(instance.units[i], patterns[k])

    variable_cost, generation = _dispatch_fixed_commitment(instance, commitment)
    return UCSolution(
        objective=float(fixed_cost + variable_cost),
        commitment=commitment,
        generation=generation,
        startups=startups,
        pattern_indices=tuple(pattern_indices),
        mip_nodes=int(getattr(result, "mip_node_count", 0) or 0),
    )


def solve_dispatch_lp(instance: UnitCommitmentInstance, commitment: np.ndarray) -> float:
    n_units = len(instance.units)
    horizon = instance.horizon
    c = np.asarray(
        [instance.units[i].marginal_cost for i in range(n_units) for _ in range(horizon)],
        dtype=float,
    )
    a_eq = np.zeros((horizon, n_units * horizon), dtype=float)
    for t in range(horizon):
        for i in range(n_units):
            a_eq[t, i * horizon + t] = 1.0
    bounds = []
    for i, unit in enumerate(instance.units):
        for t in range(horizon):
            bounds.append((unit.p_min * commitment[i, t], unit.p_max * commitment[i, t]))
    result = linprog(c, A_eq=a_eq, b_eq=instance.demand, bounds=bounds, method="highs")
    if not result.success or result.fun is None:
        raise RuntimeError("dispatch LP failed")
    return float(result.fun)
