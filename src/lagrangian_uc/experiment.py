from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .data import UnitCommitmentInstance, demo_instance
from .lagrangian import LagrangianResult, run_lagrangian_relaxation
from .master import UCSolution, solve_exact_uc


def _solution_dict(solution: UCSolution) -> dict[str, object]:
    return {
        "objective": solution.objective,
        "mip_nodes": solution.mip_nodes,
        "pattern_indices": list(solution.pattern_indices),
        "commitment": solution.commitment.tolist(),
        "generation": np.round(solution.generation, 6).tolist(),
        "startups": solution.startups.tolist(),
    }


def run_benchmark(
    instance: UnitCommitmentInstance | None = None,
    iterations: int = 120,
    theta: float = 1.6,
) -> dict[str, object]:
    instance = demo_instance() if instance is None else instance
    exact = solve_exact_uc(instance)
    lr = run_lagrangian_relaxation(instance, iterations=iterations, theta=theta)
    return benchmark_to_dict(instance, exact, lr)


def benchmark_to_dict(
    instance: UnitCommitmentInstance,
    exact: UCSolution,
    lr: LagrangianResult,
) -> dict[str, object]:
    exact_gap = exact.objective - lr.best_lower_bound
    recovery_gap = lr.best_upper_bound - exact.objective
    return {
        "horizon": instance.horizon,
        "units": [unit.name for unit in instance.units],
        "demand": instance.demand.tolist(),
        "exact": _solution_dict(exact),
        "lagrangian": {
            "iterations": len(lr.history),
            "best_lower_bound": lr.best_lower_bound,
            "best_upper_bound": lr.best_upper_bound,
            "absolute_duality_gap_to_exact": exact_gap,
            "relative_duality_gap_to_exact": exact_gap / max(1.0, abs(exact.objective)),
            "recovered_primal_gap_to_exact": recovery_gap,
            "best_multipliers": np.round(lr.best_multipliers, 6).tolist(),
            "best_primal": _solution_dict(lr.best_primal),
            "history": [asdict(item) for item in lr.history],
        },
    }
