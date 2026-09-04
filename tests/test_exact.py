import numpy as np

from lagrangian_uc import demo_instance, recover_primal, solve_exact_uc


def test_exact_solution_meets_demand_and_generation_bounds() -> None:
    instance = demo_instance()
    solution = solve_exact_uc(instance)
    assert np.allclose(solution.generation.sum(axis=0), instance.demand, atol=1e-7)
    for i, unit in enumerate(instance.units):
        assert np.all(solution.generation[i] >= unit.p_min * solution.commitment[i] - 1e-8)
        assert np.all(solution.generation[i] <= unit.p_max * solution.commitment[i] + 1e-8)


def test_recovery_returns_feasible_upper_bound() -> None:
    instance = demo_instance()
    exact = solve_exact_uc(instance)
    recovered = recover_primal(instance, exact.pattern_indices)
    assert np.isclose(recovered.objective, exact.objective, atol=1e-7)
