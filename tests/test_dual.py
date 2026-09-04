import numpy as np

from lagrangian_uc import demo_instance, evaluate_dual, solve_exact_uc


def test_lagrangian_dual_is_valid_lower_bound() -> None:
    instance = demo_instance()
    exact = solve_exact_uc(instance)
    for level in (10.0, 25.0, 40.0, 60.0):
        dual = evaluate_dual(instance, np.full(instance.horizon, level))
        assert dual.value <= exact.objective + 1e-7
        assert np.allclose(dual.subgradient, instance.demand - dual.total_generation)
