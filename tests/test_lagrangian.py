from lagrangian_uc import demo_instance, run_lagrangian_relaxation, solve_exact_uc


def test_subgradient_method_preserves_bound_interpretation() -> None:
    instance = demo_instance()
    exact = solve_exact_uc(instance)
    result = run_lagrangian_relaxation(instance, iterations=50)
    assert result.best_lower_bound <= exact.objective + 1e-7
    assert result.best_upper_bound >= exact.objective - 1e-7
    best_lbs = [row.best_lower_bound for row in result.history]
    assert all(b >= a - 1e-10 for a, b in zip(best_lbs, best_lbs[1:], strict=False))
    assert result.absolute_gap >= -1e-12
