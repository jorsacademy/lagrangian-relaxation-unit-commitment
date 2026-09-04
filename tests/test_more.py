import numpy as np

from lagrangian_uc import (
    demo_instance,
    enumerate_commitment_patterns,
    evaluate_dual,
    run_benchmark,
    solve_exact_uc,
    solve_unit_subproblem,
)


def test_demo_exact_objective_regression() -> None:
    solution = solve_exact_uc(demo_instance())
    assert np.isclose(solution.objective, 20705.0, atol=1e-7)


def test_startup_markers_match_off_to_on_transitions() -> None:
    instance = demo_instance()
    unit = instance.units[2]
    for pattern in enumerate_commitment_patterns(unit, instance.horizon):
        previous = int(unit.initial_on)
        for on, startup in zip(pattern.on, pattern.startups, strict=True):
            assert startup == int(previous == 0 and on == 1)
            previous = on


def test_high_energy_price_multiplier_encourages_generation() -> None:
    instance = demo_instance()
    unit = instance.units[-1]
    multipliers = np.full(instance.horizon, 100.0)
    result = solve_unit_subproblem(unit, instance.horizon, multipliers)
    on = np.asarray(result.pattern.on, dtype=bool)
    assert np.allclose(result.generation[on], unit.p_max)


def test_dual_subgradient_is_concave_supergradient() -> None:
    instance = demo_instance()
    base = np.full(instance.horizon, 30.0)
    evaluation = evaluate_dual(instance, base)
    perturbation = np.asarray([1.0, -0.5, 0.8, -1.2, 0.3, 0.7, -0.4, 0.2])
    trial = evaluate_dual(instance, base + perturbation)
    linear_upper = evaluation.value + float(np.dot(evaluation.subgradient, perturbation))
    assert trial.value <= linear_upper + 1e-7


def test_benchmark_reports_valid_bound_ordering() -> None:
    result = run_benchmark(iterations=20)
    assert result["lagrangian"]["best_lower_bound"] <= result["exact"]["objective"] + 1e-7
    assert result["lagrangian"]["best_upper_bound"] >= result["exact"]["objective"] - 1e-7
