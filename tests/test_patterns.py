from lagrangian_uc import ThermalUnit, enumerate_commitment_patterns


def test_minimum_up_down_transitions_respect_initial_state() -> None:
    unit = ThermalUnit(
        name="u",
        p_min=10,
        p_max=40,
        marginal_cost=20,
        no_load_cost=5,
        startup_cost=10,
        min_up=2,
        min_down=2,
        initial_on=False,
        initial_duration=1,
    )
    patterns = enumerate_commitment_patterns(unit, 3)
    assert patterns
    assert all(pattern.on[0] == 0 for pattern in patterns)
    assert any(pattern.on == (0, 1, 1) for pattern in patterns)
