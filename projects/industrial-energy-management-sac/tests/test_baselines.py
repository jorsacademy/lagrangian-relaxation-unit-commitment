from industrial_energy_management.baselines import IdleBatteryPolicy, PeakShavingPolicy, RuleBasedTariffPolicy
from industrial_energy_management.evaluate import run_policy


def test_baselines_return_finite_results():
    for policy in (IdleBatteryPolicy(), RuleBasedTariffPolicy(), PeakShavingPolicy()):
        result = run_policy(policy, episodes=2, seed=50)
        assert result.mean_total_cost >= 0.0
        assert result.mean_peak_kw >= 0.0
        assert result.mean_energy_cost >= 0.0


def test_idle_policy_has_zero_battery_wear_cost():
    result = run_policy(IdleBatteryPolicy(), episodes=2, seed=60)
    assert result.mean_degradation_cost == 0.0
