import numpy as np

from industrial_energy_management.environment import EnergyConfig, IndustrialEnergyEnv


def test_reset_is_seed_deterministic():
    env_a = IndustrialEnergyEnv()
    env_b = IndustrialEnergyEnv()
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)
    np.testing.assert_allclose(env_a._load, env_b._load)
    np.testing.assert_allclose(env_a._renewable, env_b._renewable)


def test_environment_runs_full_horizon():
    cfg = EnergyConfig(horizon=12)
    env = IndustrialEnergyEnv(cfg)
    obs, _ = env.reset(seed=7)
    steps = 0
    done = False
    while not done:
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        assert np.isfinite(reward)
        done = terminated or truncated
        steps += 1
    assert steps == cfg.horizon
    assert info["total_cost"] >= 0.0


def test_soc_respects_bounds_under_aggressive_actions():
    env = IndustrialEnergyEnv(EnergyConfig(horizon=40))
    env.reset(seed=11)
    for action in (1.0, -1.0):
        env.reset(seed=11)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
            assert 0.0 <= info["soc"] <= 1.0
            done = terminated or truncated


def test_discharging_reduces_grid_import_when_feasible():
    env_idle = IndustrialEnergyEnv(EnergyConfig(horizon=4))
    env_discharge = IndustrialEnergyEnv(EnergyConfig(horizon=4))
    env_idle.reset(seed=5)
    env_discharge.reset(seed=5)
    _, _, _, _, idle_info = env_idle.step(np.array([0.0], dtype=np.float32))
    _, _, _, _, discharge_info = env_discharge.step(np.array([0.5], dtype=np.float32))
    assert discharge_info["grid_import_kw"] <= idle_info["grid_import_kw"]
