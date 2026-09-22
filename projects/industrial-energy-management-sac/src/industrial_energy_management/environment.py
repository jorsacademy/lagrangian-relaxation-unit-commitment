from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class EnergyConfig:
    horizon: int = 96
    dt_hours: float = 0.25
    battery_capacity_kwh: float = 240.0
    battery_power_kw: float = 90.0
    initial_soc: float = 0.5
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    base_load_kw: float = 180.0
    load_amplitude_kw: float = 55.0
    renewable_peak_kw: float = 70.0
    demand_noise_std_kw: float = 8.0
    renewable_noise_std_kw: float = 5.0
    peak_limit_kw: float = 235.0
    demand_charge_rate: float = 0.08
    battery_degradation_cost_per_kwh: float = 0.015
    export_price_factor: float = 0.45


class IndustrialEnergyEnv(gym.Env):
    """A compact factory energy-management environment for continuous-control RL.

    The agent controls battery power to reduce energy cost and peak demand.
    Positive action discharges the battery; negative action charges it.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnergyConfig | None = None):
        super().__init__()
        self.config = config or EnergyConfig()
        self.action_space = spaces.Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32))
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 2.0, 2.0, 2.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._t = 0
        self._soc = self.config.initial_soc
        self._price = np.zeros(self.config.horizon, dtype=np.float32)
        self._load = np.zeros(self.config.horizon, dtype=np.float32)
        self._renewable = np.zeros(self.config.horizon, dtype=np.float32)
        self._cumulative_energy_cost = 0.0
        self._cumulative_peak_penalty = 0.0
        self._cumulative_degradation_cost = 0.0
        self._max_grid_import_kw = 0.0

    def _generate_profiles(self) -> None:
        h = self.config.horizon
        x = np.arange(h, dtype=np.float32)
        phase = 2 * np.pi * x / h

        price = 0.12 + 0.07 * (np.sin(phase - np.pi / 2) + 1.0) / 2.0
        price += 0.10 * ((x >= int(0.67 * h)) & (x <= int(0.83 * h)))

        load = self.config.base_load_kw + self.config.load_amplitude_kw * (np.sin(phase - 0.8) + 1.0) / 2.0
        load += self.np_random.normal(0.0, self.config.demand_noise_std_kw, size=h)

        solar_shape = np.maximum(0.0, np.sin(phase - np.pi / 2))
        renewable = self.config.renewable_peak_kw * solar_shape
        renewable += self.np_random.normal(0.0, self.config.renewable_noise_std_kw, size=h)

        self._price = np.clip(price, 0.05, None).astype(np.float32)
        self._load = np.clip(load, 20.0, None).astype(np.float32)
        self._renewable = np.clip(renewable, 0.0, None).astype(np.float32)

    def _observation(self) -> np.ndarray:
        idx = min(self._t, self.config.horizon - 1)
        return np.array(
            [
                self._soc,
                self._price[idx] / 0.35,
                self._load[idx] / 300.0,
                self._renewable[idx] / 120.0,
                idx / max(1, self.config.horizon - 1),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = 0
        self._soc = self.config.initial_soc
        self._cumulative_energy_cost = 0.0
        self._cumulative_peak_penalty = 0.0
        self._cumulative_degradation_cost = 0.0
        self._max_grid_import_kw = 0.0
        self._generate_profiles()
        return self._observation(), self._info()

    def _feasible_battery_power(self, requested_kw: float) -> float:
        cfg = self.config
        if requested_kw >= 0.0:  # discharge
            energy_available = self._soc * cfg.battery_capacity_kwh
            max_from_soc = energy_available * cfg.discharge_efficiency / cfg.dt_hours
            return float(min(requested_kw, cfg.battery_power_kw, max_from_soc))

        # charge
        remaining_capacity = (1.0 - self._soc) * cfg.battery_capacity_kwh
        max_to_soc = remaining_capacity / (cfg.charge_efficiency * cfg.dt_hours)
        return float(max(requested_kw, -cfg.battery_power_kw, -max_to_soc))

    def step(self, action):
        cfg = self.config
        requested_kw = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0)) * cfg.battery_power_kw
        battery_power_kw = self._feasible_battery_power(requested_kw)

        idx = self._t
        price = float(self._price[idx])
        load_kw = float(self._load[idx])
        renewable_kw = float(self._renewable[idx])

        if battery_power_kw >= 0.0:
            discharged_kwh = battery_power_kw * cfg.dt_hours / cfg.discharge_efficiency
            self._soc -= discharged_kwh / cfg.battery_capacity_kwh
        else:
            charged_kwh = (-battery_power_kw) * cfg.dt_hours * cfg.charge_efficiency
            self._soc += charged_kwh / cfg.battery_capacity_kwh
        self._soc = float(np.clip(self._soc, 0.0, 1.0))

        grid_kw = load_kw - renewable_kw - battery_power_kw
        import_kw = max(grid_kw, 0.0)
        export_kw = max(-grid_kw, 0.0)

        import_cost = import_kw * cfg.dt_hours * price
        export_credit = export_kw * cfg.dt_hours * price * cfg.export_price_factor
        energy_cost = import_cost - export_credit
        peak_excess = max(import_kw - cfg.peak_limit_kw, 0.0)
        peak_penalty = peak_excess * cfg.demand_charge_rate
        degradation_cost = abs(battery_power_kw) * cfg.dt_hours * cfg.battery_degradation_cost_per_kwh
        operating_cost = energy_cost + peak_penalty + degradation_cost

        self._cumulative_energy_cost += energy_cost
        self._cumulative_peak_penalty += peak_penalty
        self._cumulative_degradation_cost += degradation_cost
        self._max_grid_import_kw = max(self._max_grid_import_kw, import_kw)

        self._t += 1
        terminated = self._t >= cfg.horizon
        truncated = False
        reward = -float(operating_cost)
        observation = self._observation() if not terminated else np.array(
            [self._soc, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        return observation, reward, terminated, truncated, self._info(
            battery_power_kw=battery_power_kw,
            grid_import_kw=import_kw,
            grid_export_kw=export_kw,
            step_cost=operating_cost,
        )

    def _info(self, **extra) -> dict:
        total_cost = self._cumulative_energy_cost + self._cumulative_peak_penalty + self._cumulative_degradation_cost
        info = {
            "soc": float(self._soc),
            "energy_cost": float(self._cumulative_energy_cost),
            "peak_penalty": float(self._cumulative_peak_penalty),
            "degradation_cost": float(self._cumulative_degradation_cost),
            "total_cost": float(total_cost),
            "max_grid_import_kw": float(self._max_grid_import_kw),
        }
        info.update(extra)
        return info
