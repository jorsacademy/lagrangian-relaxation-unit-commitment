from __future__ import annotations

import numpy as np


class IdleBatteryPolicy:
    """Never charge or discharge the battery."""

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


class RuleBasedTariffPolicy:
    """Charge at low prices, discharge at high prices and high load."""

    def __init__(self, low_price_norm: float = 0.42, high_price_norm: float = 0.62, high_load_norm: float = 0.72):
        self.low_price_norm = low_price_norm
        self.high_price_norm = high_price_norm
        self.high_load_norm = high_load_norm

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        soc, price_norm, load_norm, renewable_norm, _ = observation
        if (price_norm >= self.high_price_norm or load_norm >= self.high_load_norm) and soc > 0.2:
            return np.array([0.85], dtype=np.float32)
        if price_norm <= self.low_price_norm and soc < 0.85:
            return np.array([-0.65], dtype=np.float32)
        if renewable_norm > 0.35 and soc < 0.9:
            return np.array([-0.45], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)


class PeakShavingPolicy:
    """Approximate MPC-style controller using the current load/renewable state."""

    def __init__(self, target_grid_norm: float = 0.72):
        self.target_grid_norm = target_grid_norm

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        soc, price_norm, load_norm, renewable_norm, _ = observation
        net_load_norm = max(load_norm - renewable_norm * 0.4, 0.0)
        error = net_load_norm - self.target_grid_norm
        if error > 0.0 and soc > 0.1:
            return np.array([min(1.0, 3.0 * error)], dtype=np.float32)
        if error < -0.12 and price_norm < 0.55 and soc < 0.9:
            return np.array([max(-1.0, 2.5 * error)], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)
