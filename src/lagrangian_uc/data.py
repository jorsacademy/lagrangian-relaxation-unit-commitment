from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThermalUnit:
    name: str
    p_min: float
    p_max: float
    marginal_cost: float
    no_load_cost: float
    startup_cost: float
    min_up: int
    min_down: int
    initial_on: bool
    initial_duration: int

    def __post_init__(self) -> None:
        if self.p_min < 0 or self.p_max <= 0 or self.p_min > self.p_max:
            raise ValueError("invalid generation limits")
        if self.marginal_cost < 0 or self.no_load_cost < 0 or self.startup_cost < 0:
            raise ValueError("cost coefficients must be nonnegative")
        if self.min_up < 1 or self.min_down < 1:
            raise ValueError("minimum up/down times must be positive integers")
        if self.initial_duration < 1:
            raise ValueError("initial_duration must be at least one hour")


@dataclass(frozen=True)
class UnitCommitmentInstance:
    units: tuple[ThermalUnit, ...]
    demand: np.ndarray

    def __post_init__(self) -> None:
        if not self.units:
            raise ValueError("at least one thermal unit is required")
        if self.demand.ndim != 1 or self.demand.size < 1:
            raise ValueError("demand must be a nonempty one-dimensional array")
        if np.any(self.demand < 0):
            raise ValueError("demand must be nonnegative")
        max_capacity = sum(unit.p_max for unit in self.units)
        min_all_on = sum(unit.p_min for unit in self.units)
        if float(np.max(self.demand)) > max_capacity + 1e-9:
            raise ValueError("demand exceeds total installed capacity")
        if min_all_on <= 0:
            raise ValueError("at least one unit must have positive minimum output")

    @property
    def horizon(self) -> int:
        return int(self.demand.size)


def demo_instance() -> UnitCommitmentInstance:
    units = (
        ThermalUnit(
            name="Base-1",
            p_min=20.0,
            p_max=80.0,
            marginal_cost=18.0,
            no_load_cost=120.0,
            startup_cost=350.0,
            min_up=3,
            min_down=2,
            initial_on=True,
            initial_duration=4,
        ),
        ThermalUnit(
            name="Base-2",
            p_min=15.0,
            p_max=60.0,
            marginal_cost=23.0,
            no_load_cost=90.0,
            startup_cost=220.0,
            min_up=2,
            min_down=2,
            initial_on=True,
            initial_duration=3,
        ),
        ThermalUnit(
            name="Mid-1",
            p_min=10.0,
            p_max=45.0,
            marginal_cost=31.0,
            no_load_cost=55.0,
            startup_cost=130.0,
            min_up=2,
            min_down=1,
            initial_on=False,
            initial_duration=2,
        ),
        ThermalUnit(
            name="Peaker",
            p_min=0.0,
            p_max=35.0,
            marginal_cost=52.0,
            no_load_cost=15.0,
            startup_cost=40.0,
            min_up=1,
            min_down=1,
            initial_on=False,
            initial_duration=3,
        ),
    )
    demand = np.asarray([95.0, 110.0, 135.0, 155.0, 145.0, 120.0, 100.0, 85.0])
    return UnitCommitmentInstance(units=units, demand=demand)
