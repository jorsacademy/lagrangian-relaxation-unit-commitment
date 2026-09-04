from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import ThermalUnit


@dataclass(frozen=True)
class CommitmentPattern:
    on: tuple[int, ...]
    startups: tuple[int, ...]

    @property
    def horizon(self) -> int:
        return len(self.on)


def _transition_allowed(unit: ThermalUnit, previous_on: bool, duration: int, next_on: bool) -> bool:
    if previous_on == next_on:
        return True
    if previous_on and not next_on:
        return duration >= unit.min_up
    return duration >= unit.min_down


def enumerate_commitment_patterns(unit: ThermalUnit, horizon: int) -> tuple[CommitmentPattern, ...]:
    if horizon < 1:
        raise ValueError("horizon must be positive")

    patterns: list[CommitmentPattern] = []

    def visit(
        t: int,
        previous_on: bool,
        duration: int,
        on: list[int],
        startups: list[int],
    ) -> None:
        if t == horizon:
            patterns.append(CommitmentPattern(tuple(on), tuple(startups)))
            return
        for next_on in (False, True):
            if not _transition_allowed(unit, previous_on, duration, next_on):
                continue
            startup = int((not previous_on) and next_on)
            next_duration = duration + 1 if previous_on == next_on else 1
            on.append(int(next_on))
            startups.append(startup)
            visit(t + 1, next_on, next_duration, on, startups)
            on.pop()
            startups.pop()

    visit(0, unit.initial_on, unit.initial_duration, [], [])
    return tuple(patterns)


def pattern_fixed_cost(unit: ThermalUnit, pattern: CommitmentPattern) -> float:
    on = np.asarray(pattern.on, dtype=float)
    startups = np.asarray(pattern.startups, dtype=float)
    return float(unit.no_load_cost * on.sum() + unit.startup_cost * startups.sum())
