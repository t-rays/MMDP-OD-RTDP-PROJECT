from __future__ import annotations

"""Numerical configuration and shared helpers for the final RTDP experiment."""

import math
from collections.abc import Iterable
from dataclasses import dataclass


class DeadlineReached(RuntimeError):
    """Raised internally when the planning deadline expires."""


def tied_by_ulp(a: float, b: float, *, ulps: int = 8) -> bool:
    if a == b:
        return True
    if not math.isfinite(a) or not math.isfinite(b):
        return False
    tolerance = ulps * max(math.ulp(a), math.ulp(b), math.ulp(1.0))
    return abs(a - b) <= tolerance


def scaled_residual_ratio(
    old_value: float,
    new_value: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    residual = abs(new_value - old_value)
    scale = max(1.0, abs(old_value), abs(new_value))
    allowed = absolute_tolerance + relative_tolerance * scale
    if allowed == 0.0:
        return 0.0 if residual == 0.0 else math.inf
    return residual / allowed


def isolated_success_attempt_bound(
    required_successes: int,
    success_probability: float,
    tail_probability: float,
) -> int:
    """Conservative attempt bound derived from a Chernoff lower-tail bound."""
    if required_successes < 0:
        raise ValueError("required_successes cannot be negative")
    if required_successes == 0:
        return 0
    if not 0.0 < success_probability <= 1.0:
        raise ValueError("success_probability must be in (0, 1]")
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must be in (0, 1)")
    if success_probability == 1.0:
        return required_successes

    failure_threshold = required_successes - 1

    def failure_bound(attempts: int) -> float:
        mean = attempts * success_probability
        if failure_threshold >= mean:
            return 1.0
        return math.exp(-((mean - failure_threshold) ** 2) / (2.0 * mean))

    lower = max(required_successes, math.ceil(required_successes / success_probability))
    upper = lower
    while failure_bound(upper) > tail_probability:
        upper *= 2
    while lower < upper:
        middle = (lower + upper) // 2
        if failure_bound(middle) <= tail_probability:
            upper = middle
        else:
            lower = middle + 1
    return lower


def sequential_multi_agent_step_bound(
    distances: Iterable[float],
    success_probability: float,
    tail_probability: float,
) -> int:
    integer_distances: list[int] = []
    for distance in distances:
        if math.isinf(distance):
            raise ValueError("All distances must be finite")
        if distance < 0:
            raise ValueError("Distances cannot be negative")
        integer_distances.append(math.ceil(distance))
    return max(
        1,
        sum(
            isolated_success_attempt_bound(
                distance,
                success_probability,
                tail_probability,
            )
            for distance in integer_distances
        ),
    )


@dataclass(frozen=True)
class RTDPConfig:
    """Configuration used by both final planners."""

    time_limit_seconds: float = 60.0
    step_tail_probability: float = 1e-6
    epsilon: float = 1e-8
    relative_epsilon: float = 1e-6
    tie_ulps: int = 8
    seed: int = 20260708

    def __post_init__(self) -> None:
        if self.time_limit_seconds <= 0.0:
            raise ValueError("time_limit_seconds must be positive")
        if not 0.0 < self.step_tail_probability < 1.0:
            raise ValueError("step_tail_probability must be in (0, 1)")
        if self.epsilon < 0.0 or self.relative_epsilon < 0.0:
            raise ValueError("residual tolerances cannot be negative")
        if self.tie_ulps <= 0:
            raise ValueError("tie_ulps must be positive")
