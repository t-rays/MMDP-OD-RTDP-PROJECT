from __future__ import annotations

"""RTDP configuration and small shared planning helpers.

The project previously kept numerical comparisons, step-bound calculations,
and control-flow exceptions in three tiny modules.  They are collected here
because all of them support the shared RTDP configuration and stopping logic.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass


class DeadlineReached(RuntimeError):
    """Signal used to stop planning when the time limit is reached."""


class MemoryLimitReached(RuntimeError):
    """Signal used to stop planning at the configured RSS delta."""


def tied_by_ulp(a: float, b: float, *, ulps: int = 8) -> bool:
    """Return True when two finite values differ only by floating-point noise."""
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
    """Return residual divided by a scale-aware admissible tolerance."""
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
    """Return a conservative attempt bound for stochastic movement.

    For ``X ~ Binomial(t, q)``, the implementation uses a Chernoff lower-tail
    bound and returns the smallest ``t`` whose probability of obtaining fewer
    than ``required_successes`` successful moves is at most
    ``tail_probability``.
    """
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

    lower = max(
        required_successes,
        math.ceil(required_successes / success_probability),
    )
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
    """Return the sum of conservative isolated per-agent attempt bounds."""
    integer_distances: list[int] = []
    for distance in distances:
        if math.isinf(distance):
            raise ValueError("All distances must be finite")
        if distance < 0:
            raise ValueError("Distances cannot be negative")
        integer_distances.append(math.ceil(distance))

    total = sum(
        isolated_success_attempt_bound(
            distance,
            success_probability,
            tail_probability,
        )
        for distance in integer_distances
    )
    return max(1, total)


@dataclass(frozen=True)
class RTDPConfig:
    """Configuration shared by Baseline RTDP and OD-RTDP.

    Algorithmic limits are optional. ``step_tail_probability`` replaces a
    fixed step multiplier with a map-derived stochastic tail bound.
    ``memory_limit_mb`` is additional process RSS above the start of planning;
    memory-limited experiments should isolate every run in a subprocess.
    """

    max_trials: int | None = None
    max_steps_per_trial: int | None = None

    # Backward-compatible override. None uses the probabilistic bound.
    step_limit_multiplier: float | None = None
    step_tail_probability: float = 1e-6

    time_limit_seconds: float | None = 60.0
    memory_limit_mb: float | None = None

    # Scale-aware Bellman-residual criterion.
    epsilon: float = 1e-8
    relative_epsilon: float = 1e-6
    stable_trials_required: int = 44
    stop_when_stable: bool = False
    # LRTDP-style stopping: stop when the initial state is labelled solved.
    stop_when_solved: bool = False
    require_goal_for_stability: bool = True

    # None means an ULP-based numerical comparison; a positive value keeps an
    # explicit absolute-tolerance option for sensitivity experiments.
    tie_tolerance: float | None = None
    tie_ulps: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_trials is not None and self.max_trials <= 0:
            raise ValueError("max_trials must be positive or None")
        if self.max_steps_per_trial is not None and self.max_steps_per_trial <= 0:
            raise ValueError("max_steps_per_trial must be positive or None")
        if self.step_limit_multiplier is not None and self.step_limit_multiplier <= 0.0:
            raise ValueError("step_limit_multiplier must be positive or None")
        if not 0.0 < self.step_tail_probability < 1.0:
            raise ValueError("step_tail_probability must be in (0, 1)")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0.0:
            raise ValueError("time_limit_seconds must be positive or None")
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0.0:
            raise ValueError("memory_limit_mb must be positive or None")
        if self.epsilon < 0.0 or self.relative_epsilon < 0.0:
            raise ValueError("residual tolerances cannot be negative")
        if self.stable_trials_required <= 0:
            raise ValueError("stable_trials_required must be positive")
        if self.tie_tolerance is not None and self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance cannot be negative or None")
        if self.tie_ulps <= 0:
            raise ValueError("tie_ulps must be positive")
        if (
            self.max_trials is None
            and self.time_limit_seconds is None
            and self.memory_limit_mb is None
            and not self.stop_when_stable
            and not self.stop_when_solved
        ):
            raise ValueError(
                "At least one stopping mechanism must be enabled: max_trials, "
                "time_limit_seconds, memory_limit_mb, stop_when_stable, "
                "or stop_when_solved"
            )
