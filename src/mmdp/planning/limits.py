from __future__ import annotations

"""Derived limits used by planning and policy evaluation.

The old implementation used a fixed ``5 * expected_distance`` multiplier.
This module replaces that hidden constant with an interpretable tail-probability
bound.  For an isolated agent that needs ``d`` successful moves and succeeds
with probability ``q`` on each attempt, we choose the smallest number of
attempts whose Chernoff upper bound on failing to obtain ``d`` successes is at
most ``tail_probability``.

For several agents, the default sequential bound sums the individual bounds.
This is intentionally conservative: it allows enough steps for the agents to
complete one after another, and cycle diagnostics distinguish an improper
policy from a merely slow stochastic execution.
"""

import math
from collections.abc import Iterable


def isolated_success_attempt_bound(
    required_successes: int,
    success_probability: float,
    tail_probability: float,
) -> int:
    """Return a dependency-free conservative attempt bound.

    The Chernoff lower-tail inequality for ``X ~ Binomial(t, q)`` is used:

        P[X <= k] <= exp(-(t*q-k)^2 / (2*t*q))

    with ``k = required_successes - 1``.  The returned integer is the smallest
    ``t`` for which this bound is no larger than ``tail_probability``.
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
        return math.exp(
            -((mean - failure_threshold) ** 2) / (2.0 * mean)
        )

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
    """Return the sum of isolated per-agent stochastic attempt bounds."""
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
