from __future__ import annotations

"""Small statistical helpers for experiment-size and stability defaults."""

import math
from statistics import NormalDist


def binomial_worst_case_sample_size(
    *, confidence: float, half_width: float
) -> int:
    """Normal-approximation sample size for a worst-case proportion p=0.5.

    This makes the default number of evaluation episodes interpretable: the
    requested confidence interval has approximately ``half_width`` precision
    in the worst case.  Exact intervals are still recommended in analysis.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not 0.0 < half_width < 1.0:
        raise ValueError("half_width must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return max(1, math.ceil((z * z * 0.25) / (half_width * half_width)))


def consecutive_trials_for_detection(
    *, confidence: float, minimum_event_probability: float
) -> int:
    """Trials needed to see at least one event with the given confidence.

    If an unstable trial would occur independently with probability at least
    ``minimum_event_probability``, observing this many consecutive stable
    trials misses it with probability at most ``1-confidence``.  RTDP trials
    are not truly independent, so this remains an empirical stopping rule, but
    its parameters are explicit rather than a bare fixed streak length.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not 0.0 < minimum_event_probability < 1.0:
        raise ValueError("minimum_event_probability must be in (0, 1)")
    return max(
        1,
        math.ceil(
            math.log(1.0 - confidence)
            / math.log(1.0 - minimum_event_probability)
        ),
    )
