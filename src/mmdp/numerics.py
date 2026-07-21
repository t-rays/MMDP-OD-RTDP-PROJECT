from __future__ import annotations

"""Scale-aware numerical comparisons shared by both planners."""

import math


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
