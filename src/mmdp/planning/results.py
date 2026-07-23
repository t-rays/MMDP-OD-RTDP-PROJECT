from __future__ import annotations

"""Planning result returned by the RTDP engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningResult:
    stop_reason: str
    elapsed_seconds: float
    peak_rss_delta_mb: float
