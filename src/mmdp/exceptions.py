from __future__ import annotations

"""Control-flow signals shared by the RTDP planner and its domains."""


class DeadlineReached(RuntimeError):
    """Signal used to stop planning when the time limit is reached."""


class MemoryLimitReached(RuntimeError):
    """Signal used to stop planning at the configured RSS delta."""
