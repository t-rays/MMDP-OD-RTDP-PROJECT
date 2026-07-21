from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RTDPConfig:
    """Configuration shared by Baseline RTDP and OD-RTDP.

    Algorithmic limits are optional.  ``step_tail_probability`` replaces the
    former fixed 5x step multiplier with a map-derived stochastic tail bound.
    ``memory_limit_mb`` is additional process RSS above the start of planning;
    final memory-limited experiments should isolate each run in a subprocess.
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
    # This is the preferred rule for run-to-convergence experiments.
    stop_when_solved: bool = False
    require_goal_for_stability: bool = True

    # None means an ULP-based numerical comparison; a positive value keeps the
    # old explicit absolute-tolerance behavior for sensitivity experiments.
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
