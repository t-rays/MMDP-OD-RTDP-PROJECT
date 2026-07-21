from __future__ import annotations

"""Result dataclasses produced by RTDP trials and complete planning runs."""

from dataclasses import dataclass
from typing import Generic, TypeVar

StateType = TypeVar("StateType")


@dataclass(frozen=True)
class TrialResult(Generic[StateType]):
    """Outcome of one forward-simulation trial.

    ``steps`` counts real environment steps for both algorithms; for OD-RTDP a
    step is completed only when a full joint action has been assembled.
    ``visited_states`` holds the states of the domain's own search space
    (complete states for baseline, OD states for operator decomposition).
    """

    maximum_residual: float
    maximum_scaled_residual: float
    steps: int
    reached_goal: bool
    reached_solved_state: bool
    hit_step_limit: bool
    visited_states: tuple[StateType, ...]


@dataclass(frozen=True)
class BasePlanningResult:
    """Fields common to both Baseline and OD planning results."""

    stop_reason: str
    trials_completed: int
    goal_reaching_trials: int
    step_limited_trials: int
    total_trial_steps: int
    elapsed_seconds: float
    bellman_backups: int
    transition_outcomes_evaluated: int
    final_trial_residual: float
    final_trial_scaled_residual: float
    consecutive_stable_trials: int
    maximum_consecutive_stable_trials: int
    stability_criterion_reached: bool
    first_stability_trial: int | None
    first_stability_elapsed_seconds: float | None
    initial_state_solved: bool
    solved_checks: int
    first_solved_trial: int | None
    first_solved_elapsed_seconds: float | None
    resolved_max_steps_per_trial: int | None
    memory_limit_mb: float | None
    memory_limit_reached: bool
    baseline_rss_mb: float
    peak_rss_mb: float
    peak_rss_delta_mb: float


@dataclass(frozen=True)
class RTDPPlanningResult(BasePlanningResult):
    planning_action_evaluations: int
    visited_states: int
    solved_states: int


@dataclass(frozen=True)
class ODRTDPPlanningResult(BasePlanningResult):
    planning_operator_evaluations: int
    complete_joint_actions_evaluated: int
    visited_od_states: int
    visited_real_states: int
    solved_od_states: int
    solved_real_states: int
