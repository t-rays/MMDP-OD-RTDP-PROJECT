from __future__ import annotations

"""CSV schema, run identity, and row construction for experiment runs.

One experiment condition produces exactly one wide CSV row.  ``make_run_id``
hashes the complete condition so finished rows can be skipped on resume.
"""

import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmdp.planning.config import RTDPConfig
from mmdp.evaluation import EvaluationConfig
from mmdp.domain.grid_mmdp import GridMMDP, MMDPConfig
from mmdp.domain.heuristic import ShortestPathHeuristic
from mmdp.domain.map_creator import MapInstance

IMPLEMENTATION_VERSION = "focused-metrics-v16-clean-architecture"


CSV_FIELDS = (
    # Run identity and status.
    "run_id",
    "implementation_version",
    "resource_mode",
    "resource_profile_name",
    "master_seed",
    "seed_index",
    "status",
    "error_type",
    "error_message",
    "created_at_utc",

    # Problem identity.
    "map_name",
    "map_folder",
    "scenario_file",
    "scenario_number",
    "task_offset",
    "n_agents",
    "starts",
    "goals",
    "algorithm",

    # Randomness.
    "planning_seed",
    "evaluation_seed",

    # Environment configuration.
    "slip_probability",
    "freeze_agents_at_goal",
    "reject_conflicting_transitions",
    "transition_cache_max_entries",
    "memory_limit_mb",

    # Planning configuration.
    "max_trials",
    "max_steps_per_trial",
    "step_limit_multiplier",
    "step_tail_probability",
    "step_cap_familywise_error",
    "time_limit_seconds",
    "epsilon",
    "relative_epsilon",
    "stable_trials_required",
    "stability_confidence",
    "minimum_unstable_trial_rate",
    "stop_when_stable",
    "stop_when_solved",
    "tie_tolerance",
    "require_goal_for_stability",

    # Evaluation configuration.
    "evaluation_episodes",
    "evaluation_confidence",
    "evaluation_half_width",
    "evaluation_max_steps_per_episode",
    "measure_conflict_risk",
    "randomize_greedy_ties",
    "evaluation_cache_only_executed_actions",
    "evaluation_collect_diagnostics",
    "initial_longest_shortest_path",
    "initial_longest_expected_isolated_steps",
    "evaluation_resolved_step_cap",
    "evaluation_step_cap_to_expected_ratio",
    "evaluation_step_cap_warning",

    # Common planning results.
    "planning_stop_reason",
    "planning_trials_completed",
    "planning_goal_reaching_trials",
    "planning_step_limited_trials",
    "planning_total_real_steps",
    "planning_elapsed_seconds",
    "planning_bellman_backups",
    "planning_transition_outcomes_evaluated",
    "planning_final_trial_residual",
    "planning_final_trial_scaled_residual",
    "planning_consecutive_stable_trials",
    "planning_maximum_consecutive_stable_trials",
    "planning_stability_criterion_reached",
    "planning_first_stability_trial",
    "planning_first_stability_elapsed_seconds",
    "planning_initial_state_solved",
    "planning_solved_states",
    "planning_solved_od_states",
    "planning_solved_real_states",
    "planning_solved_checks",
    "planning_first_solved_trial",
    "planning_first_solved_elapsed_seconds",
    "planning_resolved_max_steps_per_trial",
    "planning_memory_limit_reached",
    "planning_baseline_rss_mb",
    "planning_peak_rss_mb",
    "planning_peak_rss_delta_mb",

    # Baseline-specific planning results.
    "planning_action_evaluations",
    "planning_visited_states",

    # OD-specific planning results.
    "planning_operator_evaluations",
    "planning_complete_joint_actions_evaluated",
    "planning_visited_od_states",
    "planning_visited_real_states",

    # Evaluation results.
    "evaluation_policy_name",
    "evaluation_successful_episodes",
    "evaluation_failed_episodes",
    "evaluation_success_rate",
    "evaluation_total_environment_steps",
    "evaluation_mean_steps_all_episodes",
    "evaluation_mean_steps_successful_episodes",
    "evaluation_mean_accumulated_cost_all_episodes",
    "evaluation_mean_sum_of_costs_successful_episodes",
    "evaluation_std_sum_of_costs_successful_episodes",
    "evaluation_mean_makespan_successful_episodes",
    "evaluation_std_makespan_successful_episodes",
    "evaluation_per_agent_arrival_rates",
    "evaluation_per_agent_mean_arrival_times",
    "evaluation_mean_arrived_agents_per_episode",
    "evaluation_expected_conflict_attempts_per_episode",
    "evaluation_mean_conflict_risk_per_environment_step",
    "evaluation_mean_expected_self_loop_probability_per_step",
    "evaluation_mean_expected_shortest_path_progress_per_step",
    "evaluation_mean_expected_vertex_conflict_probability_per_step",
    "evaluation_mean_expected_edge_swap_probability_per_step",
    "evaluation_mean_expected_noncollision_no_motion_probability_per_step",
    "evaluation_mean_selected_unfinished_stay_actions_per_episode",
    "evaluation_mean_selected_unfinished_blocked_actions_per_episode",
    "evaluation_deterministic_self_loop_failures",
    "evaluation_step_limit_failures",
    "evaluation_elapsed_seconds",
    "evaluation_baseline_rss_mb",
    "evaluation_peak_rss_mb",
    "evaluation_peak_rss_delta_mb",
    "overall_peak_rss_mb",
    "overall_peak_rss_delta_from_planning_baseline_mb",
    "evaluation_total_policy_decision_seconds",
    "evaluation_mean_policy_decision_milliseconds",
    "evaluation_policy_cache_hits",
    "evaluation_policy_cache_misses",
    "evaluation_policy_cache_entries",
    "evaluation_policy_cache_hit_rate",
    "evaluation_transition_raw_cache_entries_before",
    "evaluation_transition_raw_cache_entries_after",
    "evaluation_transition_resolved_cache_entries_before",
    "evaluation_transition_resolved_cache_entries_after",
    "evaluation_transition_raw_cache_hits",
    "evaluation_transition_raw_cache_misses",
    "evaluation_transition_raw_cache_writes",
    "evaluation_transition_raw_cache_evictions",
    "evaluation_transition_resolved_cache_hits",
    "evaluation_transition_resolved_cache_misses",
    "evaluation_transition_resolved_cache_writes",
    "evaluation_transition_resolved_cache_evictions",
    "evaluation_mean_unique_states_visited",
    "evaluation_mean_repeated_state_visits",
    "evaluation_mean_maximum_state_visit_count",
    "evaluation_mean_self_transitions",
    "evaluation_mean_maximum_consecutive_self_transitions",
    "evaluation_mean_tie_decisions",
    "evaluation_mean_unique_real_states_with_policy_ties",
    "evaluation_mean_tie_decisions_per_environment_step",

    "diagnostics_file",
    "od_global_diagnostic_success_rate",
    "od_global_diagnostic_mean_cost",
    "od_global_diagnostic_mean_makespan",
    "od_global_diagnostic_summary_json",

    # Complete serialized outputs for future inspection.
    "planning_result_json",
    "evaluation_summary_json",
)


def json_text(value: Any) -> str:
    """Serialize tuples and dataclass dictionaries consistently."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _condition_payload(
    *,
    map_folder: Path,
    n_agents: int,
    scenario_number: int,
    task_offset: int,
    algorithm: str,
    planning_seed: int,
    evaluation_seed: int,
    mdp_config: MMDPConfig,
    planning_config: RTDPConfig,
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    """Return every setting that defines one reproducible experiment run."""
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "map_folder": str(map_folder.resolve()),
        "n_agents": n_agents,
        "scenario_number": scenario_number,
        "task_offset": task_offset,
        "algorithm": algorithm,
        "planning_seed": planning_seed,
        "evaluation_seed": evaluation_seed,
        "mdp_config": asdict(mdp_config),
        "planning_config": asdict(planning_config),
        "evaluation_config": asdict(evaluation_config),
    }


def make_run_id(**condition_arguments: Any) -> str:
    """Create a deterministic identifier for one complete experimental condition."""
    payload = _condition_payload(**condition_arguments)
    encoded = json_text(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def base_row(
    *,
    run_id: str,
    map_folder: Path,
    instance: MapInstance | None,
    n_agents: int,
    scenario_number: int,
    task_offset: int,
    algorithm: str,
    planning_seed: int,
    evaluation_seed: int,
    mdp_config: MMDPConfig,
    planning_config: RTDPConfig,
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    """Build the metadata shared by successful and failed rows."""
    row: dict[str, Any] = {field: "" for field in CSV_FIELDS}

    row.update(
        {
            "run_id": run_id,
            "implementation_version": IMPLEMENTATION_VERSION,
            "status": "running",
            "created_at_utc": utc_timestamp(),
            "map_name": (
                instance.grid_map.name if instance is not None else map_folder.name
            ),
            "map_folder": str(map_folder.resolve()),
            "scenario_file": (
                instance.scenario_file.name if instance is not None else ""
            ),
            "scenario_number": scenario_number,
            "task_offset": task_offset,
            "n_agents": n_agents,
            "starts": json_text(instance.starts) if instance is not None else "",
            "goals": json_text(instance.goals) if instance is not None else "",
            "algorithm": algorithm,
            "planning_seed": planning_seed,
            "evaluation_seed": evaluation_seed,
            "slip_probability": mdp_config.slip_to_stay_probability,
            "freeze_agents_at_goal": mdp_config.freeze_agents_at_goal,
            "reject_conflicting_transitions": (
                mdp_config.reject_conflicting_transitions
            ),
            "transition_cache_max_entries": (
                mdp_config.transition_cache_max_entries
            ),
            "memory_limit_mb": planning_config.memory_limit_mb,
            "max_trials": planning_config.max_trials,
            "max_steps_per_trial": planning_config.max_steps_per_trial,
            "step_limit_multiplier": planning_config.step_limit_multiplier,
            "step_tail_probability": planning_config.step_tail_probability,
            "step_cap_familywise_error": getattr(
                planning_config, "step_cap_familywise_error", ""
            ),
            "time_limit_seconds": planning_config.time_limit_seconds,
            "epsilon": planning_config.epsilon,
            "relative_epsilon": planning_config.relative_epsilon,
            "stable_trials_required": planning_config.stable_trials_required,
            "stability_confidence": "",
            "minimum_unstable_trial_rate": "",
            "stop_when_stable": planning_config.stop_when_stable,
            "stop_when_solved": planning_config.stop_when_solved,
            "tie_tolerance": planning_config.tie_tolerance,
            "require_goal_for_stability": (
                planning_config.require_goal_for_stability
            ),
            "evaluation_episodes": evaluation_config.episodes,
            "evaluation_confidence": "",
            "evaluation_half_width": "",
            "evaluation_max_steps_per_episode": (
                evaluation_config.max_steps_per_episode
            ),
            "measure_conflict_risk": evaluation_config.measure_conflict_risk,
            "randomize_greedy_ties": evaluation_config.randomize_greedy_ties,
            "evaluation_cache_only_executed_actions": (
                evaluation_config.cache_only_executed_actions
            ),
            "evaluation_collect_diagnostics": (
                evaluation_config.collect_diagnostics
            ),
        }
    )

    return row


def add_planning_result(
    row: dict[str, Any],
    *,
    algorithm: str,
    planning_result: Any,
) -> None:
    """Add common and algorithm-specific planning fields to one CSV row."""
    result_dict = asdict(planning_result)

    row.update(
        {
            "planning_stop_reason": result_dict["stop_reason"],
            "planning_trials_completed": result_dict["trials_completed"],
            "planning_goal_reaching_trials": result_dict["goal_reaching_trials"],
            "planning_step_limited_trials": result_dict["step_limited_trials"],
            "planning_total_real_steps": result_dict["total_trial_steps"],
            "planning_elapsed_seconds": result_dict["elapsed_seconds"],
            "planning_bellman_backups": result_dict["bellman_backups"],
            "planning_transition_outcomes_evaluated": result_dict[
                "transition_outcomes_evaluated"
            ],
            "planning_final_trial_residual": result_dict["final_trial_residual"],
            "planning_final_trial_scaled_residual": result_dict[
                "final_trial_scaled_residual"
            ],
            "planning_consecutive_stable_trials": result_dict[
                "consecutive_stable_trials"
            ],
            "planning_maximum_consecutive_stable_trials": result_dict[
                "maximum_consecutive_stable_trials"
            ],
            "planning_stability_criterion_reached": result_dict[
                "stability_criterion_reached"
            ],
            "planning_first_stability_trial": result_dict["first_stability_trial"],
            "planning_first_stability_elapsed_seconds": result_dict[
                "first_stability_elapsed_seconds"
            ],
            "planning_initial_state_solved": result_dict["initial_state_solved"],
            "planning_solved_states": result_dict.get("solved_states"),
            "planning_solved_od_states": result_dict.get("solved_od_states"),
            "planning_solved_real_states": result_dict.get("solved_real_states"),
            "planning_solved_checks": result_dict["solved_checks"],
            "planning_first_solved_trial": result_dict["first_solved_trial"],
            "planning_first_solved_elapsed_seconds": result_dict[
                "first_solved_elapsed_seconds"
            ],
            "planning_resolved_max_steps_per_trial": result_dict[
                "resolved_max_steps_per_trial"
            ],
            "planning_memory_limit_reached": result_dict["memory_limit_reached"],
            "planning_baseline_rss_mb": result_dict["baseline_rss_mb"],
            "planning_peak_rss_mb": result_dict["peak_rss_mb"],
            "planning_peak_rss_delta_mb": result_dict["peak_rss_delta_mb"],
            "planning_result_json": json_text(result_dict),
        }
    )

    if algorithm == "baseline":
        row.update(
            {
                "planning_action_evaluations": result_dict[
                    "planning_action_evaluations"
                ],
                "planning_visited_states": result_dict["visited_states"],
            }
        )
    elif algorithm == "od":
        row.update(
            {
                "planning_operator_evaluations": result_dict[
                    "planning_operator_evaluations"
                ],
                "planning_complete_joint_actions_evaluated": result_dict[
                    "complete_joint_actions_evaluated"
                ],
                "planning_visited_od_states": result_dict["visited_od_states"],
                "planning_visited_real_states": result_dict["visited_real_states"],
            }
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")


def add_evaluation_summary(
    row: dict[str, Any],
    evaluation_summary: Any,
) -> None:
    """Add aggregate fixed-policy evaluation fields to one CSV row."""
    summary = asdict(evaluation_summary)

    row.update(
        {
            "evaluation_policy_name": summary["policy_name"],
            "evaluation_successful_episodes": summary["successful_episodes"],
            "evaluation_failed_episodes": summary["failed_episodes"],
            "evaluation_success_rate": summary["success_rate"],
            "evaluation_total_environment_steps": summary[
                "total_environment_steps"
            ],
            "evaluation_mean_steps_all_episodes": summary[
                "mean_steps_all_episodes"
            ],
            "evaluation_mean_steps_successful_episodes": summary[
                "mean_steps_successful_episodes"
            ],
            "evaluation_mean_accumulated_cost_all_episodes": summary[
                "mean_accumulated_cost_all_episodes"
            ],
            "evaluation_mean_sum_of_costs_successful_episodes": summary[
                "mean_sum_of_costs_successful_episodes"
            ],
            "evaluation_std_sum_of_costs_successful_episodes": summary[
                "std_sum_of_costs_successful_episodes"
            ],
            "evaluation_mean_makespan_successful_episodes": summary[
                "mean_makespan_successful_episodes"
            ],
            "evaluation_std_makespan_successful_episodes": summary[
                "std_makespan_successful_episodes"
            ],
            "evaluation_per_agent_arrival_rates": json_text(
                summary["per_agent_arrival_rates"]
            ),
            "evaluation_per_agent_mean_arrival_times": json_text(
                summary["per_agent_mean_arrival_times"]
            ),
            "evaluation_mean_arrived_agents_per_episode": summary[
                "mean_arrived_agents_per_episode"
            ],
            "evaluation_expected_conflict_attempts_per_episode": summary[
                "expected_conflict_attempts_per_episode"
            ],
            "evaluation_mean_conflict_risk_per_environment_step": summary[
                "mean_conflict_risk_per_environment_step"
            ],
            "evaluation_mean_expected_self_loop_probability_per_step": summary[
                "mean_expected_self_loop_probability_per_step"
            ],
            "evaluation_mean_expected_shortest_path_progress_per_step": summary[
                "mean_expected_shortest_path_progress_per_step"
            ],
            "evaluation_mean_expected_vertex_conflict_probability_per_step": summary[
                "mean_expected_vertex_conflict_probability_per_step"
            ],
            "evaluation_mean_expected_edge_swap_probability_per_step": summary[
                "mean_expected_edge_swap_probability_per_step"
            ],
            "evaluation_mean_expected_noncollision_no_motion_probability_per_step": summary[
                "mean_expected_noncollision_no_motion_probability_per_step"
            ],
            "evaluation_mean_selected_unfinished_stay_actions_per_episode": summary[
                "mean_selected_unfinished_stay_actions_per_episode"
            ],
            "evaluation_mean_selected_unfinished_blocked_actions_per_episode": summary[
                "mean_selected_unfinished_blocked_actions_per_episode"
            ],
            "evaluation_deterministic_self_loop_failures": summary[
                "deterministic_self_loop_failures"
            ],
            "evaluation_step_limit_failures": summary["step_limit_failures"],
            "evaluation_elapsed_seconds": summary["evaluation_elapsed_seconds"],
            "evaluation_baseline_rss_mb": summary["evaluation_baseline_rss_mb"],
            "evaluation_peak_rss_mb": summary["evaluation_peak_rss_mb"],
            "evaluation_peak_rss_delta_mb": summary["evaluation_peak_rss_delta_mb"],
            "evaluation_total_policy_decision_seconds": summary[
                "total_policy_decision_seconds"
            ],
            "evaluation_mean_policy_decision_milliseconds": summary[
                "mean_policy_decision_milliseconds"
            ],
            "evaluation_policy_cache_hits": summary["policy_cache_hits"],
            "evaluation_policy_cache_misses": summary["policy_cache_misses"],
            "evaluation_policy_cache_entries": summary["policy_cache_entries"],
            "evaluation_policy_cache_hit_rate": summary["policy_cache_hit_rate"],
            "evaluation_transition_raw_cache_entries_before": summary[
                "transition_raw_cache_entries_before"
            ],
            "evaluation_transition_raw_cache_entries_after": summary[
                "transition_raw_cache_entries_after"
            ],
            "evaluation_transition_resolved_cache_entries_before": summary[
                "transition_resolved_cache_entries_before"
            ],
            "evaluation_transition_resolved_cache_entries_after": summary[
                "transition_resolved_cache_entries_after"
            ],
            "evaluation_transition_raw_cache_hits": summary[
                "transition_raw_cache_hits"
            ],
            "evaluation_transition_raw_cache_misses": summary[
                "transition_raw_cache_misses"
            ],
            "evaluation_transition_raw_cache_writes": summary[
                "transition_raw_cache_writes"
            ],
            "evaluation_transition_raw_cache_evictions": summary[
                "transition_raw_cache_evictions"
            ],
            "evaluation_transition_resolved_cache_hits": summary[
                "transition_resolved_cache_hits"
            ],
            "evaluation_transition_resolved_cache_misses": summary[
                "transition_resolved_cache_misses"
            ],
            "evaluation_transition_resolved_cache_writes": summary[
                "transition_resolved_cache_writes"
            ],
            "evaluation_transition_resolved_cache_evictions": summary[
                "transition_resolved_cache_evictions"
            ],
            "evaluation_mean_unique_states_visited": summary[
                "mean_unique_states_visited"
            ],
            "evaluation_mean_repeated_state_visits": summary[
                "mean_repeated_state_visits"
            ],
            "evaluation_mean_maximum_state_visit_count": summary[
                "mean_maximum_state_visit_count"
            ],
            "evaluation_mean_self_transitions": summary["mean_self_transitions"],
            "evaluation_mean_maximum_consecutive_self_transitions": summary[
                "mean_maximum_consecutive_self_transitions"
            ],
            "evaluation_mean_tie_decisions": summary["mean_tie_decisions"],
            "evaluation_mean_unique_real_states_with_policy_ties": summary[
                "mean_unique_real_states_with_policy_ties"
            ],
            "evaluation_mean_tie_decisions_per_environment_step": summary[
                "mean_tie_decisions_per_environment_step"
            ],
            "evaluation_summary_json": json_text(summary),
        }
    )


def add_step_cap_diagnostics(
    row: dict[str, Any],
    *,
    mdp: GridMMDP,
    heuristic: ShortestPathHeuristic,
    planner: Any,
    evaluation_config: EvaluationConfig,
) -> bool:
    """Record whether an explicit evaluation cap is likely to truncate runs.

    The expected isolated travel time ``d / q`` is not a hard completion bound,
    but a cap below it makes zero-success results difficult to interpret.  The
    planner's automatic cap is used when evaluation has no explicit cap.
    """
    distances = heuristic.distance_summary(mdp.initial_state())
    longest_distance = max(distances, default=0.0)
    q = 1.0 - mdp.config.slip_to_stay_probability
    expected_isolated = (
        longest_distance / q if q > 0.0 else float("inf")
    )
    resolved_cap = (
        evaluation_config.max_steps_per_episode
        if evaluation_config.max_steps_per_episode is not None
        else planner.resolved_max_steps_per_trial
    )
    ratio = (
        resolved_cap / expected_isolated
        if resolved_cap is not None and expected_isolated > 0.0
        else None
    )
    warning = bool(
        resolved_cap is not None
        and math.isfinite(expected_isolated)
        and resolved_cap < math.ceil(expected_isolated)
    )

    row.update(
        {
            "initial_longest_shortest_path": longest_distance,
            "initial_longest_expected_isolated_steps": expected_isolated,
            "evaluation_resolved_step_cap": resolved_cap,
            "evaluation_step_cap_to_expected_ratio": ratio,
            "evaluation_step_cap_warning": warning,
        }
    )
    return warning


def normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep the output schema stable and convert None to an empty CSV field."""
    return {
        field: ("" if row.get(field) is None else row.get(field, ""))
        for field in CSV_FIELDS
    }
