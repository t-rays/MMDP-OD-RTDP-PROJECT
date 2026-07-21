from __future__ import annotations

# Allow this script to be executed from the repository root without installing a package.
from pathlib import Path as _PathForSysPath
import sys as _sys_for_path
_PROJECT_ROOT_FOR_IMPORTS = _PathForSysPath(__file__).resolve().parents[1]
_SRC_FOR_IMPORTS = _PROJECT_ROOT_FOR_IMPORTS / "src"
if str(_SRC_FOR_IMPORTS) not in _sys_for_path.path:
    _sys_for_path.path.insert(0, str(_SRC_FOR_IMPORTS))

"""
Automated experiment runner for Baseline RTDP and OD-RTDP.

For every requested combination of:

    map folder
    x number of agents
    x planning seed
    x algorithm

the script:

1. Creates the same MapInstance for both algorithms.
2. Creates the stochastic MMDP.
3. Builds the shortest-path heuristic.
4. Runs planner.solve().
5. Evaluates the resulting fixed policy.
6. Writes one row to a CSV file immediately.

The CSV can be resumed safely. A deterministic run_id is calculated from the
complete experimental condition and configuration. Existing run_ids are skipped
unless --overwrite is supplied.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

from baseline_rtdp import BaselineRTDP, RTDPConfig
from evaluation import (
    EvaluationConfig,
    MethodPolicyAdapter,
    evaluate_policy,
)
from grid_mmdp import GridMMDP, MMDPConfig
from heuristic import ShortestPathHeuristic
from map_creator import MapInstance, create_map_instance
from od_rtdp import OperatorDecompositionRTDP
from statistics_utils import (
    binomial_worst_case_sample_size,
    consecutive_trials_for_detection,
)
from experiment_profiles import (
    RESOURCE_MODES,
    load_profile,
    resolve_resource_limits,
)


ALGORITHMS = ("baseline", "od")
IMPLEMENTATION_VERSION = "focused-metrics-v14-bounded-evaluation"


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


def parse_optional_int(text: str) -> int | None:
    """Parse an integer or the word 'none'."""
    normalized = text.strip().lower()

    if normalized in {"none", "null", "off", "disabled"}:
        return None

    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an integer or 'none', received {text!r}"
        ) from exc

    if value <= 0:
        raise argparse.ArgumentTypeError(
            "The value must be positive or 'none'"
        )

    return value


def parse_optional_nonnegative_int(text: str) -> int | None:
    """Parse a non-negative integer or the word 'none'."""
    normalized = text.strip().lower()

    if normalized in {"none", "null", "off", "disabled"}:
        return None

    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative integer or 'none', received {text!r}"
        ) from exc

    if value < 0:
        raise argparse.ArgumentTypeError(
            "The value must be non-negative or 'none'"
        )

    return value


def parse_optional_float(text: str) -> float | None:
    """Parse a positive float or the word 'none'."""
    normalized = text.strip().lower()

    if normalized in {"none", "null", "off", "disabled"}:
        return None

    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a number or 'none', received {text!r}"
        ) from exc

    if value <= 0.0:
        raise argparse.ArgumentTypeError(
            "The value must be positive or 'none'"
        )

    return value


def parse_optional_nonnegative_float(text: str) -> float | None:
    normalized = text.strip().lower()
    if normalized in {"none", "null", "off", "disabled"}:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative number or 'none', received {text!r}"
        ) from exc
    if value < 0.0:
        raise argparse.ArgumentTypeError(
            "The value must be non-negative or 'none'"
        )
    return value


def _json_text(value: Any) -> str:
    """Serialize tuples and dataclass dictionaries consistently."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _utc_timestamp() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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
    """
    Return every setting that defines one reproducible experiment run.
    """
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


def make_run_id(
    **condition_arguments: Any,
) -> str:
    """
    Create a deterministic identifier for one complete experimental condition.
    """
    payload = _condition_payload(
        **condition_arguments
    )

    encoded = _json_text(
        payload
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()[:20]


def _base_row(
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
    """
    Build the metadata shared by successful and failed rows.
    """
    row: dict[str, Any] = {
        field: ""
        for field in CSV_FIELDS
    }

    row.update(
        {
            "run_id": run_id,
            "implementation_version": IMPLEMENTATION_VERSION,
            "status": "running",
            "created_at_utc": _utc_timestamp(),
            "map_name": (
                instance.grid_map.name
                if instance is not None
                else map_folder.name
            ),
            "map_folder": str(
                map_folder.resolve()
            ),
            "scenario_file": (
                instance.scenario_file.name
                if instance is not None
                else ""
            ),
            "scenario_number": scenario_number,
            "task_offset": task_offset,
            "n_agents": n_agents,
            "starts": (
                _json_text(instance.starts)
                if instance is not None
                else ""
            ),
            "goals": (
                _json_text(instance.goals)
                if instance is not None
                else ""
            ),
            "algorithm": algorithm,
            "planning_seed": planning_seed,
            "evaluation_seed": evaluation_seed,
            "slip_probability": (
                mdp_config.slip_to_stay_probability
            ),
            "freeze_agents_at_goal": (
                mdp_config.freeze_agents_at_goal
            ),
            "reject_conflicting_transitions": (
                mdp_config.reject_conflicting_transitions
            ),
            "transition_cache_max_entries": (
                mdp_config.transition_cache_max_entries
            ),
            "memory_limit_mb": planning_config.memory_limit_mb,
            "max_trials": planning_config.max_trials,
            "max_steps_per_trial": (
                planning_config.max_steps_per_trial
            ),
            "step_limit_multiplier": (
                planning_config.step_limit_multiplier
            ),
            "step_tail_probability": planning_config.step_tail_probability,
            "step_cap_familywise_error": getattr(planning_config, "step_cap_familywise_error", ""),
            "time_limit_seconds": (
                planning_config.time_limit_seconds
            ),
            "epsilon": planning_config.epsilon,
            "relative_epsilon": planning_config.relative_epsilon,
            "stable_trials_required": (
                planning_config.stable_trials_required
            ),
            "stability_confidence": "",
            "minimum_unstable_trial_rate": "",
            "stop_when_stable": planning_config.stop_when_stable,
            "stop_when_solved": planning_config.stop_when_solved,
            "tie_tolerance": (
                planning_config.tie_tolerance
            ),
            "require_goal_for_stability": (
                planning_config.require_goal_for_stability
            ),
            "evaluation_episodes": (
                evaluation_config.episodes
            ),
            "evaluation_confidence": "",
            "evaluation_half_width": "",
            "evaluation_max_steps_per_episode": (
                evaluation_config.max_steps_per_episode
            ),
            "measure_conflict_risk": (
                evaluation_config.measure_conflict_risk
            ),
            "randomize_greedy_ties": (
                evaluation_config.randomize_greedy_ties
            ),
            "evaluation_cache_only_executed_actions": (
                evaluation_config.cache_only_executed_actions
            ),
            "evaluation_collect_diagnostics": (
                evaluation_config.collect_diagnostics
            ),
        }
    )

    return row


def _add_planning_result(
    row: dict[str, Any],
    *,
    algorithm: str,
    planning_result: Any,
) -> None:
    """Add common and algorithm-specific planning fields to one CSV row."""
    result_dict = asdict(
        planning_result
    )

    row.update(
        {
            "planning_stop_reason": result_dict["stop_reason"],
            "planning_trials_completed": result_dict[
                "trials_completed"
            ],
            "planning_goal_reaching_trials": result_dict[
                "goal_reaching_trials"
            ],
            "planning_step_limited_trials": result_dict[
                "step_limited_trials"
            ],
            "planning_elapsed_seconds": result_dict[
                "elapsed_seconds"
            ],
            "planning_bellman_backups": result_dict[
                "bellman_backups"
            ],
            "planning_transition_outcomes_evaluated": result_dict[
                "transition_outcomes_evaluated"
            ],
            "planning_final_trial_residual": result_dict[
                "final_trial_residual"
            ],
            "planning_final_trial_scaled_residual": result_dict.get(
                "final_trial_scaled_residual"
            ),
            "planning_consecutive_stable_trials": result_dict[
                "consecutive_stable_trials"
            ],
            "planning_maximum_consecutive_stable_trials": result_dict[
                "maximum_consecutive_stable_trials"
            ],
            "planning_stability_criterion_reached": result_dict[
                "stability_criterion_reached"
            ],
            "planning_first_stability_trial": result_dict[
                "first_stability_trial"
            ],
            "planning_first_stability_elapsed_seconds": result_dict.get(
                "first_stability_elapsed_seconds"
            ),
            "planning_initial_state_solved": result_dict.get(
                "initial_state_solved"
            ),
            "planning_solved_states": result_dict.get("solved_states"),
            "planning_solved_od_states": result_dict.get("solved_od_states"),
            "planning_solved_real_states": result_dict.get("solved_real_states"),
            "planning_solved_checks": result_dict.get("solved_checks"),
            "planning_first_solved_trial": result_dict.get("first_solved_trial"),
            "planning_first_solved_elapsed_seconds": result_dict.get(
                "first_solved_elapsed_seconds"
            ),
            "planning_resolved_max_steps_per_trial": result_dict[
                "resolved_max_steps_per_trial"
            ],
            "planning_memory_limit_reached": result_dict.get(
                "memory_limit_reached"
            ),
            "planning_baseline_rss_mb": result_dict.get("baseline_rss_mb"),
            "planning_peak_rss_mb": result_dict.get("peak_rss_mb"),
            "planning_peak_rss_delta_mb": result_dict.get(
                "peak_rss_delta_mb"
            ),
            "planning_result_json": _json_text(
                result_dict
            ),
        }
    )

    if algorithm == "baseline":
        row.update(
            {
                "planning_total_real_steps": result_dict[
                    "total_trial_steps"
                ],
                "planning_action_evaluations": result_dict[
                    "planning_action_evaluations"
                ],
                "planning_visited_states": result_dict[
                    "visited_states"
                ],
            }
        )

    elif algorithm == "od":
        row.update(
            {
                "planning_total_real_steps": result_dict[
                    "total_real_steps"
                ],
                "planning_operator_evaluations": result_dict[
                    "planning_operator_evaluations"
                ],
                "planning_complete_joint_actions_evaluated": result_dict[
                    "complete_joint_actions_evaluated"
                ],
                "planning_visited_od_states": result_dict[
                    "visited_od_states"
                ],
                "planning_visited_real_states": result_dict[
                    "visited_real_states"
                ],
            }
        )

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}"
        )


def _add_evaluation_summary(
    row: dict[str, Any],
    evaluation_summary: Any,
) -> None:
    """Add aggregate fixed-policy evaluation fields to one CSV row."""
    summary = asdict(
        evaluation_summary
    )

    row.update(
        {
            "evaluation_policy_name": summary[
                "policy_name"
            ],
            "evaluation_successful_episodes": summary[
                "successful_episodes"
            ],
            "evaluation_failed_episodes": summary[
                "failed_episodes"
            ],
            "evaluation_success_rate": summary[
                "success_rate"
            ],
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
            "evaluation_per_agent_arrival_rates": _json_text(
                summary["per_agent_arrival_rates"]
            ),
            "evaluation_per_agent_mean_arrival_times": _json_text(
                summary[
                    "per_agent_mean_arrival_times"
                ]
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
            "evaluation_mean_expected_vertex_conflict_probability_per_step": summary.get(
                "mean_expected_vertex_conflict_probability_per_step"
            ),
            "evaluation_mean_expected_edge_swap_probability_per_step": summary.get(
                "mean_expected_edge_swap_probability_per_step"
            ),
            "evaluation_mean_expected_noncollision_no_motion_probability_per_step": summary.get(
                "mean_expected_noncollision_no_motion_probability_per_step"
            ),
            "evaluation_mean_selected_unfinished_stay_actions_per_episode": summary.get(
                "mean_selected_unfinished_stay_actions_per_episode"
            ),
            "evaluation_mean_selected_unfinished_blocked_actions_per_episode": summary.get(
                "mean_selected_unfinished_blocked_actions_per_episode"
            ),
            "evaluation_deterministic_self_loop_failures": summary.get(
                "deterministic_self_loop_failures"
            ),
            "evaluation_step_limit_failures": summary.get(
                "step_limit_failures"
            ),
            "evaluation_elapsed_seconds": summary[
                "evaluation_elapsed_seconds"
            ],
            "evaluation_baseline_rss_mb": summary.get(
                "evaluation_baseline_rss_mb"
            ),
            "evaluation_peak_rss_mb": summary.get("evaluation_peak_rss_mb"),
            "evaluation_peak_rss_delta_mb": summary.get(
                "evaluation_peak_rss_delta_mb"
            ),
            "evaluation_total_policy_decision_seconds": summary[
                "total_policy_decision_seconds"
            ],
            "evaluation_mean_policy_decision_milliseconds": summary[
                "mean_policy_decision_milliseconds"
            ],
            "evaluation_policy_cache_hits": summary[
                "policy_cache_hits"
            ],
            "evaluation_policy_cache_misses": summary[
                "policy_cache_misses"
            ],
            "evaluation_policy_cache_entries": summary[
                "policy_cache_entries"
            ],
            "evaluation_policy_cache_hit_rate": summary[
                "policy_cache_hit_rate"
            ],
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
            "evaluation_mean_self_transitions": summary[
                "mean_self_transitions"
            ],
            "evaluation_mean_maximum_consecutive_self_transitions": summary[
                "mean_maximum_consecutive_self_transitions"
            ],
            "evaluation_mean_tie_decisions": summary[
                "mean_tie_decisions"
            ],
            "evaluation_mean_unique_real_states_with_policy_ties": summary[
                "mean_unique_real_states_with_policy_ties"
            ],
            "evaluation_mean_tie_decisions_per_environment_step": summary[
                "mean_tie_decisions_per_environment_step"
            ],
            "evaluation_summary_json": _json_text(
                summary
            ),
        }
    )


def _create_planner(
    *,
    algorithm: str,
    mdp: GridMMDP,
    heuristic: ShortestPathHeuristic,
    config: RTDPConfig,
) -> BaselineRTDP | OperatorDecompositionRTDP:
    if algorithm == "baseline":
        return BaselineRTDP(
            mdp=mdp,
            heuristic=heuristic,
            config=config,
        )

    if algorithm == "od":
        return OperatorDecompositionRTDP(
            mdp=mdp,
            heuristic=heuristic,
            config=config,
        )

    raise ValueError(
        f"Unknown algorithm: {algorithm!r}"
    )


def _read_existing_run_ids(
    output_path: Path,
) -> set[str]:
    """Read completed or failed run IDs from an existing CSV file."""
    if not output_path.is_file():
        return set()

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            return set()

        if "run_id" not in reader.fieldnames:
            raise ValueError(
                f"Existing CSV {output_path} has no run_id column"
            )

        return {
            row["run_id"]
            for row in reader
            if row.get("run_id")
        }


def _normalized_row(
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    Keep the output schema stable and convert None to an empty CSV field.
    """
    return {
        field: (
            ""
            if row.get(field) is None
            else row.get(field, "")
        )
        for field in CSV_FIELDS
    }


def _write_row(
    writer: csv.DictWriter,
    file: Any,
    row: dict[str, Any],
) -> None:
    """
    Write and flush immediately so completed runs survive later interruption.
    """
    writer.writerow(
        _normalized_row(row)
    )
    file.flush()


def _planning_config_for_seed(
    args: argparse.Namespace,
    planning_seed: int,
    *,
    map_name: str,
    n_agents: int,
    resource_profile: dict[str, Any],
) -> RTDPConfig:
    profile_time, profile_memory = resolve_resource_limits(
        resource_profile, map_name=map_name, n_agents=n_agents
    )

    mode = args.resource_mode
    if mode == "unconstrained":
        time_limit = None
        memory_limit = None
        stop_when_stable = False
        stop_when_solved = True
    elif mode == "time":
        time_limit = (
            args.time_limit_seconds
            if args.time_limit_seconds is not None
            else profile_time
        )
        memory_limit = None
        stop_when_stable = False
        stop_when_solved = False
    elif mode == "time_or_solved":
        time_limit = (
            args.time_limit_seconds
            if args.time_limit_seconds is not None
            else profile_time
        )
        memory_limit = None
        stop_when_stable = False
        stop_when_solved = True
    elif mode == "memory":
        time_limit = None
        memory_limit = (
            args.memory_limit_mb
            if args.memory_limit_mb is not None
            else profile_memory
        )
        stop_when_stable = False
        stop_when_solved = True
    elif mode == "time_memory":
        time_limit = (
            args.time_limit_seconds
            if args.time_limit_seconds is not None
            else profile_time
        )
        memory_limit = (
            args.memory_limit_mb
            if args.memory_limit_mb is not None
            else profile_memory
        )
        stop_when_stable = False
        stop_when_solved = False
    elif mode == "time_memory_or_solved":
        time_limit = (
            args.time_limit_seconds
            if args.time_limit_seconds is not None
            else profile_time
        )
        memory_limit = (
            args.memory_limit_mb
            if args.memory_limit_mb is not None
            else profile_memory
        )
        stop_when_stable = False
        stop_when_solved = True
    else:
        time_limit = args.time_limit_seconds
        memory_limit = args.memory_limit_mb
        stop_when_stable = args.stop_when_stable
        stop_when_solved = args.stop_when_solved

    if mode in {"time", "time_or_solved", "time_memory", "time_memory_or_solved"} and time_limit is None:
        raise ValueError(
            f"No time limit configured for {map_name}, agents={n_agents}"
        )
    if mode in {"memory", "time_memory", "time_memory_or_solved"} and memory_limit is None:
        raise ValueError(
            f"No memory limit configured for {map_name}, agents={n_agents}"
        )

    step_tail_probability = (
        args.step_tail_probability
        if args.step_tail_probability is not None
        else args.step_cap_familywise_error
        / (args.evaluation_episodes * n_agents)
    )

    return RTDPConfig(
        max_trials=args.max_trials,
        max_steps_per_trial=args.max_steps_per_trial,
        step_limit_multiplier=args.step_limit_multiplier,
        step_tail_probability=step_tail_probability,
        time_limit_seconds=time_limit,
        memory_limit_mb=memory_limit,
        epsilon=args.epsilon,
        relative_epsilon=args.relative_epsilon,
        stable_trials_required=args.stable_trials_required,
        stop_when_stable=stop_when_stable,
        stop_when_solved=stop_when_solved,
        require_goal_for_stability=True,
        tie_tolerance=args.tie_tolerance,
        tie_ulps=args.tie_ulps,
        seed=planning_seed,
    )


def _evaluation_config_for_seed(
    args: argparse.Namespace,
    evaluation_seed: int,
) -> EvaluationConfig:
    return EvaluationConfig(
        episodes=args.evaluation_episodes,
        seed=evaluation_seed,
        max_steps_per_episode=(
            args.evaluation_max_steps
        ),
        measure_conflict_risk=(
            not args.disable_conflict_risk
        ),
        randomize_greedy_ties=args.randomize_evaluation_ties,
        cache_only_executed_actions=(
            not args.cache_all_evaluation_transitions
        ),
        collect_diagnostics=(
            not args.disable_evaluation_diagnostics
        ),
        time_limit_seconds=args.evaluation_time_limit_seconds,
    )


def _mdp_config_from_args(
    args: argparse.Namespace,
) -> MMDPConfig:
    return MMDPConfig(
        slip_to_stay_probability=args.slip,
        freeze_agents_at_goal=True,
        reject_conflicting_transitions=True,
        transition_cache_max_entries=(
            args.transition_cache_max_entries
        ),
    )


def _add_step_cap_diagnostics(
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
        longest_distance / q
        if q > 0.0
        else float("inf")
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


def _resolve_seed_pairs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.planning_seeds is not None:
        planning = list(args.planning_seeds)
        if args.evaluation_seeds is not None:
            evaluation = list(args.evaluation_seeds)
            if len(evaluation) != len(planning):
                raise ValueError(
                    "--evaluation-seeds must have the same length as "
                    "--planning-seeds"
                )
        else:
            rng = __import__("random").Random(args.master_seed)
            evaluation = [rng.randrange(0, 2**63) for _ in planning]
        return list(zip(planning, evaluation))

    rng = __import__("random").Random(args.master_seed)
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    while len(pairs) < args.seed_count:
        planning_seed = rng.randrange(0, 2**31)
        evaluation_seed = rng.randrange(0, 2**63)
        if planning_seed in used:
            continue
        used.add(planning_seed)
        pairs.append((planning_seed, evaluation_seed))
    return pairs


def _write_diagnostics_file(
    output_dir: Path,
    *,
    run_id: str,
    evaluation_result: Any,
    global_result: Any | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_id}_diagnostics.json"
    payload = {
        "summary": evaluation_result.summary.to_dict(),
        "failed_episodes": [
            result.to_dict()
            for result in evaluation_result.episode_results
            if not result.success
        ],
        "global_od_summary": (
            global_result.summary.to_dict() if global_result is not None else None
        ),
        "global_od_failed_episodes": (
            [
                result.to_dict()
                for result in global_result.episode_results
                if not result.success
            ]
            if global_result is not None
            else []
        ),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=list),
        encoding="utf-8",
    )
    return path


def run_experiments(args: argparse.Namespace) -> None:
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resource_profile = load_profile(args.resource_profile)
    profile_name = str(resource_profile.get("profile_name", ""))
    seed_pairs = _resolve_seed_pairs(args)
    scenario_numbers = list(args.scenario_numbers)
    task_offsets = list(args.task_offsets)

    existing_run_ids = (
        set() if args.overwrite else _read_existing_run_ids(output_path)
    )
    file_mode = "w" if args.overwrite or not output_path.exists() else "a"
    write_header = file_mode == "w" or output_path.stat().st_size == 0
    total_requested = (
        len(args.map_folders)
        * len(args.agent_counts)
        * len(scenario_numbers)
        * len(task_offsets)
        * len(seed_pairs)
        * len(args.algorithms)
    )
    completed_now = skipped = failed = sequence_number = 0

    with output_path.open(file_mode, encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=CSV_FIELDS, extrasaction="raise"
        )
        if write_header:
            writer.writeheader()
            output_file.flush()

        for map_folder_argument in args.map_folders:
            map_folder = map_folder_argument.resolve()
            for n_agents in args.agent_counts:
                for scenario_number in scenario_numbers:
                    for task_offset in task_offsets:
                        instance: MapInstance | None = None
                        instance_error: Exception | None = None
                        try:
                            instance = create_map_instance(
                                map_folder=map_folder,
                                n_agents=n_agents,
                                scenario_number=scenario_number,
                                task_offset=task_offset,
                                require_4way_reachability=True,
                            )
                        except Exception as exc:
                            instance_error = exc

                        map_name = (
                            instance.grid_map.name
                            if instance is not None
                            else map_folder.name
                        )

                        for seed_index, (planning_seed, evaluation_seed) in enumerate(
                            seed_pairs
                        ):
                            for algorithm in args.algorithms:
                                sequence_number += 1
                                planning_config = _planning_config_for_seed(
                                    args,
                                    planning_seed,
                                    map_name=map_name,
                                    n_agents=n_agents,
                                    resource_profile=resource_profile,
                                )
                                evaluation_config = _evaluation_config_for_seed(
                                    args, evaluation_seed
                                )
                                mdp_config = _mdp_config_from_args(args)
                                run_id = make_run_id(
                                    map_folder=map_folder,
                                    n_agents=n_agents,
                                    scenario_number=scenario_number,
                                    task_offset=task_offset,
                                    algorithm=algorithm,
                                    planning_seed=planning_seed,
                                    evaluation_seed=evaluation_seed,
                                    mdp_config=mdp_config,
                                    planning_config=planning_config,
                                    evaluation_config=evaluation_config,
                                )
                                progress = (
                                    f"[{sequence_number}/{total_requested}] "
                                    f"{map_name}, agents={n_agents}, "
                                    f"scenario={scenario_number}, offset={task_offset}, "
                                    f"pseed={planning_seed}, eseed={evaluation_seed}, "
                                    f"{algorithm}"
                                )
                                if run_id in existing_run_ids:
                                    skipped += 1
                                    print(f"{progress} -- skipped")
                                    continue

                                row = _base_row(
                                    run_id=run_id,
                                    map_folder=map_folder,
                                    instance=instance,
                                    n_agents=n_agents,
                                    scenario_number=scenario_number,
                                    task_offset=task_offset,
                                    algorithm=algorithm,
                                    planning_seed=planning_seed,
                                    evaluation_seed=evaluation_seed,
                                    mdp_config=mdp_config,
                                    planning_config=planning_config,
                                    evaluation_config=evaluation_config,
                                )
                                row.update(
                                    {
                                        "resource_mode": args.resource_mode,
                                        "resource_profile_name": profile_name,
                                        "master_seed": args.master_seed,
                                        "seed_index": seed_index,
                                        "step_cap_familywise_error": args.step_cap_familywise_error,
                                        "stability_confidence": args.stability_confidence,
                                        "minimum_unstable_trial_rate": args.minimum_unstable_trial_rate,
                                        "evaluation_confidence": args.evaluation_confidence,
                                        "evaluation_half_width": args.evaluation_half_width,
                                    }
                                )

                                if instance_error is not None:
                                    row.update(
                                        {
                                            "status": "error",
                                            "error_type": type(instance_error).__name__,
                                            "error_message": str(instance_error),
                                        }
                                    )
                                    _write_row(writer, output_file, row)
                                    existing_run_ids.add(run_id)
                                    failed += 1
                                    print(f"{progress} -- ERROR: {instance_error}")
                                    if args.fail_fast:
                                        raise instance_error
                                    continue

                                assert instance is not None
                                print(f"{progress} -- planning...", flush=True)
                                try:
                                    mdp = GridMMDP(instance=instance, config=mdp_config)
                                    heuristic = ShortestPathHeuristic(mdp)
                                    planner = _create_planner(
                                        algorithm=algorithm,
                                        mdp=mdp,
                                        heuristic=heuristic,
                                        config=planning_config,
                                    )
                                    _add_step_cap_diagnostics(
                                        row,
                                        mdp=mdp,
                                        heuristic=heuristic,
                                        planner=planner,
                                        evaluation_config=evaluation_config,
                                    )
                                    planning_result = planner.solve()
                                    _add_planning_result(
                                        row,
                                        algorithm=algorithm,
                                        planning_result=planning_result,
                                    )

                                    print(f"{progress} -- evaluating...", flush=True)
                                    evaluation_result = evaluate_policy(
                                        mdp=mdp,
                                        planner=planner,
                                        config=evaluation_config,
                                    )
                                    _add_evaluation_summary(
                                        row, evaluation_result.summary
                                    )
                                    planning_baseline = float(
                                        row["planning_baseline_rss_mb"]
                                    )
                                    overall_peak = max(
                                        float(row["planning_peak_rss_mb"]),
                                        float(row["evaluation_peak_rss_mb"]),
                                    )
                                    row["overall_peak_rss_mb"] = overall_peak
                                    row[
                                        "overall_peak_rss_delta_from_planning_baseline_mb"
                                    ] = max(0.0, overall_peak - planning_baseline)

                                    global_result = None
                                    if (
                                        algorithm == "od"
                                        and args.evaluate_od_global_diagnostic
                                    ):
                                        adapter = MethodPolicyAdapter(
                                            planner,
                                            action_method="global_policy_action",
                                            info_method="global_policy_action_with_info",
                                            policy_name="ODGlobalRealValueDiagnostic",
                                        )
                                        global_result = evaluate_policy(
                                            mdp=mdp,
                                            planner=adapter,
                                            config=evaluation_config,
                                        )
                                        row.update(
                                            {
                                                "od_global_diagnostic_success_rate": (
                                                    global_result.summary.success_rate
                                                ),
                                                "od_global_diagnostic_mean_cost": (
                                                    global_result.summary.mean_sum_of_costs_successful_episodes
                                                ),
                                                "od_global_diagnostic_mean_makespan": (
                                                    global_result.summary.mean_makespan_successful_episodes
                                                ),
                                                "od_global_diagnostic_summary_json": _json_text(
                                                    global_result.summary.to_dict()
                                                ),
                                            }
                                        )

                                    if args.diagnostics_output_dir is not None:
                                        diagnostic_path = _write_diagnostics_file(
                                            args.diagnostics_output_dir.resolve(),
                                            run_id=run_id,
                                            evaluation_result=evaluation_result,
                                            global_result=global_result,
                                        )
                                        row["diagnostics_file"] = str(diagnostic_path)

                                    row["status"] = "ok"
                                    _write_row(writer, output_file, row)
                                    existing_run_ids.add(run_id)
                                    completed_now += 1
                                    print(
                                        f"{progress} -- done; "
                                        f"stop={planning_result.stop_reason}, "
                                        f"success={evaluation_result.summary.success_rate:.3f}"
                                    )
                                except KeyboardInterrupt:
                                    print("\nInterrupted; completed rows are saved.")
                                    raise
                                except Exception as exc:
                                    row.update(
                                        {
                                            "status": "error",
                                            "error_type": type(exc).__name__,
                                            "error_message": str(exc),
                                        }
                                    )
                                    _write_row(writer, output_file, row)
                                    existing_run_ids.add(run_id)
                                    failed += 1
                                    print(f"{progress} -- ERROR: {exc}")
                                    if args.fail_fast:
                                        raise

    print("\nExperiment batch finished.")
    print(f"Output: {output_path}")
    print(f"Completed now: {completed_now}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Baseline RTDP and OD-RTDP experiments with optional "
            "time and process-memory resource regimes."
        )
    )
    parser.add_argument("map_folders", type=Path, nargs="+")
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[2, 3])
    parser.add_argument(
        "--algorithms", choices=ALGORITHMS, nargs="+", default=list(ALGORITHMS)
    )

    parser.add_argument("--planning-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--master-seed", type=int, default=20260708)

    parser.add_argument("--scenario-numbers", type=int, nargs="+", default=[1])
    parser.add_argument("--task-offsets", type=int, nargs="+", default=[0])
    parser.add_argument("--slip", type=float, default=0.20)

    parser.add_argument(
        "--resource-mode",
        choices=RESOURCE_MODES,
        default="custom",
        help="custom, unconstrained, time, memory, or time_memory",
    )
    parser.add_argument("--resource-profile", type=Path)
    parser.add_argument("--memory-limit-mb", type=parse_optional_float, default=None)
    parser.add_argument("--time-limit-seconds", type=parse_optional_float, default=None)
    parser.add_argument("--max-trials", type=parse_optional_int, default=None)
    parser.add_argument("--max-steps-per-trial", type=parse_optional_int, default=None)
    parser.add_argument(
        "--step-limit-multiplier",
        type=parse_optional_float,
        default=None,
        help="Legacy override. Default uses the stochastic tail bound.",
    )
    parser.add_argument(
        "--step-tail-probability", type=parse_optional_float, default=None,
        help="Explicit per-agent tail probability. Default derives it from the family-wise error target.",
    )
    parser.add_argument(
        "--step-cap-familywise-error", type=float, default=0.01,
        help="Target upper bound across all evaluation episodes and agents.",
    )

    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--relative-epsilon", type=float, default=1e-6)
    parser.add_argument("--stable-trials-required", type=parse_optional_int, default=None)
    parser.add_argument("--stability-confidence", type=float, default=0.99)
    parser.add_argument("--minimum-unstable-trial-rate", type=float, default=0.10)
    parser.add_argument("--stop-when-stable", action="store_true")
    parser.add_argument(
        "--stop-when-solved",
        action="store_true",
        help=(
            "Use LRTDP-style solved-state stopping. This is selected "
            "automatically by unconstrained, memory, time_or_solved, "
            "and time_memory_or_solved resource modes."
        ),
    )
    parser.add_argument(
        "--transition-cache-max-entries",
        type=parse_optional_nonnegative_int,
        default=None,
        help="Optional cache-entry proxy limit. Final memory modes should use RSS.",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=parse_optional_nonnegative_float,
        default=None,
        help="Explicit absolute tie tolerance; default uses ULP comparison.",
    )
    parser.add_argument("--tie-ulps", type=int, default=8)

    parser.add_argument("--evaluation-episodes", type=parse_optional_int, default=None)
    parser.add_argument("--evaluation-confidence", type=float, default=0.95)
    parser.add_argument("--evaluation-half-width", type=float, default=0.10)
    parser.add_argument("--evaluation-max-steps", type=parse_optional_int, default=None)
    parser.add_argument(
        "--evaluation-time-limit-seconds",
        type=parse_optional_float,
        default=None,
        help="Stop starting new evaluation episodes after this budget.",
    )
    parser.add_argument("--disable-conflict-risk", action="store_true")
    parser.add_argument("--randomize-evaluation-ties", action="store_true")
    parser.add_argument("--cache-all-evaluation-transitions", action="store_true")
    parser.add_argument("--disable-evaluation-diagnostics", action="store_true")
    parser.add_argument("--evaluate-od-global-diagnostic", action="store_true")
    parser.add_argument("--diagnostics-output-dir", type=Path)

    parser.add_argument(
        "--output", type=Path, default=Path("results/raw_results_v7.csv")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _resolve_derived_defaults(args: argparse.Namespace) -> None:
    if args.evaluation_episodes is None:
        args.evaluation_episodes = binomial_worst_case_sample_size(
            confidence=args.evaluation_confidence,
            half_width=args.evaluation_half_width,
        )
    if args.stable_trials_required is None:
        args.stable_trials_required = consecutive_trials_for_detection(
            confidence=args.stability_confidence,
            minimum_event_probability=args.minimum_unstable_trial_rate,
        )


def _validate_cli_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if any(count <= 0 for count in args.agent_counts):
        parser.error("Every agent count must be positive")
    if any(number <= 0 for number in args.scenario_numbers):
        parser.error("Scenario numbers must be positive")
    if any(offset < 0 for offset in args.task_offsets):
        parser.error("Task offsets cannot be negative")
    if not 0.0 <= args.slip < 1.0:
        parser.error("--slip must be in [0, 1)")
    if not 0.0 < args.evaluation_confidence < 1.0:
        parser.error("--evaluation-confidence must be in (0, 1)")
    if not 0.0 < args.evaluation_half_width < 1.0:
        parser.error("--evaluation-half-width must be in (0, 1)")
    if not 0.0 < args.stability_confidence < 1.0:
        parser.error("--stability-confidence must be in (0, 1)")
    if not 0.0 < args.minimum_unstable_trial_rate < 1.0:
        parser.error("--minimum-unstable-trial-rate must be in (0, 1)")
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    if args.evaluation_seeds is not None and args.planning_seeds is None:
        parser.error("Explicit evaluation seeds require explicit planning seeds")
    if args.epsilon < 0.0 or args.relative_epsilon < 0.0:
        parser.error("Residual tolerances cannot be negative")
    if (
        args.step_tail_probability is not None
        and not 0.0 < args.step_tail_probability < 1.0
    ):
        parser.error("--step-tail-probability must be in (0, 1) or none")
    if not 0.0 < args.step_cap_familywise_error < 1.0:
        parser.error("--step-cap-familywise-error must be in (0, 1)")
    if args.stable_trials_required is not None and args.stable_trials_required <= 0:
        parser.error("--stable-trials-required must be positive")
    if args.tie_ulps <= 0:
        parser.error("--tie-ulps must be positive")
    if args.evaluation_episodes <= 0:
        parser.error("--evaluation-episodes must be positive")
    if (
        args.resource_mode == "custom"
        and args.max_trials is None
        and args.time_limit_seconds is None
        and args.memory_limit_mb is None
        and not args.stop_when_stable
        and not args.stop_when_solved
    ):
        parser.error("Custom mode needs a stopping mechanism")


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    _resolve_derived_defaults(args)
    _validate_cli_arguments(parser, args)
    run_experiments(args)


if __name__ == "__main__":
    main()
