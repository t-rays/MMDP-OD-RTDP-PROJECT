from __future__ import annotations

"""Experiment orchestration loop with resumable CSV output.

Every requested map/agent/algorithm condition uses one explicit fixed seed.
The same seed initializes separate planning and evaluation RNG instances, so a
run remains reproducible without treating seeds as an experimental dimension.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mmdp.evaluation import MethodPolicyAdapter, evaluate_policy
from mmdp.experiments.factory import (
    load_profile,
    create_planner,
    evaluation_config_from_args,
    mdp_config_from_args,
    planning_config_from_args,
)
from mmdp.experiments.schema import (
    CSV_FIELDS,
    add_evaluation_summary,
    add_planning_result,
    add_step_cap_diagnostics,
    base_row,
    json_text,
    make_run_id,
    normalized_row,
)
from mmdp.domain.grid_mmdp import GridMMDP
from mmdp.domain.heuristic import ShortestPathHeuristic
from mmdp.domain.map_creator import MapInstance, create_map_instance


def read_existing_run_ids(output_path: Path) -> set[str]:
    """Read completed or failed run IDs from an existing CSV file."""
    if not output_path.is_file():
        return set()

    with output_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return set()
        if "run_id" not in reader.fieldnames:
            raise ValueError(f"Existing CSV {output_path} has no run_id column")
        return {row["run_id"] for row in reader if row.get("run_id")}


def write_row(writer: csv.DictWriter, file: Any, row: dict[str, Any]) -> None:
    """Write and flush immediately so completed runs survive interruption."""
    writer.writerow(normalized_row(row))
    file.flush()


def write_diagnostics_file(
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
    scenario_numbers = list(args.scenario_numbers)
    task_offsets = list(args.task_offsets)

    existing_run_ids = (
        set() if args.overwrite else read_existing_run_ids(output_path)
    )
    file_mode = "w" if args.overwrite or not output_path.exists() else "a"
    write_header = file_mode == "w" or output_path.stat().st_size == 0
    total_requested = (
        len(args.map_folders)
        * len(args.agent_counts)
        * len(scenario_numbers)
        * len(task_offsets)
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

                        for algorithm in args.algorithms:
                            sequence_number += 1
                            planning_config = planning_config_from_args(
                                args,
                                map_name=map_name,
                                n_agents=n_agents,
                                resource_profile=resource_profile,
                            )
                            evaluation_config = evaluation_config_from_args(args)
                            mdp_config = mdp_config_from_args(args)
                            run_id = make_run_id(
                                map_folder=map_folder,
                                n_agents=n_agents,
                                scenario_number=scenario_number,
                                task_offset=task_offset,
                                algorithm=algorithm,
                                planning_seed=args.seed,
                                evaluation_seed=args.seed,
                                mdp_config=mdp_config,
                                planning_config=planning_config,
                                evaluation_config=evaluation_config,
                            )
                            progress = (
                                f"[{sequence_number}/{total_requested}] "
                                f"{map_name}, agents={n_agents}, "
                                f"scenario={scenario_number}, offset={task_offset}, "
                                f"seed={args.seed}, {algorithm}"
                            )
                            if run_id in existing_run_ids:
                                skipped += 1
                                print(f"{progress} -- skipped")
                                continue

                            row = base_row(
                                run_id=run_id,
                                map_folder=map_folder,
                                instance=instance,
                                n_agents=n_agents,
                                scenario_number=scenario_number,
                                task_offset=task_offset,
                                algorithm=algorithm,
                                planning_seed=args.seed,
                                evaluation_seed=args.seed,
                                mdp_config=mdp_config,
                                planning_config=planning_config,
                                evaluation_config=evaluation_config,
                            )
                            row.update(
                                {
                                    "resource_mode": args.resource_mode,
                                    "resource_profile_name": profile_name,
                                    "master_seed": args.seed,
                                    "seed_index": 0,
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
                                write_row(writer, output_file, row)
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
                                planner = create_planner(
                                    algorithm=algorithm,
                                    mdp=mdp,
                                    heuristic=heuristic,
                                    config=planning_config,
                                )
                                add_step_cap_diagnostics(
                                    row,
                                    mdp=mdp,
                                    heuristic=heuristic,
                                    planner=planner,
                                    evaluation_config=evaluation_config,
                                )
                                planning_result = planner.solve()
                                add_planning_result(
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
                                add_evaluation_summary(row, evaluation_result.summary)
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
                                if algorithm == "od" and args.evaluate_od_global_diagnostic:
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
                                            "od_global_diagnostic_summary_json": json_text(
                                                global_result.summary.to_dict()
                                            ),
                                        }
                                    )

                                if args.diagnostics_output_dir is not None:
                                    diagnostic_path = write_diagnostics_file(
                                        args.diagnostics_output_dir.resolve(),
                                        run_id=run_id,
                                        evaluation_result=evaluation_result,
                                        global_result=global_result,
                                    )
                                    row["diagnostics_file"] = str(diagnostic_path)

                                row["status"] = "ok"
                                write_row(writer, output_file, row)
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
                                write_row(writer, output_file, row)
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
