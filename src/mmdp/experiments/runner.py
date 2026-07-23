from __future__ import annotations

"""Execute one fixed experimental condition."""

from pathlib import Path

from mmdp.domain.heuristic import ShortestPathHeuristic
from mmdp.domain.map_creator import create_map_instance
from mmdp.evaluation import evaluate_policy
from mmdp.experiments.factory import (
    create_evaluation_config,
    create_mdp,
    create_planner,
    create_planning_config,
)
from mmdp.experiments.final_config import FIXED_SEED
from mmdp.experiments.schema import make_run_id


def run_condition(
    *,
    map_group: str,
    map_folder: Path,
    scenario_number: int,
    task_offset: int,
    evaluation_max_steps: int,
    n_agents: int,
    algorithm: str,
) -> dict:
    instance = create_map_instance(
        map_folder=map_folder,
        n_agents=n_agents,
        scenario_number=scenario_number,
        task_offset=task_offset,
    )
    mdp = create_mdp(instance)
    heuristic = ShortestPathHeuristic(mdp)
    planning_config = create_planning_config(n_agents)
    planner = create_planner(algorithm, mdp, heuristic, planning_config)
    planning_result = planner.solve()
    evaluation_summary = evaluate_policy(
        mdp,
        planner,
        create_evaluation_config(evaluation_max_steps),
    )

    return {
        "run_id": make_run_id(map_group, mdp.map_name, n_agents, algorithm),
        "map_group": map_group,
        "map_name": mdp.map_name,
        "n_agents": n_agents,
        "algorithm": algorithm,
        "seed": FIXED_SEED,
        "status": "ok",
        "planning_stop_reason": planning_result.stop_reason,
        "planning_time_seconds": planning_result.elapsed_seconds,
        "planning_peak_memory_delta_mb": planning_result.peak_rss_delta_mb,
        "evaluation_successful_episodes": evaluation_summary.successful_episodes,
        "evaluation_failed_episodes": evaluation_summary.failed_episodes,
        "evaluation_episodes_completed": evaluation_summary.completed_episodes,
        "evaluation_uncompleted_episodes": evaluation_summary.uncompleted_episodes,
        "evaluation_scheduled_episodes": evaluation_summary.scheduled_episodes,
        "evaluation_success_rate": evaluation_summary.success_rate,
        "evaluation_time_seconds": evaluation_summary.elapsed_seconds,
    }
