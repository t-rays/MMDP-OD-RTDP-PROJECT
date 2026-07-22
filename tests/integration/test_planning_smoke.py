from __future__ import annotations

import pytest

from mmdp import (
    EvaluationConfig,
    GridMMDP,
    ODRTDPPlanningResult,
    RTDPConfig,
    RTDPPlanningResult,
    ShortestPathHeuristic,
    evaluate_policy,
)
from mmdp.experiments.factory import create_planner

PLANNING_CONFIG = RTDPConfig(time_limit_seconds=15.0, stop_when_solved=True, seed=7)


def _solve(algorithm: str, mdp: GridMMDP, heuristic: ShortestPathHeuristic):
    planner = create_planner(
        algorithm=algorithm, mdp=mdp, heuristic=heuristic, config=PLANNING_CONFIG
    )
    return planner, planner.solve()


@pytest.mark.parametrize("algorithm", ["baseline", "od"])
def test_planner_solves_easy_map(
    algorithm: str, easy_mdp: GridMMDP, easy_heuristic: ShortestPathHeuristic
) -> None:
    planner, result = _solve(algorithm, easy_mdp, easy_heuristic)

    expected_type = RTDPPlanningResult if algorithm == "baseline" else ODRTDPPlanningResult
    assert isinstance(result, expected_type)
    assert result.stop_reason == "initial_state_solved"
    assert result.trials_completed >= 1
    assert result.total_trial_steps > 0
    assert result.bellman_backups > 0
    assert result.transition_outcomes_evaluated > 0
    assert result.initial_state_solved
    assert result.resolved_max_steps_per_trial is not None
    assert planner.resolved_max_steps_per_trial == result.resolved_max_steps_per_trial


def test_od_result_contains_od_specific_metrics(
    easy_mdp: GridMMDP, easy_heuristic: ShortestPathHeuristic
) -> None:
    _, result = _solve("od", easy_mdp, easy_heuristic)
    assert result.planning_operator_evaluations > 0
    assert result.complete_joint_actions_evaluated > 0
    assert result.visited_od_states >= result.visited_real_states > 0


@pytest.mark.parametrize("algorithm", ["baseline", "od"])
def test_policy_evaluation_after_planning(
    algorithm: str, easy_mdp: GridMMDP, easy_heuristic: ShortestPathHeuristic
) -> None:
    planner, _ = _solve(algorithm, easy_mdp, easy_heuristic)

    evaluation = evaluate_policy(
        mdp=easy_mdp,
        planner=planner,
        config=EvaluationConfig(episodes=3, seed=11),
    )
    summary = evaluation.summary
    assert summary.episodes == 3
    assert summary.success_rate == 1.0
    # Cache stats must flow through the planner facade, not silently report 0.
    assert summary.policy_cache_misses > 0


@pytest.mark.parametrize("algorithm", ["baseline", "od"])
def test_solve_is_deterministic_for_fixed_seed(
    algorithm: str, easy_mdp: GridMMDP, easy_heuristic: ShortestPathHeuristic
) -> None:
    _, first = _solve(algorithm, easy_mdp, easy_heuristic)
    _, second = _solve(algorithm, easy_mdp, easy_heuristic)
    assert first.trials_completed == second.trials_completed
    assert first.total_trial_steps == second.total_trial_steps
    assert first.bellman_backups == second.bellman_backups
