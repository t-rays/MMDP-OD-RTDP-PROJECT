from __future__ import annotations

import pytest

from mmdp import EvaluationConfig, GridMMDP, RTDPConfig, ShortestPathHeuristic, evaluate_policy
from mmdp.experiments.factory import create_planner


def _solve(algorithm: str, mdp: GridMMDP, heuristic: ShortestPathHeuristic):
    config = RTDPConfig(
        time_limit_seconds=15.0,
        step_tail_probability=0.001,
        seed=7,
    )
    planner = create_planner(algorithm, mdp, heuristic, config)
    return planner, planner.solve()


@pytest.mark.parametrize("algorithm", ["baseline", "od"])
def test_planner_solves_easy_map(algorithm, easy_mdp, easy_heuristic) -> None:
    planner, result = _solve(algorithm, easy_mdp, easy_heuristic)
    assert result.stop_reason == "initial_state_solved"
    assert result.elapsed_seconds > 0.0


@pytest.mark.parametrize("algorithm", ["baseline", "od"])
def test_policy_evaluation_after_planning(algorithm, easy_mdp, easy_heuristic) -> None:
    planner, _ = _solve(algorithm, easy_mdp, easy_heuristic)
    summary = evaluate_policy(
        easy_mdp,
        planner,
        EvaluationConfig(
            episodes=3,
            seed=11,
            max_steps_per_episode=80,
            time_limit_seconds=5.0,
        ),
    )
    assert summary.scheduled_episodes == 3
    assert summary.completed_episodes == 3
    assert summary.successful_episodes == 3
