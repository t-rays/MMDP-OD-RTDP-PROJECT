from __future__ import annotations

"""Paired diagnostics for explaining Baseline-vs-OD policy differences.

This script is not part of the main experiment batch. It solves the same map
with both planners and compares their greedy action distributions on a shared
sample of real states using one neutral, common one-step score:

    c(s) + E[h(next_state)]

where h is the same slip-aware shortest-path heuristic for both algorithms.
It also reports expected self-loop probability, conflict probability, and
expected shortest-path progress. This helps distinguish computational savings
from policy-quality differences under a finite planning budget.
"""

import argparse
import csv
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
from typing import Iterable

from baseline_rtdp import BaselineRTDP, RTDPConfig
from grid_mmdp import GridMMDP, JointAction, MMDPConfig, State
from heuristic import ShortestPathHeuristic
from map_creator import create_map_instance
from od_rtdp import OperatorDecompositionRTDP


@dataclass(frozen=True)
class ActionMetrics:
    neutral_q: float
    self_loop_probability: float
    conflict_probability: float
    expected_shortest_path_progress: float


@dataclass(frozen=True)
class StateComparison:
    state: str
    is_initial_state: bool
    baseline_candidate_count: int
    od_joint_candidate_count: int
    candidate_overlap_count: int
    candidate_jaccard: float
    baseline_neutral_q: float
    od_neutral_q: float
    neutral_q_difference_od_minus_baseline: float
    baseline_self_loop_probability: float
    od_self_loop_probability: float
    self_loop_difference_od_minus_baseline: float
    baseline_conflict_probability: float
    od_conflict_probability: float
    baseline_expected_progress: float
    od_expected_progress: float
    progress_difference_od_minus_baseline: float


def _distance_sum(
    heuristic: ShortestPathHeuristic,
    state: State,
) -> float:
    return float(sum(heuristic.distance_summary(state)))


def _action_metrics(
    mdp: GridMMDP,
    heuristic: ShortestPathHeuristic,
    state: State,
    action: JointAction,
) -> ActionMetrics:
    with mdp.transition_cache_writes(False):
        transitions = mdp.joint_transitions(state, action)
        conflict = mdp.conflict_probability(state, action)

    immediate = mdp.transition_cost(state, action, transitions[0][0])
    neutral_q = immediate + sum(
        probability * heuristic(next_state)
        for next_state, probability in transitions
    )
    self_loop = sum(
        probability
        for next_state, probability in transitions
        if next_state == state
    )
    current_distance = _distance_sum(heuristic, state)
    expected_next_distance = sum(
        probability * _distance_sum(heuristic, next_state)
        for next_state, probability in transitions
    )

    return ActionMetrics(
        neutral_q=neutral_q,
        self_loop_probability=self_loop,
        conflict_probability=conflict,
        expected_shortest_path_progress=(
            current_distance - expected_next_distance
        ),
    )


def _baseline_distribution(
    planner: BaselineRTDP,
    state: State,
) -> dict[JointAction, float]:
    candidates = planner.greedy_action_candidates(state)
    probability = 1.0 / len(candidates)
    return {action: probability for action in candidates}


def _od_distribution(
    planner: OperatorDecompositionRTDP,
    state: State,
    *,
    maximum_joint_actions: int = 100_000,
) -> dict[JointAction, float]:
    """Return the actual uniform distribution used by OD evaluation."""
    candidates = planner.greedy_joint_action_candidates(state)

    if len(candidates) > maximum_joint_actions:
        raise RuntimeError(
            "OD greedy policy produced more than "
            f"{maximum_joint_actions} complete candidates"
        )

    probability = 1.0 / len(candidates)
    return {action: probability for action in candidates}


def _weighted_metrics(
    distribution: dict[JointAction, float],
    metrics: dict[JointAction, ActionMetrics],
) -> ActionMetrics:
    return ActionMetrics(
        neutral_q=sum(
            probability * metrics[action].neutral_q
            for action, probability in distribution.items()
        ),
        self_loop_probability=sum(
            probability * metrics[action].self_loop_probability
            for action, probability in distribution.items()
        ),
        conflict_probability=sum(
            probability * metrics[action].conflict_probability
            for action, probability in distribution.items()
        ),
        expected_shortest_path_progress=sum(
            probability * metrics[action].expected_shortest_path_progress
            for action, probability in distribution.items()
        ),
    )


def compare_state(
    mdp: GridMMDP,
    heuristic: ShortestPathHeuristic,
    baseline: BaselineRTDP,
    od: OperatorDecompositionRTDP,
    state: State,
) -> StateComparison:
    baseline_distribution = _baseline_distribution(baseline, state)
    od_distribution = _od_distribution(od, state)
    all_actions = set(baseline_distribution) | set(od_distribution)
    metrics = {
        action: _action_metrics(mdp, heuristic, state, action)
        for action in all_actions
    }

    baseline_metrics = _weighted_metrics(baseline_distribution, metrics)
    od_metrics = _weighted_metrics(od_distribution, metrics)

    baseline_set = set(baseline_distribution)
    od_set = set(od_distribution)
    overlap = baseline_set & od_set
    union = baseline_set | od_set

    return StateComparison(
        state=json.dumps(state),
        is_initial_state=(state == mdp.initial_state()),
        baseline_candidate_count=len(baseline_set),
        od_joint_candidate_count=len(od_set),
        candidate_overlap_count=len(overlap),
        candidate_jaccard=(len(overlap) / len(union) if union else 1.0),
        baseline_neutral_q=baseline_metrics.neutral_q,
        od_neutral_q=od_metrics.neutral_q,
        neutral_q_difference_od_minus_baseline=(
            od_metrics.neutral_q - baseline_metrics.neutral_q
        ),
        baseline_self_loop_probability=(
            baseline_metrics.self_loop_probability
        ),
        od_self_loop_probability=od_metrics.self_loop_probability,
        self_loop_difference_od_minus_baseline=(
            od_metrics.self_loop_probability
            - baseline_metrics.self_loop_probability
        ),
        baseline_conflict_probability=baseline_metrics.conflict_probability,
        od_conflict_probability=od_metrics.conflict_probability,
        baseline_expected_progress=(
            baseline_metrics.expected_shortest_path_progress
        ),
        od_expected_progress=od_metrics.expected_shortest_path_progress,
        progress_difference_od_minus_baseline=(
            od_metrics.expected_shortest_path_progress
            - baseline_metrics.expected_shortest_path_progress
        ),
    )


def _sample_states(
    baseline: BaselineRTDP,
    od: OperatorDecompositionRTDP,
    *,
    maximum_states: int,
    seed: int,
) -> list[State]:
    initial = baseline.mdp.initial_state()
    pool = set(baseline.V)
    pool.update(state for state, _ in od.V)
    pool.discard(initial)
    pool = {state for state in pool if not baseline.mdp.is_terminal(state)}

    rng = random.Random(seed)
    sampled = list(pool)
    rng.shuffle(sampled)
    return [initial] + sampled[: max(0, maximum_states - 1)]


def _write_csv(path: Path, rows: Iterable[StateComparison]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(asdict(materialized[0]).keys()),
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Baseline and OD greedy policies on shared states"
    )
    parser.add_argument("map_folder", type=Path)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--scenario-number", type=int, default=1)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--slip", type=float, default=0.20)
    parser.add_argument("--sample-states", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paired_policy_diagnostics.csv"),
    )
    args = parser.parse_args()

    instance = create_map_instance(
        map_folder=args.map_folder,
        n_agents=args.agents,
        scenario_number=args.scenario_number,
        task_offset=args.task_offset,
        require_4way_reachability=True,
    )
    mdp_config = MMDPConfig(
        slip_to_stay_probability=args.slip,
        freeze_agents_at_goal=True,
        reject_conflicting_transitions=True,
    )
    planning_config = RTDPConfig(
        max_trials=None,
        max_steps_per_trial=None,
        step_limit_multiplier=5.0,
        time_limit_seconds=args.time_limit_seconds,
        epsilon=1e-4,
        stable_trials_required=20,
        require_goal_for_stability=True,
        tie_tolerance=1e-9,
        seed=args.seed,
    )

    baseline_mdp = GridMMDP(instance, mdp_config)
    od_mdp = GridMMDP(instance, mdp_config)
    baseline = BaselineRTDP(
        baseline_mdp,
        ShortestPathHeuristic(baseline_mdp),
        planning_config,
    )
    od = OperatorDecompositionRTDP(
        od_mdp,
        ShortestPathHeuristic(od_mdp),
        planning_config,
    )

    print("Planning Baseline...")
    baseline_result = baseline.solve()
    print("Planning OD...")
    od_result = od.solve()

    diagnostic_mdp = GridMMDP(instance, mdp_config)
    diagnostic_heuristic = ShortestPathHeuristic(diagnostic_mdp)
    states = _sample_states(
        baseline,
        od,
        maximum_states=args.sample_states,
        seed=args.seed,
    )

    rows = [
        compare_state(
            diagnostic_mdp,
            diagnostic_heuristic,
            baseline,
            od,
            state,
        )
        for state in states
    ]
    _write_csv(args.output, rows)

    baseline_better_q = sum(
        row.neutral_q_difference_od_minus_baseline > 1e-9
        for row in rows
    )
    od_better_q = sum(
        row.neutral_q_difference_od_minus_baseline < -1e-9
        for row in rows
    )
    baseline_lower_self_loop = sum(
        row.self_loop_difference_od_minus_baseline > 1e-9
        for row in rows
    )
    od_lower_self_loop = sum(
        row.self_loop_difference_od_minus_baseline < -1e-9
        for row in rows
    )

    print()
    print("Planning summary")
    print(
        "Baseline:",
        baseline_result.stop_reason,
        f"trials={baseline_result.trials_completed}",
        f"residual={baseline_result.final_trial_residual:.6g}",
    )
    print(
        "OD:",
        od_result.stop_reason,
        f"trials={od_result.trials_completed}",
        f"residual={od_result.final_trial_residual:.6g}",
    )
    print()
    print(f"Compared states: {len(rows)}")
    print(f"Lower common heuristic Q -- Baseline: {baseline_better_q}")
    print(f"Lower common heuristic Q -- OD: {od_better_q}")
    print(f"Lower self-loop probability -- Baseline: {baseline_lower_self_loop}")
    print(f"Lower self-loop probability -- OD: {od_lower_self_loop}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
