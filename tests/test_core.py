from __future__ import annotations

"""
Core correctness tests for the multi-agent RTDP project.

Place this file at:

    final_project/tests/test_core.py

Run all tests from the project root with:

    python -m unittest discover -s tests -p "test_*.py" -v

The tests use only Python's standard unittest module.
"""

import math
import random
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


# Allow imports from final_project/src when this file is placed in tests/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
else:
    # Fallback that also makes the downloaded standalone file easy to test.
    CURRENT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(CURRENT_DIR))


from baseline_rtdp import BaselineRTDP, RTDPConfig
from evaluation import EvaluationConfig, evaluate_policy
from grid_mmdp import GridMMDP, MMDPConfig
from heuristic import ShortestPathHeuristic
from map_creator import GridMap, MapInstance, load_map_file
from od_rtdp import OperatorDecompositionRTDP
from limits import isolated_success_attempt_bound, sequential_multi_agent_step_bound
from statistics_utils import (
    binomial_worst_case_sample_size,
    consecutive_trials_for_detection,
)


def make_instance(
    rows: list[str],
    starts: tuple[tuple[int, int], ...],
    goals: tuple[tuple[int, int], ...],
    *,
    name: str = "test-map",
) -> MapInstance:
    """
    Create a small in-memory instance without external .map/.scen files.

    "." is free and "@" is blocked.
    """
    if not rows:
        raise ValueError("rows cannot be empty")

    width = len(rows[0])

    if width == 0:
        raise ValueError("rows cannot be empty strings")

    if any(len(row) != width for row in rows):
        raise ValueError("all rows must have equal width")

    free_cells: set[tuple[int, int]] = set()
    obstacles: set[tuple[int, int]] = set()

    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            position = (x, y)

            if symbol == ".":
                free_cells.add(position)
            elif symbol == "@":
                obstacles.add(position)
            else:
                raise ValueError(
                    f"Unsupported test-map symbol: {symbol!r}"
                )

    grid_map = GridMap(
        name=name,
        path=Path(f"{name}.map"),
        width=width,
        height=len(rows),
        grid=tuple(rows),
        obstacles=frozenset(obstacles),
        free_cells=frozenset(free_cells),
    )

    return MapInstance(
        grid_map=grid_map,
        scenario_file=Path(f"{name}.scen"),
        starts=starts,
        goals=goals,
        tasks=(),
    )


class MapCreatorTests(unittest.TestCase):
    def test_load_map_file(self) -> None:
        content = "\n".join(
            (
                "type octile",
                "height 3",
                "width 4",
                "map",
                "....",
                ".@@.",
                "....",
            )
        )

        with tempfile.TemporaryDirectory() as temp_directory:
            map_path = Path(temp_directory) / "small.map"
            map_path.write_text(content, encoding="utf-8")

            grid_map = load_map_file(map_path)

        self.assertEqual(grid_map.width, 4)
        self.assertEqual(grid_map.height, 3)
        self.assertIn((1, 1), grid_map.obstacles)
        self.assertIn((0, 0), grid_map.free_cells)
        self.assertNotIn((1, 1), grid_map.free_cells)


class DerivedParameterTests(unittest.TestCase):
    def test_evaluation_sample_size_is_derived_from_precision(self) -> None:
        self.assertEqual(
            binomial_worst_case_sample_size(
                confidence=0.95, half_width=0.10
            ),
            97,
        )

    def test_stability_streak_is_derived_from_detection_target(self) -> None:
        required = consecutive_trials_for_detection(
            confidence=0.99, minimum_event_probability=0.10
        )
        self.assertEqual(required, 44)
        self.assertLessEqual(0.9 ** required, 0.01)

    def test_probabilistic_step_bound_is_monotone_and_compositional(self) -> None:
        short = isolated_success_attempt_bound(10, 0.8, 1e-4)
        long = isolated_success_attempt_bound(20, 0.8, 1e-4)
        self.assertGreaterEqual(short, 10)
        self.assertGreater(long, short)
        self.assertEqual(
            sequential_multi_agent_step_bound([10, 20], 0.8, 1e-4),
            short + long,
        )


class MMDPTests(unittest.TestCase):
    def test_joint_transition_probabilities_sum_to_one(self) -> None:
        instance = make_instance(
            rows=[
                "....",
                "....",
            ],
            starts=((0, 0), (3, 1)),
            goals=((3, 0), (0, 1)),
        )

        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.20,
            ),
        )

        transitions = mdp.joint_transitions(
            mdp.initial_state(),
            ("right", "left"),
        )

        probability_sum = sum(
            probability
            for _, probability in transitions
        )

        self.assertTrue(
            math.isclose(
                probability_sum,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    def test_zero_slip_produces_one_deterministic_successor(self) -> None:
        instance = make_instance(
            rows=["..."],
            starts=((0, 0),),
            goals=((2, 0),),
        )

        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.0,
            ),
        )

        transitions = mdp.joint_transitions(
            ((0, 0),),
            ("right",),
        )

        self.assertEqual(
            transitions,
            ((((1, 0),), 1.0),),
        )

    def test_transition_cost_counts_unfinished_agents(self) -> None:
        instance = make_instance(
            rows=["..."],
            starts=((0, 0), (1, 0)),
            goals=((0, 0), (2, 0)),
        )

        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.0,
            ),
        )

        state = ((0, 0), (1, 0))
        action = ("stay", "right")
        next_state = ((0, 0), (2, 0))

        self.assertEqual(
            mdp.unfinished_agent_count(state),
            1,
        )

        self.assertEqual(
            mdp.transition_cost(
                state,
                action,
                next_state,
            ),
            1.0,
        )

        self.assertEqual(
            mdp.transition_cost(
                next_state,
                ("stay", "stay"),
                next_state,
            ),
            0.0,
        )

    def test_vertex_conflict_is_rejected_to_current_state(self) -> None:
        instance = make_instance(
            rows=["..."],
            starts=((0, 0), (2, 0)),
            goals=((2, 0), (0, 0)),
        )

        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.0,
                reject_conflicting_transitions=True,
            ),
        )

        state = mdp.initial_state()
        action = ("right", "left")

        self.assertEqual(
            mdp.conflict_probability(state, action),
            1.0,
        )

        self.assertEqual(
            mdp.joint_transitions(state, action),
            ((state, 1.0),),
        )

    def test_edge_swap_is_rejected_to_current_state(self) -> None:
        instance = make_instance(
            rows=[".."],
            starts=((0, 0), (1, 0)),
            goals=((1, 0), (0, 0)),
        )

        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.0,
                reject_conflicting_transitions=True,
            ),
        )

        state = mdp.initial_state()
        action = ("right", "left")

        self.assertEqual(
            mdp.edge_swap_conflict_count(
                state,
                ((1, 0), (0, 0)),
            ),
            1,
        )

        self.assertEqual(
            mdp.conflict_probability(state, action),
            1.0,
        )

        self.assertEqual(
            mdp.joint_transitions(state, action),
            ((state, 1.0),),
        )


    def test_transition_distributions_are_cached(self) -> None:
        instance = make_instance(
            rows=["..."],
            starts=((0, 0),),
            goals=((2, 0),),
        )

        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.20,
            ),
        )

        state = mdp.initial_state()
        action = ("right",)

        first = mdp.joint_transitions(
            state,
            action,
        )
        second = mdp.joint_transitions(
            state,
            action,
        )

        self.assertIs(
            first,
            second,
        )
        self.assertEqual(
            len(mdp._raw_transition_cache),
            1,
        )
        self.assertEqual(
            len(mdp._resolved_transition_cache),
            1,
        )

        mdp.clear_transition_cache()

        self.assertEqual(
            len(mdp._raw_transition_cache),
            0,
        )
        self.assertEqual(
            len(mdp._resolved_transition_cache),
            0,
        )

    def test_transition_cache_writes_can_be_suppressed(self) -> None:
        instance = make_instance(
            rows=["..."],
            starts=((0, 0),),
            goals=((2, 0),),
        )
        mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.2),
        )
        state = mdp.initial_state()

        with mdp.transition_cache_writes(False):
            first = mdp.joint_transitions(state, ("right",))

        stats = mdp.transition_cache_stats()
        self.assertEqual(stats["raw_entries"], 0)
        self.assertEqual(stats["resolved_entries"], 0)

        second = mdp.joint_transitions(state, ("right",))
        self.assertEqual(first, second)
        stats = mdp.transition_cache_stats()
        self.assertEqual(stats["raw_entries"], 1)
        self.assertEqual(stats["resolved_entries"], 1)

    def test_transition_cache_lru_bound_is_enforced(self) -> None:
        instance = make_instance(
            rows=["...."],
            starts=((0, 0),),
            goals=((3, 0),),
        )
        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.2,
                transition_cache_max_entries=1,
            ),
        )
        state = mdp.initial_state()
        mdp.joint_transitions(state, ("right",))
        mdp.joint_transitions(state, ("stay",))
        stats = mdp.transition_cache_stats()

        self.assertLessEqual(stats["raw_entries"], 1)
        self.assertLessEqual(stats["resolved_entries"], 1)
        self.assertGreaterEqual(stats["raw_evictions"], 1)
        self.assertGreaterEqual(stats["resolved_evictions"], 1)

    def test_self_loop_probability_includes_slip_and_waiting(self) -> None:
        instance = make_instance(
            rows=["....", "...."],
            starts=((0, 0), (0, 1)),
            goals=((3, 0), (3, 1)),
        )
        mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.2),
        )
        state = mdp.initial_state()

        self.assertAlmostEqual(
            mdp.self_loop_probability(state, ("right", "right")),
            0.04,
        )
        self.assertAlmostEqual(
            mdp.self_loop_probability(state, ("right", "stay")),
            0.2,
        )


    def test_action_risk_breakdown_separates_vertex_and_edge_conflicts(self) -> None:
        vertex_instance = make_instance(
            rows=["..."],
            starts=((0, 0), (2, 0)),
            goals=((2, 0), (0, 0)),
        )
        vertex_mdp = GridMMDP(
            vertex_instance,
            MMDPConfig(
                slip_to_stay_probability=0.0,
                reject_conflicting_transitions=True,
            ),
        )
        vertex = vertex_mdp.action_risk_breakdown(
            vertex_mdp.initial_state(), ("right", "left")
        )
        self.assertEqual(vertex["vertex_conflict_probability"], 1.0)
        self.assertEqual(vertex["edge_swap_probability"], 0.0)
        self.assertEqual(vertex["self_loop_probability"], 1.0)

        edge_instance = make_instance(
            rows=[".."],
            starts=((0, 0), (1, 0)),
            goals=((1, 0), (0, 0)),
        )
        edge_mdp = GridMMDP(
            edge_instance,
            MMDPConfig(
                slip_to_stay_probability=0.0,
                reject_conflicting_transitions=True,
            ),
        )
        edge = edge_mdp.action_risk_breakdown(
            edge_mdp.initial_state(), ("right", "left")
        )
        self.assertEqual(edge["vertex_conflict_probability"], 0.0)
        self.assertEqual(edge["edge_swap_probability"], 1.0)
        self.assertEqual(edge["self_loop_probability"], 1.0)



class HeuristicTests(unittest.TestCase):
    def test_shortest_path_respects_obstacles(self) -> None:
        instance = make_instance(
            rows=[
                "...",
                ".@.",
                "...",
            ],
            starts=((0, 1),),
            goals=((2, 1),),
        )

        mdp = GridMMDP(instance)
        heuristic = ShortestPathHeuristic(mdp)

        # The direct horizontal route is blocked, so the deterministic
        # shortest path has length 4. With the default q=0.8, the slip-aware
        # lower bound is 4 / 0.8 = 5.
        self.assertEqual(
            heuristic.agent_distance(0, (0, 1)),
            4.0,
        )
        self.assertEqual(
            heuristic(mdp.initial_state()),
            5.0,
        )

    def test_terminal_heuristic_is_zero(self) -> None:
        instance = make_instance(
            rows=["..."],
            starts=((0, 0),),
            goals=((2, 0),),
        )

        mdp = GridMMDP(instance)
        heuristic = ShortestPathHeuristic(mdp)

        self.assertEqual(
            heuristic(mdp.goal_state()),
            0.0,
        )

        self.assertEqual(
            heuristic.od_value(
                mdp.goal_state(),
                (),
            ),
            0.0,
        )

    def test_empty_od_prefix_matches_baseline_heuristic(self) -> None:
        instance = make_instance(
            rows=[
                "...",
                ".@.",
                "...",
            ],
            starts=((0, 1),),
            goals=((2, 1),),
        )

        mdp = GridMMDP(instance)
        heuristic = ShortestPathHeuristic(mdp)
        state = mdp.initial_state()

        self.assertEqual(
            heuristic.od_value(state, ()),
            heuristic(state),
        )

    def test_slip_aware_od_heuristic_penalizes_waiting(self) -> None:
        instance = make_instance(
            rows=["..."],
            starts=((0, 0),),
            goals=((2, 0),),
        )

        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.20,
            ),
        )
        heuristic = ShortestPathHeuristic(mdp)
        state = mdp.initial_state()

        self.assertAlmostEqual(
            heuristic(state),
            2.5,
        )
        self.assertAlmostEqual(
            heuristic.od_value(state, ()),
            2.5,
        )
        self.assertAlmostEqual(
            heuristic.od_value(state, ("right",)),
            2.5,
        )
        self.assertAlmostEqual(
            heuristic.od_value(state, ("stay",)),
            3.5,
        )

    def test_od_prefix_heuristic_is_safe_under_collision_rejection(self) -> None:
        # Agent 0 is forced to move away from its goal into agent 1, which is
        # already frozen at its goal.  Every completion of the prefix is
        # rejected back to the current state.  The uncapped isolated estimate
        # would be 3, while the correct admissible prefix lower bound is 2.
        instance = make_instance(
            rows=["..."],
            starts=((1, 0), (2, 0)),
            goals=((0, 0), (2, 0)),
        )
        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.0,
                freeze_agents_at_goal=True,
                reject_conflicting_transitions=True,
            ),
        )
        heuristic = ShortestPathHeuristic(mdp)
        state = mdp.initial_state()

        self.assertEqual(
            heuristic.expected_stochastic_distance_after_action(
                0, state[0], "right"
            ),
            2.0,
        )
        self.assertEqual(
            heuristic.safe_expected_stochastic_distance_after_action(
                0, state[0], "right"
            ),
            1.0,
        )
        self.assertEqual(heuristic.od_value(state, ("right",)), 2.0)

        od = OperatorDecompositionRTDP(
            mdp,
            heuristic,
            RTDPConfig(
                max_trials=1,
                max_steps_per_trial=5,
                step_limit_multiplier=None,
                time_limit_seconds=1.0,
                seed=0,
            ),
        )
        child = (state, ("right",))
        exact_backup = min(
            od.operator_value(child, action, count_metrics=False)
            for action in ("stay", "up", "down", "left", "right")
        )
        self.assertEqual(heuristic.od_value(*child), exact_backup)

    def test_path_diversity_guidance_breaks_equal_distance_tie(self) -> None:
        # From (2,2) both left and up reduce distance from 4 to 3. However,
        # left preserves two shortest continuations while up preserves one.
        instance = make_instance(
            rows=[
                ".@..",
                "....",
                "....",
                "....",
            ],
            starts=((2, 2),),
            goals=((0, 0),),
        )

        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.0,
            ),
        )
        heuristic = ShortestPathHeuristic(mdp)
        state = mdp.initial_state()

        self.assertEqual(
            heuristic.distance_after_action(0, state[0], "left"),
            3.0,
        )
        self.assertEqual(
            heuristic.distance_after_action(0, state[0], "up"),
            3.0,
        )
        self.assertGreater(
            heuristic.shortest_path_log_count(0, (1, 2)),
            heuristic.shortest_path_log_count(0, (2, 1)),
        )

        baseline = BaselineRTDP(
            mdp,
            heuristic,
            RTDPConfig(
                max_trials=1,
                max_steps_per_trial=10,
                step_limit_multiplier=None,
                time_limit_seconds=1.0,
                seed=0,
            ),
        )
        od = OperatorDecompositionRTDP(
            mdp,
            heuristic,
            baseline.config,
        )

        self.assertEqual(
            baseline.greedy_action_candidates(state),
            (("left",),),
        )
        self.assertEqual(
            od.greedy_operator_candidates((state, ())),
            ("left",),
        )

    def test_complete_guidance_prefers_lower_exact_self_loop(self) -> None:
        instance = make_instance(
            rows=["....", "...."],
            starts=((0, 0), (0, 1)),
            goals=((3, 0), (3, 1)),
        )
        mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.2),
        )
        heuristic = ShortestPathHeuristic(mdp)
        state = mdp.initial_state()

        moving_key = heuristic.joint_action_guidance_key(
            state,
            ("right", "right"),
        )
        waiting_key = heuristic.joint_action_guidance_key(
            state,
            ("right", "stay"),
        )

        self.assertLess(moving_key, waiting_key)


class PlannerTests(unittest.TestCase):
    @staticmethod
    def make_one_agent_problem() -> tuple[
        GridMMDP,
        ShortestPathHeuristic,
        RTDPConfig,
    ]:
        instance = make_instance(
            rows=["..."],
            starts=((0, 0),),
            goals=((2, 0),),
        )

        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.0,
            ),
        )

        heuristic = ShortestPathHeuristic(mdp)

        config = RTDPConfig(
            max_trials=50,
            max_steps_per_trial=10,
            step_limit_multiplier=None,
            time_limit_seconds=5.0,
            epsilon=1e-9,
            stable_trials_required=2,
            stop_when_stable=True,
            tie_tolerance=1e-9,
            seed=0,
        )

        return mdp, heuristic, config

    def test_single_agent_baseline_and_od_agree(self) -> None:
        mdp, heuristic, config = self.make_one_agent_problem()

        baseline = BaselineRTDP(
            mdp,
            heuristic,
            config,
        )

        od = OperatorDecompositionRTDP(
            mdp,
            heuristic,
            config,
        )

        baseline_result = baseline.solve()
        od_result = od.solve()

        initial_state = mdp.initial_state()

        self.assertEqual(
            baseline_result.stop_reason,
            "stable_trials",
        )

        self.assertEqual(
            od_result.stop_reason,
            "stable_trials",
        )

        self.assertEqual(
            baseline.policy_action(initial_state),
            ("right",),
        )

        self.assertEqual(
            od.policy_action(initial_state),
            ("right",),
        )

        self.assertTrue(
            math.isclose(
                baseline.value(initial_state),
                od.real_state_value(initial_state),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

        self.assertTrue(
            math.isclose(
                baseline.value(initial_state),
                2.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

    def test_fixed_budget_mode_measures_stability_without_early_stop(self) -> None:
        instance = make_instance(
            rows=[".."],
            starts=((0, 0),),
            goals=((1, 0),),
        )
        mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.0),
        )
        planner = BaselineRTDP(
            mdp,
            ShortestPathHeuristic(mdp),
            RTDPConfig(
                max_trials=5,
                max_steps_per_trial=5,
                step_limit_multiplier=None,
                time_limit_seconds=None,
                epsilon=1e-9,
                stable_trials_required=1,
                stop_when_stable=False,
                seed=0,
            ),
        )
        result = planner.solve()

        self.assertEqual(result.stop_reason, "max_trials")
        self.assertEqual(result.trials_completed, 5)
        self.assertTrue(result.stability_criterion_reached)
        self.assertIsNotNone(result.first_stability_trial)

    def test_deterministic_tie_choice_is_reproducible(self) -> None:
        mdp, heuristic, config = self.make_one_agent_problem()
        baseline = BaselineRTDP(mdp, heuristic, config)

        candidates = [("stay",), ("right",)]
        state = mdp.initial_state()

        first = baseline._deterministic_tie_choice(candidates, state)
        second = baseline._deterministic_tie_choice(candidates, state)

        self.assertEqual(first, second)
        self.assertIn(first, candidates)

    def test_greedy_candidate_sets_are_memoized(self) -> None:
        instance = make_instance(
            rows=[
                "..",
                "..",
            ],
            starts=((1, 1),),
            goals=((0, 0),),
        )
        mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.0),
        )
        heuristic = ShortestPathHeuristic(mdp)
        config = RTDPConfig(
            max_trials=1,
            max_steps_per_trial=5,
            step_limit_multiplier=None,
            time_limit_seconds=1.0,
            seed=0,
        )
        state = mdp.initial_state()

        baseline = BaselineRTDP(mdp, heuristic, config)
        with patch.object(
            baseline,
            "q_value",
            wraps=baseline.q_value,
        ) as wrapped_q:
            first = baseline.greedy_action_candidates(state)
            calls_after_first = wrapped_q.call_count
            second = baseline.greedy_action_candidates(state)

        self.assertIs(first, second)
        self.assertEqual(wrapped_q.call_count, calls_after_first)
        self.assertEqual(set(first), {("up",), ("left",)})

        od = OperatorDecompositionRTDP(mdp, heuristic, config)
        od_state = (state, ())
        with patch.object(
            od,
            "operator_value",
            wraps=od.operator_value,
        ) as wrapped_operator:
            first_od = od.greedy_operator_candidates(od_state)
            calls_after_first_od = wrapped_operator.call_count
            second_od = od.greedy_operator_candidates(od_state)

        self.assertIs(first_od, second_od)
        self.assertEqual(
            wrapped_operator.call_count,
            calls_after_first_od,
        )
        self.assertEqual(set(first_od), {"up", "left"})

    def test_od_global_guidance_resolves_early_prefix_tie_by_self_loop(
        self,
    ) -> None:
        instance = make_instance(
            rows=["....", "...."],
            starts=((0, 0), (0, 1)),
            goals=((3, 0), (3, 1)),
        )
        mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.2),
        )
        heuristic = ShortestPathHeuristic(mdp)
        config = RTDPConfig(
            max_trials=1,
            max_steps_per_trial=10,
            step_limit_multiplier=None,
            time_limit_seconds=1.0,
            seed=0,
        )
        od = OperatorDecompositionRTDP(mdp, heuristic, config)
        state = mdp.initial_state()

        def tied_operator_value(
            od_state,
            action,
            *,
            count_metrics=True,
            deadline=None,
        ):
            del count_metrics, deadline
            _, prefix = od_state

            if prefix == ():
                return 0.0 if action in {"stay", "right"} else 1.0
            if prefix in {("stay",), ("right",)}:
                return 0.0 if action == "right" else 1.0
            raise AssertionError(f"Unexpected prefix: {prefix}")

        with patch.object(
            od,
            "operator_value",
            side_effect=tied_operator_value,
        ):
            candidates = od.greedy_joint_action_candidates(state)

        self.assertEqual(candidates, (("right", "right"),))

    def test_od_policy_canonicalizes_frozen_goal_agent_to_stay(self) -> None:
        instance = make_instance(
            rows=["...", "..."],
            starts=((0, 0), (0, 1)),
            goals=((0, 0), (2, 1)),
        )
        mdp = GridMMDP(
            instance,
            MMDPConfig(
                slip_to_stay_probability=0.2,
                freeze_agents_at_goal=True,
            ),
        )
        heuristic = ShortestPathHeuristic(mdp)
        od = OperatorDecompositionRTDP(
            mdp,
            heuristic,
            RTDPConfig(
                max_trials=1,
                max_steps_per_trial=10,
                step_limit_multiplier=None,
                time_limit_seconds=1.0,
                seed=0,
            ),
        )

        candidates = od.greedy_joint_action_candidates(mdp.initial_state())

        self.assertTrue(candidates)
        self.assertTrue(all(action[0] == "stay" for action in candidates))

    def test_seeded_stochastic_tie_breaking_is_reproducible(self) -> None:
        instance = make_instance(
            rows=[
                "..",
                "..",
            ],
            starts=((1, 1),),
            goals=((0, 0),),
        )
        mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.0),
        )
        heuristic = ShortestPathHeuristic(mdp)
        config = RTDPConfig(
            max_trials=1,
            max_steps_per_trial=5,
            step_limit_multiplier=None,
            time_limit_seconds=1.0,
            seed=0,
        )
        planner = BaselineRTDP(mdp, heuristic, config)
        state = mdp.initial_state()

        rng_a = random.Random(12345)
        rng_b = random.Random(12345)

        sequence_a = [
            planner.policy_action_with_info(state, tie_rng=rng_a)
            for _ in range(20)
        ]
        sequence_b = [
            planner.policy_action_with_info(state, tie_rng=rng_b)
            for _ in range(20)
        ]

        self.assertEqual(sequence_a, sequence_b)
        self.assertTrue(all(tie_count == 1 for _, tie_count in sequence_a))
        self.assertGreater(len({action for action, _ in sequence_a}), 1)

        od = OperatorDecompositionRTDP(mdp, heuristic, config)
        od_rng_a = random.Random(54321)
        od_rng_b = random.Random(54321)
        od_sequence_a = [
            od.policy_action_with_info(state, tie_rng=od_rng_a)
            for _ in range(20)
        ]
        od_sequence_b = [
            od.policy_action_with_info(state, tie_rng=od_rng_b)
            for _ in range(20)
        ]

        self.assertEqual(od_sequence_a, od_sequence_b)
        self.assertTrue(
            all(tie_count == 1 for _, tie_count in od_sequence_a)
        )
        self.assertGreater(
            len({action for action, _ in od_sequence_a}),
            1,
        )

    def test_evaluation_cycle_diagnostics_detect_staying_loop(self) -> None:
        instance = make_instance(
            rows=["..."],
            starts=((0, 0),),
            goals=((2, 0),),
        )
        mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.0),
        )

        class StayPlanner:
            def __init__(self, planner_mdp: GridMMDP) -> None:
                self.mdp = planner_mdp
                self.resolved_max_steps_per_trial = 5

            def policy_action_with_info(
                self,
                state: tuple[tuple[int, int], ...],
                *,
                tie_rng: random.Random | None = None,
            ) -> tuple[tuple[str, ...], int]:
                return ("stay",), 0

            def policy_action(
                self,
                state: tuple[tuple[int, int], ...],
                *,
                tie_rng: random.Random | None = None,
            ) -> tuple[str, ...]:
                return ("stay",)

        evaluation = evaluate_policy(
            mdp,
            StayPlanner(mdp),
            EvaluationConfig(
                episodes=1,
                seed=7,
                max_steps_per_episode=5,
            ),
        )
        episode = evaluation.episode_results[0]

        self.assertFalse(episode.success)
        self.assertEqual(episode.failure_reason, "deterministic_self_loop_policy")
        self.assertEqual(episode.deterministic_self_loop_actions, 1)
        self.assertEqual(episode.steps, 0)
        self.assertEqual(episode.unique_states_visited, 1)
        self.assertEqual(episode.repeated_state_visits, 0)
        self.assertEqual(episode.maximum_state_visit_count, 1)
        self.assertEqual(episode.self_transitions, 0)
        self.assertEqual(
            episode.maximum_consecutive_self_transitions,
            0,
        )

    def test_evaluation_does_not_modify_baseline_solution(self) -> None:
        mdp, heuristic, config = self.make_one_agent_problem()

        planner = BaselineRTDP(
            mdp,
            heuristic,
            config,
        )

        planner.solve()

        values_before = dict(planner.V)
        counters_before = (
            planner.bellman_backups,
            planner.planning_action_evaluations,
            planner.transition_outcomes_evaluated,
        )

        evaluation = evaluate_policy(
            mdp,
            planner,
            EvaluationConfig(
                episodes=5,
                seed=123,
                max_steps_per_episode=10,
            ),
        )

        counters_after = (
            planner.bellman_backups,
            planner.planning_action_evaluations,
            planner.transition_outcomes_evaluated,
        )

        self.assertEqual(
            planner.V,
            values_before,
        )

        self.assertEqual(
            counters_after,
            counters_before,
        )

        self.assertEqual(
            evaluation.summary.success_rate,
            1.0,
        )

        self.assertEqual(
            evaluation.summary.mean_sum_of_costs_successful_episodes,
            2.0,
        )

        self.assertEqual(
            evaluation.summary.mean_makespan_successful_episodes,
            2.0,
        )

    def test_optimized_evaluation_preserves_results_and_reduces_cache_writes(
        self,
    ) -> None:
        instance = make_instance(
            rows=["....", "....", "...."],
            starts=((0, 0), (0, 1), (0, 2)),
            goals=((3, 0), (3, 1), (3, 2)),
        )

        def run(cache_only_executed_actions: bool):
            mdp = GridMMDP(
                instance,
                MMDPConfig(slip_to_stay_probability=0.2),
            )
            planner = BaselineRTDP(
                mdp,
                ShortestPathHeuristic(mdp),
                RTDPConfig(
                    max_trials=1,
                    max_steps_per_trial=20,
                    step_limit_multiplier=None,
                    time_limit_seconds=5.0,
                    stable_trials_required=1,
                    seed=0,
                ),
            )
            return evaluate_policy(
                mdp,
                planner,
                EvaluationConfig(
                    episodes=5,
                    seed=44,
                    max_steps_per_episode=20,
                    cache_only_executed_actions=(
                        cache_only_executed_actions
                    ),
                ),
            )

        optimized = run(True)
        old_style = run(False)

        self.assertEqual(
            [result.success for result in optimized.episode_results],
            [result.success for result in old_style.episode_results],
        )
        self.assertEqual(
            [result.arrival_times for result in optimized.episode_results],
            [result.arrival_times for result in old_style.episode_results],
        )
        self.assertLess(
            optimized.summary.transition_resolved_cache_writes,
            old_style.summary.transition_resolved_cache_writes,
        )


    def test_lrtdp_solved_stopping_labels_initial_state(self) -> None:
        instance = make_instance(
            rows=["..."],
            starts=((0, 0),),
            goals=((2, 0),),
        )
        config = RTDPConfig(
            max_trials=None,
            max_steps_per_trial=10,
            step_limit_multiplier=None,
            time_limit_seconds=None,
            memory_limit_mb=None,
            epsilon=1e-9,
            relative_epsilon=0.0,
            stable_trials_required=100,
            stop_when_stable=False,
            stop_when_solved=True,
            tie_tolerance=1e-9,
            seed=4,
        )

        baseline_mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.0),
        )
        od_mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.0),
        )
        baseline = BaselineRTDP(
            baseline_mdp,
            ShortestPathHeuristic(baseline_mdp),
            config,
        )
        od = OperatorDecompositionRTDP(
            od_mdp,
            ShortestPathHeuristic(od_mdp),
            config,
        )

        baseline_result = baseline.solve()
        od_result = od.solve()

        self.assertEqual(
            baseline_result.stop_reason,
            "initial_state_solved",
        )
        self.assertEqual(
            od_result.stop_reason,
            "initial_state_solved",
        )
        self.assertTrue(baseline_result.initial_state_solved)
        self.assertTrue(od_result.initial_state_solved)
        self.assertGreater(baseline_result.solved_checks, 0)
        self.assertGreater(od_result.solved_checks, 0)
        self.assertEqual(
            baseline.policy_action(baseline_mdp.initial_state()),
            ("right",),
        )
        self.assertEqual(
            od.policy_action(od_mdp.initial_state()),
            ("right",),
        )

    def test_two_agent_baseline_and_od_agree_after_convergence(self) -> None:
        instance = make_instance(
            rows=["...", "..."],
            starts=((0, 0), (0, 1)),
            goals=((2, 0), (2, 1)),
        )
        config = RTDPConfig(
            max_trials=500,
            max_steps_per_trial=30,
            step_limit_multiplier=None,
            time_limit_seconds=10.0,
            epsilon=1e-8,
            stable_trials_required=5,
            stop_when_stable=True,
            seed=3,
        )

        baseline_mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.2),
        )
        od_mdp = GridMMDP(
            instance,
            MMDPConfig(slip_to_stay_probability=0.2),
        )
        baseline = BaselineRTDP(
            baseline_mdp,
            ShortestPathHeuristic(baseline_mdp),
            config,
        )
        od = OperatorDecompositionRTDP(
            od_mdp,
            ShortestPathHeuristic(od_mdp),
            config,
        )

        baseline_result = baseline.solve()
        od_result = od.solve()

        self.assertEqual(baseline_result.stop_reason, "stable_trials")
        self.assertEqual(od_result.stop_reason, "stable_trials")
        self.assertTrue(
            math.isclose(
                baseline.value(baseline_mdp.initial_state()),
                od.real_state_value(od_mdp.initial_state()),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        )

        baseline_eval = evaluate_policy(
            baseline_mdp,
            baseline,
            EvaluationConfig(
                episodes=20,
                seed=99,
                max_steps_per_episode=30,
            ),
        )
        od_eval = evaluate_policy(
            od_mdp,
            od,
            EvaluationConfig(
                episodes=20,
                seed=99,
                max_steps_per_episode=30,
            ),
        )
        self.assertEqual(baseline_eval.summary.success_rate, 1.0)
        self.assertEqual(od_eval.summary.success_rate, 1.0)



if __name__ == "__main__":
    unittest.main(verbosity=2)
