from __future__ import annotations

"""Fixed-policy evaluation for Baseline RTDP and OD-RTDP.

Evaluation never changes the planners' value tables. The optimized path keeps
transition-cache reads enabled while suppressing writes during the expensive
search over candidate actions. After one action is selected, only that action's
transition distribution is memoized and sampled.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import math
import random
import statistics
import time
from typing import Any, Protocol

from grid_mmdp import GridMMDP, JointAction, State, Transition
from resource_monitor import ResourceMonitor


class PolicyPlanner(Protocol):
    def policy_action(
        self,
        state: State,
        *,
        tie_rng: random.Random | None = None,
    ) -> JointAction:
        ...

    def policy_action_with_info(
        self,
        state: State,
        *,
        tie_rng: random.Random | None = None,
    ) -> tuple[JointAction, int]:
        ...


class MethodPolicyAdapter:
    """Expose an alternate planner policy method through the standard protocol."""

    def __init__(
        self,
        planner: Any,
        *,
        action_method: str,
        info_method: str,
        policy_name: str,
    ) -> None:
        self._planner = planner
        self.mdp = planner.mdp
        self.heuristic = getattr(planner, "heuristic", None)
        self.resolved_max_steps_per_trial = getattr(
            planner, "resolved_max_steps_per_trial", None
        )
        self._action_method = action_method
        self._info_method = info_method
        self.policy_name = policy_name

    def policy_action(self, state: State, *, tie_rng=None) -> JointAction:
        return getattr(self._planner, self._action_method)(
            state, tie_rng=tie_rng
        )

    def policy_action_with_info(self, state: State, *, tie_rng=None):
        return getattr(self._planner, self._info_method)(
            state, tie_rng=tie_rng
        )

    def reset_policy_cache_stats(self) -> None:
        method = getattr(self._planner, "reset_policy_cache_stats", None)
        if callable(method):
            method()

    def policy_cache_stats(self):
        method = getattr(self._planner, "policy_cache_stats", None)
        return method() if callable(method) else {
            "hits": 0, "misses": 0, "entries": 0, "hit_rate": 0.0
        }


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluating one fixed value function."""

    episodes: int = 100
    seed: int = 0
    max_steps_per_episode: int | None = None
    measure_conflict_risk: bool = True
    randomize_greedy_ties: bool = False

    # Main optimization: policy extraction may read transitions produced in
    # planning, but it does not cache every rejected candidate action. The
    # selected action is cached immediately afterwards when it is executed.
    cache_only_executed_actions: bool = True

    # Cycle/tie/action-quality diagnostics are useful for pilot runs but can be
    # disabled for the fastest large experiment.
    collect_diagnostics: bool = True

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if (
            self.max_steps_per_episode is not None
            and self.max_steps_per_episode <= 0
        ):
            raise ValueError(
                "max_steps_per_episode must be positive or None"
            )


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    episode_seed: int
    success: bool
    hit_step_limit: bool
    steps: int
    accumulated_cost: float
    makespan: int | None
    arrival_times: tuple[int | None, ...]
    arrived_agents: int
    cumulative_conflict_risk: float
    mean_conflict_risk_per_step: float
    policy_decision_seconds: float
    episode_elapsed_seconds: float

    unique_states_visited: int
    repeated_state_visits: int
    maximum_state_visit_count: int
    self_transitions: int
    maximum_consecutive_self_transitions: int
    tie_decisions: int
    unique_real_states_with_policy_ties: int

    cumulative_expected_self_loop_probability: float
    mean_expected_self_loop_probability_per_step: float
    cumulative_expected_shortest_path_progress: float
    mean_expected_shortest_path_progress_per_step: float

    cumulative_expected_vertex_conflict_probability: float
    cumulative_expected_edge_swap_probability: float
    cumulative_expected_noncollision_no_motion_probability: float
    mean_expected_vertex_conflict_probability_per_step: float
    mean_expected_edge_swap_probability_per_step: float
    mean_expected_noncollision_no_motion_probability_per_step: float
    selected_unfinished_stay_actions: int
    selected_unfinished_blocked_actions: int
    deterministic_self_loop_actions: int
    failure_reason: str | None
    repeated_state_action_pairs: tuple[dict[str, Any], ...]

    final_state: State

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSummary:
    policy_name: str
    map_name: str
    scenario_name: str
    n_agents: int
    evaluation_seed: int
    episodes: int
    max_steps_per_episode: int
    randomize_greedy_ties: bool
    cache_only_executed_actions: bool
    collect_diagnostics: bool

    successful_episodes: int
    failed_episodes: int
    success_rate: float
    total_environment_steps: int
    mean_steps_all_episodes: float
    mean_steps_successful_episodes: float | None
    mean_accumulated_cost_all_episodes: float
    mean_sum_of_costs_successful_episodes: float | None
    std_sum_of_costs_successful_episodes: float | None
    mean_makespan_successful_episodes: float | None
    std_makespan_successful_episodes: float | None
    per_agent_arrival_rates: tuple[float, ...]
    per_agent_mean_arrival_times: tuple[float | None, ...]
    mean_arrived_agents_per_episode: float

    expected_conflict_attempts_per_episode: float
    mean_conflict_risk_per_environment_step: float
    mean_expected_self_loop_probability_per_step: float
    mean_expected_shortest_path_progress_per_step: float
    mean_expected_vertex_conflict_probability_per_step: float
    mean_expected_edge_swap_probability_per_step: float
    mean_expected_noncollision_no_motion_probability_per_step: float
    mean_selected_unfinished_stay_actions_per_episode: float
    mean_selected_unfinished_blocked_actions_per_episode: float
    deterministic_self_loop_failures: int
    step_limit_failures: int

    evaluation_elapsed_seconds: float
    evaluation_baseline_rss_mb: float
    evaluation_peak_rss_mb: float
    evaluation_peak_rss_delta_mb: float
    total_policy_decision_seconds: float
    mean_policy_decision_milliseconds: float

    policy_cache_hits: int
    policy_cache_misses: int
    policy_cache_entries: int
    policy_cache_hit_rate: float

    transition_raw_cache_entries_before: int
    transition_raw_cache_entries_after: int
    transition_resolved_cache_entries_before: int
    transition_resolved_cache_entries_after: int
    transition_raw_cache_hits: int
    transition_raw_cache_misses: int
    transition_raw_cache_writes: int
    transition_raw_cache_evictions: int
    transition_resolved_cache_hits: int
    transition_resolved_cache_misses: int
    transition_resolved_cache_writes: int
    transition_resolved_cache_evictions: int

    mean_unique_states_visited: float
    mean_repeated_state_visits: float
    mean_maximum_state_visit_count: float
    mean_self_transitions: float
    mean_maximum_consecutive_self_transitions: float
    mean_tie_decisions: float
    mean_unique_real_states_with_policy_ties: float
    mean_tie_decisions_per_environment_step: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    config: EvaluationConfig
    summary: EvaluationSummary
    episode_results: tuple[EpisodeResult, ...]

    def summary_dict(self) -> dict[str, Any]:
        return self.summary.to_dict()


def _mean_or_none(values: list[float] | list[int]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _sample_std_or_none(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values))


def _resolve_episode_step_limit(
    planner: PolicyPlanner,
    config: EvaluationConfig,
) -> int:
    if config.max_steps_per_episode is not None:
        return config.max_steps_per_episode

    planner_limit = getattr(planner, "resolved_max_steps_per_trial", None)
    if planner_limit is None:
        raise ValueError(
            "Evaluation requires a finite step limit. Supply "
            "EvaluationConfig(max_steps_per_episode=...)."
        )
    if not isinstance(planner_limit, int) or planner_limit <= 0:
        raise ValueError(
            "planner.resolved_max_steps_per_trial must be a positive integer"
        )
    return planner_limit


def _validate_planner_environment(
    mdp: GridMMDP,
    planner: PolicyPlanner,
) -> None:
    planner_mdp = getattr(planner, "mdp", None)
    if planner_mdp is not None and planner_mdp is not mdp:
        raise ValueError(
            "The planner belongs to a different GridMMDP object."
        )


def _policy_action_with_info(
    planner: PolicyPlanner,
    state: State,
    tie_rng: random.Random | None,
    collect_diagnostics: bool,
) -> tuple[JointAction, int]:
    if collect_diagnostics:
        method = getattr(planner, "policy_action_with_info", None)
        if callable(method):
            return method(state, tie_rng=tie_rng)

    policy_method = getattr(planner, "policy_action")
    try:
        return policy_method(state, tie_rng=tie_rng), 0
    except TypeError:
        return policy_method(state), 0


def _distance_sum(planner: PolicyPlanner, state: State) -> float | None:
    heuristic = getattr(planner, "heuristic", None)
    method = getattr(heuristic, "distance_summary", None)
    if not callable(method):
        return None
    distances = method(state)
    if any(math.isinf(distance) for distance in distances):
        return None
    return float(sum(distances))


def _selected_action_diagnostics(
    planner: PolicyPlanner,
    state: State,
    transitions: tuple[Transition, ...],
) -> tuple[float, float]:
    """Return expected self-loop probability and shortest-path progress."""
    self_loop_probability = sum(
        probability
        for next_state, probability in transitions
        if next_state == state
    )

    current_distance = _distance_sum(planner, state)
    if current_distance is None:
        return self_loop_probability, 0.0

    expected_next_distance = 0.0
    for next_state, probability in transitions:
        next_distance = _distance_sum(planner, next_state)
        if next_distance is None:
            return self_loop_probability, 0.0
        expected_next_distance += probability * next_distance

    return self_loop_probability, current_distance - expected_next_distance


def evaluate_episode(
    mdp: GridMMDP,
    planner: PolicyPlanner,
    *,
    episode_index: int,
    episode_seed: int,
    max_steps: int,
    measure_conflict_risk: bool = True,
    randomize_greedy_ties: bool = False,
    cache_only_executed_actions: bool = True,
    collect_diagnostics: bool = True,
) -> EpisodeResult:
    if episode_index < 0:
        raise ValueError("episode_index cannot be negative")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    _validate_planner_environment(mdp, planner)
    episode_started_at = time.perf_counter()

    transition_rng = random.Random(episode_seed)
    tie_rng = (
        random.Random(episode_seed ^ 0xD1B54A32D192ED03)
        if randomize_greedy_ties
        else None
    )

    state = mdp.initial_state()
    steps = 0
    accumulated_cost = 0.0
    arrival_times: list[int | None] = [
        0 if position == goal else None
        for position, goal in zip(state, mdp.goals)
    ]

    cumulative_conflict_risk = 0.0
    policy_decision_seconds = 0.0
    cumulative_expected_self_loop_probability = 0.0
    cumulative_expected_shortest_path_progress = 0.0
    cumulative_expected_vertex_conflict_probability = 0.0
    cumulative_expected_edge_swap_probability = 0.0
    cumulative_expected_noncollision_no_motion_probability = 0.0
    selected_unfinished_stay_actions = 0
    selected_unfinished_blocked_actions = 0
    deterministic_self_loop_actions = 0
    failure_reason: str | None = None
    state_action_counts: Counter[tuple[State, JointAction]] | None = (
        Counter() if collect_diagnostics else None
    )

    visit_counts: Counter[State] | None = (
        Counter({state: 1}) if collect_diagnostics else None
    )
    repeated_state_visits = 0
    self_transitions = 0
    current_self_transition_streak = 0
    maximum_consecutive_self_transitions = 0
    tie_decisions = 0
    real_states_with_ties: set[State] | None = (
        set() if collect_diagnostics else None
    )

    while not mdp.is_terminal(state) and steps < max_steps:
        decision_started_at = time.perf_counter()
        with mdp.transition_cache_writes(
            not cache_only_executed_actions
        ):
            joint_action, step_tie_decisions = _policy_action_with_info(
                planner,
                state,
                tie_rng,
                collect_diagnostics,
            )
        policy_decision_seconds += time.perf_counter() - decision_started_at

        if collect_diagnostics:
            tie_decisions += step_tie_decisions
            if step_tie_decisions > 0 and real_states_with_ties is not None:
                real_states_with_ties.add(state)

        mdp.validate_joint_action(joint_action)

        # Writes are enabled again here. Thus only the action actually used by
        # the environment is added to the transition cache.
        transitions = mdp.joint_transitions(state, joint_action)

        if measure_conflict_risk:
            cumulative_conflict_risk += mdp.conflict_probability(
                state,
                joint_action,
            )

        if collect_diagnostics:
            self_loop_probability, expected_progress = (
                _selected_action_diagnostics(
                    planner,
                    state,
                    transitions,
                )
            )
            cumulative_expected_self_loop_probability += self_loop_probability
            cumulative_expected_shortest_path_progress += expected_progress
            breakdown = mdp.action_risk_breakdown(state, joint_action)
            cumulative_expected_vertex_conflict_probability += float(
                breakdown["vertex_conflict_probability"]
            )
            cumulative_expected_edge_swap_probability += float(
                breakdown["edge_swap_probability"]
            )
            cumulative_expected_noncollision_no_motion_probability += float(
                breakdown["noncollision_no_motion_probability"]
            )
            selected_unfinished_stay_actions += int(
                breakdown["unfinished_stay_actions"]
            )
            selected_unfinished_blocked_actions += int(
                breakdown["unfinished_blocked_actions"]
            )
            if state_action_counts is not None:
                state_action_counts[(state, joint_action)] += 1
            if (
                tie_rng is None
                and self_loop_probability >= 1.0 - 1e-15
                and not mdp.is_terminal(state)
            ):
                deterministic_self_loop_actions += 1
                failure_reason = "deterministic_self_loop_policy"
                break

        next_state = mdp.sample_from_transitions(
            transitions,
            transition_rng,
        )

        accumulated_cost += mdp.transition_cost(
            state,
            joint_action,
            next_state,
        )
        steps += 1

        for agent_index, (next_position, goal) in enumerate(
            zip(next_state, mdp.goals)
        ):
            if (
                arrival_times[agent_index] is None
                and next_position == goal
            ):
                arrival_times[agent_index] = steps

        if collect_diagnostics and visit_counts is not None:
            if next_state in visit_counts:
                repeated_state_visits += 1
            visit_counts[next_state] += 1

            if next_state == state:
                self_transitions += 1
                current_self_transition_streak += 1
                maximum_consecutive_self_transitions = max(
                    maximum_consecutive_self_transitions,
                    current_self_transition_streak,
                )
            else:
                current_self_transition_streak = 0

        state = next_state

    success = mdp.is_terminal(state)
    hit_step_limit = not success and steps >= max_steps
    if not success and failure_reason is None:
        failure_reason = "step_limit" if hit_step_limit else "terminated"
    arrived_agents = sum(value is not None for value in arrival_times)
    makespan = (
        max(value for value in arrival_times if value is not None)
        if success
        else None
    )

    if success and mdp.config.freeze_agents_at_goal:
        arrival_time_sum = float(
            sum(value for value in arrival_times if value is not None)
        )
        if not math.isclose(
            accumulated_cost,
            arrival_time_sum,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "Accumulated cost does not equal the sum of arrival times."
            )

    unique_states = len(visit_counts) if visit_counts is not None else 0
    maximum_state_visit_count = (
        max(visit_counts.values()) if visit_counts else 0
    )

    repeated_pairs: tuple[dict[str, Any], ...] = ()
    if state_action_counts is not None:
        repeated_pairs = tuple(
            {
                "state": state_key,
                "action": action_key,
                "count": count,
                "risk": mdp.action_risk_breakdown(state_key, action_key),
            }
            for (state_key, action_key), count in sorted(
                state_action_counts.items(),
                key=lambda item: (-item[1], repr(item[0])),
            )
            if count > 1
        )

    return EpisodeResult(
        episode_index=episode_index,
        episode_seed=episode_seed,
        success=success,
        hit_step_limit=hit_step_limit,
        steps=steps,
        accumulated_cost=accumulated_cost,
        makespan=makespan,
        arrival_times=tuple(arrival_times),
        arrived_agents=arrived_agents,
        cumulative_conflict_risk=cumulative_conflict_risk,
        mean_conflict_risk_per_step=(
            cumulative_conflict_risk / steps if steps else 0.0
        ),
        policy_decision_seconds=policy_decision_seconds,
        episode_elapsed_seconds=time.perf_counter() - episode_started_at,
        unique_states_visited=unique_states,
        repeated_state_visits=repeated_state_visits,
        maximum_state_visit_count=maximum_state_visit_count,
        self_transitions=self_transitions,
        maximum_consecutive_self_transitions=(
            maximum_consecutive_self_transitions
        ),
        tie_decisions=tie_decisions,
        unique_real_states_with_policy_ties=(
            len(real_states_with_ties) if real_states_with_ties is not None else 0
        ),
        cumulative_expected_self_loop_probability=(
            cumulative_expected_self_loop_probability
        ),
        mean_expected_self_loop_probability_per_step=(
            cumulative_expected_self_loop_probability / steps if steps else 0.0
        ),
        cumulative_expected_shortest_path_progress=(
            cumulative_expected_shortest_path_progress
        ),
        mean_expected_shortest_path_progress_per_step=(
            cumulative_expected_shortest_path_progress / steps if steps else 0.0
        ),
        cumulative_expected_vertex_conflict_probability=(
            cumulative_expected_vertex_conflict_probability
        ),
        cumulative_expected_edge_swap_probability=(
            cumulative_expected_edge_swap_probability
        ),
        cumulative_expected_noncollision_no_motion_probability=(
            cumulative_expected_noncollision_no_motion_probability
        ),
        mean_expected_vertex_conflict_probability_per_step=(
            cumulative_expected_vertex_conflict_probability / steps if steps else 0.0
        ),
        mean_expected_edge_swap_probability_per_step=(
            cumulative_expected_edge_swap_probability / steps if steps else 0.0
        ),
        mean_expected_noncollision_no_motion_probability_per_step=(
            cumulative_expected_noncollision_no_motion_probability / steps if steps else 0.0
        ),
        selected_unfinished_stay_actions=selected_unfinished_stay_actions,
        selected_unfinished_blocked_actions=selected_unfinished_blocked_actions,
        deterministic_self_loop_actions=deterministic_self_loop_actions,
        failure_reason=failure_reason,
        repeated_state_action_pairs=repeated_pairs,
        final_state=state,
    )


def _planner_cache_stats(planner: PolicyPlanner) -> dict[str, int | float]:
    method = getattr(planner, "policy_cache_stats", None)
    if not callable(method):
        return {"hits": 0, "misses": 0, "entries": 0, "hit_rate": 0.0}
    return method()


def _build_summary(
    mdp: GridMMDP,
    planner: PolicyPlanner,
    config: EvaluationConfig,
    max_steps_per_episode: int,
    episode_results: tuple[EpisodeResult, ...],
    evaluation_elapsed_seconds: float,
    transition_stats_before: dict[str, int | bool | None],
    transition_stats_after: dict[str, int | bool | None],
    resource_snapshot: Any,
) -> EvaluationSummary:
    successful = [result for result in episode_results if result.success]
    successful_costs = [result.accumulated_cost for result in successful]
    successful_steps = [result.steps for result in successful]
    successful_makespans = [
        result.makespan
        for result in successful
        if result.makespan is not None
    ]

    total_environment_steps = sum(result.steps for result in episode_results)
    total_policy_decision_seconds = sum(
        result.policy_decision_seconds for result in episode_results
    )
    total_conflict_risk = sum(
        result.cumulative_conflict_risk for result in episode_results
    )
    total_tie_decisions = sum(
        result.tie_decisions for result in episode_results
    )
    total_expected_self_loop = sum(
        result.cumulative_expected_self_loop_probability
        for result in episode_results
    )
    total_expected_progress = sum(
        result.cumulative_expected_shortest_path_progress
        for result in episode_results
    )
    total_vertex = sum(
        result.cumulative_expected_vertex_conflict_probability
        for result in episode_results
    )
    total_edge = sum(
        result.cumulative_expected_edge_swap_probability
        for result in episode_results
    )
    total_no_motion = sum(
        result.cumulative_expected_noncollision_no_motion_probability
        for result in episode_results
    )

    per_agent_arrival_rates: list[float] = []
    per_agent_mean_arrival_times: list[float | None] = []
    for agent_index in range(mdp.n_agents):
        times = [
            result.arrival_times[agent_index]
            for result in episode_results
            if result.arrival_times[agent_index] is not None
        ]
        per_agent_arrival_rates.append(len(times) / config.episodes)
        per_agent_mean_arrival_times.append(_mean_or_none(times))

    policy_stats = _planner_cache_stats(planner)
    successful_episodes = len(successful)

    return EvaluationSummary(
        policy_name=getattr(planner, "policy_name", type(planner).__name__),
        map_name=mdp.map_name,
        scenario_name=mdp.instance.scenario_file.name,
        n_agents=mdp.n_agents,
        evaluation_seed=config.seed,
        episodes=config.episodes,
        max_steps_per_episode=max_steps_per_episode,
        randomize_greedy_ties=config.randomize_greedy_ties,
        cache_only_executed_actions=config.cache_only_executed_actions,
        collect_diagnostics=config.collect_diagnostics,
        successful_episodes=successful_episodes,
        failed_episodes=config.episodes - successful_episodes,
        success_rate=successful_episodes / config.episodes,
        total_environment_steps=total_environment_steps,
        mean_steps_all_episodes=float(
            statistics.fmean(result.steps for result in episode_results)
        ),
        mean_steps_successful_episodes=_mean_or_none(successful_steps),
        mean_accumulated_cost_all_episodes=float(
            statistics.fmean(
                result.accumulated_cost for result in episode_results
            )
        ),
        mean_sum_of_costs_successful_episodes=(
            _mean_or_none(successful_costs)
        ),
        std_sum_of_costs_successful_episodes=(
            _sample_std_or_none(successful_costs)
        ),
        mean_makespan_successful_episodes=(
            _mean_or_none(successful_makespans)
        ),
        std_makespan_successful_episodes=(
            _sample_std_or_none(successful_makespans)
        ),
        per_agent_arrival_rates=tuple(per_agent_arrival_rates),
        per_agent_mean_arrival_times=tuple(per_agent_mean_arrival_times),
        mean_arrived_agents_per_episode=float(
            statistics.fmean(
                result.arrived_agents for result in episode_results
            )
        ),
        expected_conflict_attempts_per_episode=float(
            statistics.fmean(
                result.cumulative_conflict_risk for result in episode_results
            )
        ),
        mean_conflict_risk_per_environment_step=(
            total_conflict_risk / total_environment_steps
            if total_environment_steps
            else 0.0
        ),
        mean_expected_self_loop_probability_per_step=(
            total_expected_self_loop / total_environment_steps
            if total_environment_steps
            else 0.0
        ),
        mean_expected_shortest_path_progress_per_step=(
            total_expected_progress / total_environment_steps
            if total_environment_steps
            else 0.0
        ),
        mean_expected_vertex_conflict_probability_per_step=(
            total_vertex / total_environment_steps if total_environment_steps else 0.0
        ),
        mean_expected_edge_swap_probability_per_step=(
            total_edge / total_environment_steps if total_environment_steps else 0.0
        ),
        mean_expected_noncollision_no_motion_probability_per_step=(
            total_no_motion / total_environment_steps if total_environment_steps else 0.0
        ),
        mean_selected_unfinished_stay_actions_per_episode=float(
            statistics.fmean(r.selected_unfinished_stay_actions for r in episode_results)
        ),
        mean_selected_unfinished_blocked_actions_per_episode=float(
            statistics.fmean(r.selected_unfinished_blocked_actions for r in episode_results)
        ),
        deterministic_self_loop_failures=sum(
            r.failure_reason == "deterministic_self_loop_policy" for r in episode_results
        ),
        step_limit_failures=sum(
            r.failure_reason == "step_limit" for r in episode_results
        ),
        evaluation_elapsed_seconds=evaluation_elapsed_seconds,
        evaluation_baseline_rss_mb=resource_snapshot.baseline_rss_mb,
        evaluation_peak_rss_mb=resource_snapshot.peak_rss_mb,
        evaluation_peak_rss_delta_mb=resource_snapshot.peak_rss_delta_mb,
        total_policy_decision_seconds=total_policy_decision_seconds,
        mean_policy_decision_milliseconds=(
            1_000.0 * total_policy_decision_seconds / total_environment_steps
            if total_environment_steps
            else 0.0
        ),
        policy_cache_hits=int(policy_stats["hits"]),
        policy_cache_misses=int(policy_stats["misses"]),
        policy_cache_entries=int(policy_stats["entries"]),
        policy_cache_hit_rate=float(policy_stats["hit_rate"]),
        transition_raw_cache_entries_before=int(
            transition_stats_before["raw_entries"]
        ),
        transition_raw_cache_entries_after=int(
            transition_stats_after["raw_entries"]
        ),
        transition_resolved_cache_entries_before=int(
            transition_stats_before["resolved_entries"]
        ),
        transition_resolved_cache_entries_after=int(
            transition_stats_after["resolved_entries"]
        ),
        transition_raw_cache_hits=int(transition_stats_after["raw_hits"]),
        transition_raw_cache_misses=int(transition_stats_after["raw_misses"]),
        transition_raw_cache_writes=int(transition_stats_after["raw_writes"]),
        transition_raw_cache_evictions=int(
            transition_stats_after.get("raw_evictions", 0)
        ),
        transition_resolved_cache_hits=int(
            transition_stats_after["resolved_hits"]
        ),
        transition_resolved_cache_misses=int(
            transition_stats_after["resolved_misses"]
        ),
        transition_resolved_cache_writes=int(
            transition_stats_after["resolved_writes"]
        ),
        transition_resolved_cache_evictions=int(
            transition_stats_after.get("resolved_evictions", 0)
        ),
        mean_unique_states_visited=float(
            statistics.fmean(
                result.unique_states_visited for result in episode_results
            )
        ),
        mean_repeated_state_visits=float(
            statistics.fmean(
                result.repeated_state_visits for result in episode_results
            )
        ),
        mean_maximum_state_visit_count=float(
            statistics.fmean(
                result.maximum_state_visit_count for result in episode_results
            )
        ),
        mean_self_transitions=float(
            statistics.fmean(
                result.self_transitions for result in episode_results
            )
        ),
        mean_maximum_consecutive_self_transitions=float(
            statistics.fmean(
                result.maximum_consecutive_self_transitions
                for result in episode_results
            )
        ),
        mean_tie_decisions=float(
            statistics.fmean(
                result.tie_decisions for result in episode_results
            )
        ),
        mean_unique_real_states_with_policy_ties=float(
            statistics.fmean(
                result.unique_real_states_with_policy_ties
                for result in episode_results
            )
        ),
        mean_tie_decisions_per_environment_step=(
            total_tie_decisions / total_environment_steps
            if total_environment_steps
            else 0.0
        ),
    )


def evaluate_policy(
    mdp: GridMMDP,
    planner: PolicyPlanner,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """Evaluate a fixed planner value function over stochastic episodes."""
    evaluation_config = config if config is not None else EvaluationConfig()
    _validate_planner_environment(mdp, planner)
    max_steps_per_episode = _resolve_episode_step_limit(
        planner,
        evaluation_config,
    )

    reset_policy_stats = getattr(planner, "reset_policy_cache_stats", None)
    if callable(reset_policy_stats):
        reset_policy_stats()

    transition_stats_before = mdp.transition_cache_stats()
    mdp.reset_transition_cache_stats()

    master_rng = random.Random(evaluation_config.seed)
    episode_seeds = [
        master_rng.randrange(0, 2**63)
        for _ in range(evaluation_config.episodes)
    ]

    evaluation_started_at = time.perf_counter()
    monitor = ResourceMonitor().start()
    try:
        episode_results = tuple(
            evaluate_episode(
                mdp,
                planner,
                episode_index=episode_index,
                episode_seed=episode_seed,
                max_steps=max_steps_per_episode,
                measure_conflict_risk=evaluation_config.measure_conflict_risk,
                randomize_greedy_ties=evaluation_config.randomize_greedy_ties,
                cache_only_executed_actions=(
                    evaluation_config.cache_only_executed_actions
                ),
                collect_diagnostics=evaluation_config.collect_diagnostics,
            )
            for episode_index, episode_seed in enumerate(episode_seeds)
        )
    finally:
        resource_snapshot = monitor.stop()
    evaluation_elapsed_seconds = time.perf_counter() - evaluation_started_at
    transition_stats_after = mdp.transition_cache_stats()

    summary = _build_summary(
        mdp,
        planner,
        evaluation_config,
        max_steps_per_episode,
        episode_results,
        evaluation_elapsed_seconds,
        transition_stats_before,
        transition_stats_after,
        resource_snapshot,
    )

    return EvaluationResult(
        config=evaluation_config,
        summary=summary,
        episode_results=episode_results,
    )
