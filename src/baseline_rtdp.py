from __future__ import annotations

"""
Baseline RTDP for the stochastic cooperative grid MMDP.

The planner solves a cost-minimization Stochastic Shortest Path problem:

    V(s) = min_a sum_s' P(s' | s, a)
           [c(s, a, s') + V(s')]

There is no discount factor.

For a state that has not yet received a Bellman backup, its value is
initialized using the obstacle-aware shortest-path heuristic.

This is the baseline implementation: every Bellman backup enumerates all
complete joint actions. Therefore, with five actions per agent, the branching
factor is:

    5 ** number_of_agents
"""

from dataclasses import dataclass
import hashlib
import math
import random
import time
from typing import Callable, Iterator

from grid_mmdp import ACTIONS, GridMMDP, JointAction, State
from heuristic import ShortestPathHeuristic
from limits import sequential_multi_agent_step_bound
from numerics import scaled_residual_ratio, tied_by_ulp
from resource_monitor import ResourceMonitor


StateHeuristic = Callable[[State], float]


class _DeadlineReached(RuntimeError):
    """Internal signal used to stop planning when the time limit is reached."""


class _MemoryLimitReached(RuntimeError):
    """Internal signal used to stop planning at the configured RSS delta."""


@dataclass(frozen=True)
class RTDPConfig:
    """Configuration shared by Baseline RTDP and OD-RTDP.

    Algorithmic limits are optional.  ``step_tail_probability`` replaces the
    former fixed 5x step multiplier with a map-derived stochastic tail bound.
    ``memory_limit_mb`` is additional process RSS above the start of planning;
    final memory-limited experiments should isolate each run in a subprocess.
    """

    max_trials: int | None = None
    max_steps_per_trial: int | None = None

    # Backward-compatible override. None uses the probabilistic bound.
    step_limit_multiplier: float | None = None
    step_tail_probability: float = 1e-6

    time_limit_seconds: float | None = 60.0
    memory_limit_mb: float | None = None

    # Scale-aware Bellman-residual criterion.
    epsilon: float = 1e-8
    relative_epsilon: float = 1e-6
    stable_trials_required: int = 44
    stop_when_stable: bool = False
    # LRTDP-style stopping: stop when the initial state is labelled solved.
    # This is the preferred rule for run-to-convergence experiments.
    stop_when_solved: bool = False
    require_goal_for_stability: bool = True

    # None means an ULP-based numerical comparison; a positive value keeps the
    # old explicit absolute-tolerance behavior for sensitivity experiments.
    tie_tolerance: float | None = None
    tie_ulps: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_trials is not None and self.max_trials <= 0:
            raise ValueError("max_trials must be positive or None")
        if self.max_steps_per_trial is not None and self.max_steps_per_trial <= 0:
            raise ValueError("max_steps_per_trial must be positive or None")
        if self.step_limit_multiplier is not None and self.step_limit_multiplier <= 0.0:
            raise ValueError("step_limit_multiplier must be positive or None")
        if not 0.0 < self.step_tail_probability < 1.0:
            raise ValueError("step_tail_probability must be in (0, 1)")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0.0:
            raise ValueError("time_limit_seconds must be positive or None")
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0.0:
            raise ValueError("memory_limit_mb must be positive or None")
        if self.epsilon < 0.0 or self.relative_epsilon < 0.0:
            raise ValueError("residual tolerances cannot be negative")
        if self.stable_trials_required <= 0:
            raise ValueError("stable_trials_required must be positive")
        if self.tie_tolerance is not None and self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance cannot be negative or None")
        if self.tie_ulps <= 0:
            raise ValueError("tie_ulps must be positive")
        if (
            self.max_trials is None
            and self.time_limit_seconds is None
            and self.memory_limit_mb is None
            and not self.stop_when_stable
            and not self.stop_when_solved
        ):
            raise ValueError(
                "At least one stopping mechanism must be enabled: max_trials, "
                "time_limit_seconds, memory_limit_mb, stop_when_stable, "
                "or stop_when_solved"
            )

@dataclass(frozen=True)
class TrialResult:
    maximum_residual: float
    maximum_scaled_residual: float
    steps: int
    reached_goal: bool
    reached_solved_state: bool
    hit_step_limit: bool
    visited_states: tuple[State, ...]


@dataclass(frozen=True)
class RTDPPlanningResult:
    stop_reason: str
    trials_completed: int
    goal_reaching_trials: int
    step_limited_trials: int
    total_trial_steps: int
    elapsed_seconds: float
    bellman_backups: int
    planning_action_evaluations: int
    transition_outcomes_evaluated: int
    visited_states: int
    final_trial_residual: float
    final_trial_scaled_residual: float
    consecutive_stable_trials: int
    maximum_consecutive_stable_trials: int
    stability_criterion_reached: bool
    first_stability_trial: int | None
    first_stability_elapsed_seconds: float | None
    initial_state_solved: bool
    solved_states: int
    solved_checks: int
    first_solved_trial: int | None
    first_solved_elapsed_seconds: float | None
    resolved_max_steps_per_trial: int | None
    memory_limit_mb: float | None
    memory_limit_reached: bool
    baseline_rss_mb: float
    peak_rss_mb: float
    peak_rss_delta_mb: float


class BaselineRTDP:
    """
    Standard RTDP over complete states and complete joint actions.

    At each visited state, the planner:

    1. Enumerates every complete joint action.
    2. Computes the expected cost of each action.
    3. Selects an action with minimum expected cost.
    4. Stores the resulting Bellman value in V.
    5. Samples one stochastic successor.
    6. Continues the internal trial from that successor.
    """

    def __init__(
        self,
        mdp: GridMMDP,
        heuristic: StateHeuristic | None = None,
        config: RTDPConfig | None = None,
    ) -> None:
        self.mdp = mdp

        self.heuristic = (
            heuristic
            if heuristic is not None
            else ShortestPathHeuristic(mdp)
        )

        self.config = config or RTDPConfig()

        # Calculate the actual per-trial step limit once for this instance.
        self.resolved_max_steps_per_trial = (
            self._resolve_max_steps_per_trial()
        )

        # Store only states that have received a Bellman backup.
        self.V: dict[State, float] = {}

        # LRTDP solved labels. Terminal states are treated as solved without
        # being inserted into this set.
        self._solved_states: set[State] = set()
        self.solved_checks = 0

        # During evaluation, cache the complete set of greedy joint actions
        # for each real state. The selected action is still sampled from this
        # set on every visit when evaluation uses stochastic tie breaking.
        self._policy_candidate_cache: dict[
            State,
            tuple[JointAction, ...],
        ] = {}
        self._policy_cache_hits = 0
        self._policy_cache_misses = 0

        # The immediate SSP cost depends only on the current real state.
        # Cache it because one greedy decision evaluates up to 5**n actions.
        self._state_step_cost_cache: dict[State, float] = {}

        self.bellman_backups = 0
        self.planning_action_evaluations = 0
        self.transition_outcomes_evaluated = 0

        # Separate generators make transition sampling independent of the
        # number of random tie-breaking decisions.
        self.transition_rng = random.Random(
            self.config.seed
        )
        self.tie_rng = random.Random(
            self.config.seed + 1
        )
        self._resource_monitor: ResourceMonitor | None = None

    def _resolve_max_steps_per_trial(self) -> int | None:
        """Resolve a finite, map-derived safety cap for one sampled trial."""
        if self.config.max_steps_per_trial is not None:
            return self.config.max_steps_per_trial

        distance_summary_method = getattr(self.heuristic, "distance_summary", None)
        if not callable(distance_summary_method):
            if self.config.step_limit_multiplier is None:
                return None
            raise ValueError(
                "Automatic step-limit calculation requires distance_summary"
            )

        distances = distance_summary_method(self.mdp.initial_state())
        if any(math.isinf(distance) for distance in distances):
            raise ValueError("At least one start cannot reach its assigned goal")
        success_probability = 1.0 - self.mdp.config.slip_to_stay_probability
        if success_probability <= 0.0:
            raise ValueError("Movement success probability must be positive")

        if self.config.step_limit_multiplier is not None:
            longest = max(distances, default=0.0)
            return max(
                1,
                math.ceil(
                    self.config.step_limit_multiplier
                    * longest
                    / success_probability
                ),
            )

        return sequential_multi_agent_step_bound(
            distances,
            success_probability,
            self.config.step_tail_probability,
        )

    def reset(self) -> None:
        """
        Remove the current solution and restore counters and RNG states.
        """
        self.V.clear()
        self._solved_states.clear()
        self.solved_checks = 0
        self._policy_candidate_cache.clear()
        self.reset_policy_cache_stats()
        self._state_step_cost_cache.clear()

        self.bellman_backups = 0
        self.planning_action_evaluations = 0
        self.transition_outcomes_evaluated = 0

        self.transition_rng.seed(
            self.config.seed
        )
        self.tie_rng.seed(
            self.config.seed + 1
        )

    def _check_deadline(self, deadline: float | None) -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise _DeadlineReached
        if (
            self._resource_monitor is not None
            and self._resource_monitor.limit_reached()
        ):
            raise _MemoryLimitReached

    def _values_tied(self, first: float, second: float) -> bool:
        if self.config.tie_tolerance is not None:
            return math.isclose(
                first, second, rel_tol=0.0, abs_tol=self.config.tie_tolerance
            )
        return tied_by_ulp(first, second, ulps=self.config.tie_ulps)

    def value(
        self,
        state: State,
    ) -> float:
        """
        Return the current estimate of V(state).

        Terminal state:
            0

        State already updated:
            Stored Bellman value

        State not yet updated:
            Shortest-path heuristic
        """
        if self.mdp.is_terminal(state):
            return 0.0

        if state in self.V:
            return self.V[state]

        return self.heuristic(state)

    def _value_unchecked(self, state: State) -> float:
        """Fast internal value lookup for successors produced by the MMDP."""
        if state == self.mdp.goals:
            return 0.0
        stored = self.V.get(state)
        if stored is not None:
            return stored
        return self.heuristic(state)

    def _step_cost_unchecked(self, state: State) -> float:
        cached = self._state_step_cost_cache.get(state)
        if cached is not None:
            return cached
        cost = float(
            sum(
                position != goal
                for position, goal in zip(state, self.mdp.goals)
            )
        )
        self._state_step_cost_cache[state] = cost
        return cost

    def q_value(
        self,
        state: State,
        joint_action: JointAction,
        *,
        count_metrics: bool = True,
        deadline: float | None = None,
    ) -> float:
        """
        Compute the expected cost of one complete joint action.

            Q(s,a) =
                sum_s' P(s'|s,a)
                [transition_cost(s,a,s') + V(s')]

        If a successor has not been updated yet, V(s') is initialized by the
        shortest-path heuristic.
        """
        self._check_deadline(
            deadline
        )

        transitions = self.mdp.joint_transitions(
            state,
            joint_action,
        )

        self._check_deadline(
            deadline
        )

        if count_metrics:
            self.planning_action_evaluations += 1
            self.transition_outcomes_evaluated += len(
                transitions
            )

        # In this MMDP the immediate cost is the number of unfinished agents
        # in the current state.  It is identical for every successor of every
        # action at this state, so compute it once instead of once per outcome.
        immediate_cost = self._step_cost_unchecked(state)
        expected_cost = 0.0

        for next_state, probability in transitions:
            self._check_deadline(deadline)
            future_cost = self._value_unchecked(next_state)
            # Keep the original floating-point operation order so tie and
            # residual behavior remain reproducible.
            expected_cost += probability * (immediate_cost + future_cost)

        return expected_cost

    def _deterministic_tie_choice(
        self,
        candidates: list[JointAction],
        state: State,
    ) -> JointAction:
        """
        Choose reproducibly among equal-valued joint actions.

        The choice depends on the planning seed, the current state, and the
        tied candidates. This avoids always preferring the first action in
        ACTIONS (usually ``stay``) while keeping evaluation deterministic.
        """
        if not candidates:
            raise ValueError(
                "candidates cannot be empty"
            )

        payload = repr(
            (
                self.config.seed,
                state,
                tuple(candidates),
            )
        ).encode("utf-8")

        digest = hashlib.sha256(
            payload
        ).digest()

        index = int.from_bytes(
            digest[:8],
            byteorder="big",
        ) % len(candidates)

        return candidates[index]

    def _guided_joint_actions(
        self,
        state: State,
        candidates: list[JointAction],
    ) -> list[JointAction]:
        """
        Refine a Q-tied set using structured heuristic guidance.

        Bellman/Q values remain the primary criterion. The heuristic key is
        consulted only after actions are equal within tie_tolerance, so this
        reduces arbitrary shortest-path ties without perturbing the objective.
        Custom heuristics that do not expose joint_action_guidance_key keep the
        original candidate set unchanged.
        """
        if len(candidates) <= 1:
            return candidates

        guidance_method = getattr(
            self.heuristic,
            "joint_action_guidance_key",
            None,
        )

        if not callable(guidance_method):
            return candidates

        keyed_candidates = [
            (guidance_method(state, action), action)
            for action in candidates
        ]
        best_key = min(
            key
            for key, _ in keyed_candidates
        )

        return [
            action
            for key, action in keyed_candidates
            if key == best_key
        ]

    def best_action(
        self,
        state: State,
        *,
        count_metrics: bool = True,
        random_ties: bool = True,
        deadline: float | None = None,
    ) -> tuple[JointAction, float]:
        """
        Return a minimum-expected-cost complete joint action.
        """
        if self.mdp.is_terminal(state):
            stay_action: JointAction = tuple(
                "stay"
                for _ in range(self.mdp.n_agents)
            )
            return stay_action, 0.0

        best_value = math.inf
        best_actions: list[JointAction] = []

        for joint_action in self.mdp.all_joint_actions():
            self._check_deadline(
                deadline
            )

            action_value = self.q_value(
                state,
                joint_action,
                count_metrics=count_metrics,
                deadline=deadline,
            )

            if action_value < best_value and not self._values_tied(
                action_value, best_value
            ):
                best_value = action_value
                best_actions = [
                    joint_action
                ]

            elif self._values_tied(action_value, best_value):
                best_actions.append(
                    joint_action
                )

        if not best_actions:
            raise RuntimeError(
                "No joint action was generated"
            )

        best_actions = self._guided_joint_actions(
            state,
            best_actions,
        )

        if random_ties:
            selected_action = self.tie_rng.choice(
                best_actions
            )
        else:
            selected_action = self._deterministic_tie_choice(
                best_actions,
                state,
            )

        return selected_action, best_value

    def backup(
        self,
        state: State,
        *,
        deadline: float | None = None,
    ) -> tuple[JointAction, float, float]:
        old_value = self.value(state)
        joint_action, new_value = self.best_action(
            state, count_metrics=True, random_ties=True, deadline=deadline
        )
        self.V[state] = new_value
        self.bellman_backups += 1
        residual = abs(new_value - old_value)
        scaled = scaled_residual_ratio(
            old_value,
            new_value,
            absolute_tolerance=self.config.epsilon,
            relative_tolerance=self.config.relative_epsilon,
        )
        return joint_action, residual, scaled

    def _trial_step_numbers(
        self,
    ) -> Iterator[int]:
        """
        Generate step indices for one internal trial.

        If resolved_max_steps_per_trial is None, the iterator is unbounded.
        """
        step_number = 0

        while (
            self.resolved_max_steps_per_trial is None
            or step_number
            < self.resolved_max_steps_per_trial
        ):
            yield step_number
            step_number += 1

    def run_trial(
        self,
        *,
        deadline: float | None = None,
    ) -> TrialResult:
        state = self.mdp.initial_state()
        maximum_residual = 0.0
        maximum_scaled_residual = 0.0
        steps = 0
        visited_states: list[State] = []

        if self.mdp.is_terminal(state):
            return TrialResult(0.0, 0.0, 0, True, False, False, ())

        for _ in self._trial_step_numbers():
            self._check_deadline(deadline)

            # In LRTDP mode there is no reason to continue through a state
            # whose greedy envelope has already been proved solved.
            if self.config.stop_when_solved and state in self._solved_states:
                return TrialResult(
                    maximum_residual,
                    maximum_scaled_residual,
                    steps,
                    False,
                    True,
                    False,
                    tuple(visited_states),
                )

            visited_states.append(state)
            joint_action, residual, scaled = self.backup(
                state, deadline=deadline
            )
            maximum_residual = max(maximum_residual, residual)
            maximum_scaled_residual = max(maximum_scaled_residual, scaled)
            state = self.mdp.sample_next(
                state, joint_action, self.transition_rng
            )
            steps += 1
            if self.mdp.is_terminal(state):
                return TrialResult(
                    maximum_residual,
                    maximum_scaled_residual,
                    steps,
                    True,
                    False,
                    False,
                    tuple(visited_states),
                )

        return TrialResult(
            maximum_residual,
            maximum_scaled_residual,
            steps,
            False,
            False,
            True,
            tuple(visited_states),
        )

    def _greedy_successors_for_solved_check(
        self,
        state: State,
        action: JointAction,
    ) -> tuple[State, ...]:
        """Return all positive-probability successors of one fixed greedy action."""
        transitions = self.mdp.joint_transitions(state, action)
        return tuple(
            next_state
            for next_state, probability in transitions
            if probability > 0.0
        )

    def check_solved(
        self,
        root: State,
        *,
        deadline: float | None = None,
    ) -> bool:
        """LRTDP-style solved-state check for the greedy policy envelope.

        A nonterminal state is labelled solved only when its Bellman residual
        is within the configured scale-aware tolerance and every
        positive-probability successor of the deterministic greedy action is
        terminal, already solved, or belongs to the same locally consistent
        envelope.  If the envelope is not solved, reverse Bellman backups are
        performed to propagate information before the next trial.
        """
        self.solved_checks += 1
        if self.mdp.is_terminal(root) or root in self._solved_states:
            return True

        open_stack: list[State] = [root]
        open_set: set[State] = {root}
        closed: list[State] = []
        closed_set: set[State] = set()
        envelope_is_solved = True

        while open_stack:
            self._check_deadline(deadline)
            state = open_stack.pop()
            open_set.discard(state)
            if self.mdp.is_terminal(state) or state in self._solved_states:
                continue
            if state in closed_set:
                continue

            closed.append(state)
            closed_set.add(state)

            old_value = self.value(state)
            action, bellman_value = self.best_action(
                state,
                count_metrics=True,
                random_ties=False,
                deadline=deadline,
            )
            scaled = scaled_residual_ratio(
                old_value,
                bellman_value,
                absolute_tolerance=self.config.epsilon,
                relative_tolerance=self.config.relative_epsilon,
            )
            if scaled > 1.0:
                envelope_is_solved = False
                continue

            for successor in self._greedy_successors_for_solved_check(
                state, action
            ):
                if (
                    self.mdp.is_terminal(successor)
                    or successor in self._solved_states
                    or successor in closed_set
                    or successor in open_set
                ):
                    continue
                open_stack.append(successor)
                open_set.add(successor)

        if envelope_is_solved:
            self._solved_states.update(closed_set)
            return True

        # Standard LRTDP repair step: back up the inconsistent envelope in
        # reverse discovery order, then continue sampling on the next trial.
        for state in reversed(closed):
            self._check_deadline(deadline)
            if state not in self._solved_states:
                self.backup(state, deadline=deadline)
        return False

    def _label_trial_path(
        self,
        visited_states: tuple[State, ...],
        *,
        deadline: float | None = None,
    ) -> None:
        """Try to label states on a sampled trial path, from end to start."""
        for state in reversed(visited_states):
            self._check_deadline(deadline)
            if not self.check_solved(state, deadline=deadline):
                break

    def _trial_numbers(
        self,
    ) -> Iterator[int]:
        """
        Generate bounded or unbounded internal trial numbers.
        """
        trial_number = 1

        while (
            self.config.max_trials is None
            or trial_number <= self.config.max_trials
        ):
            yield trial_number
            trial_number += 1

    def solve(self, *, reset: bool = True) -> RTDPPlanningResult:
        if reset:
            self.reset()
        self._policy_candidate_cache.clear()
        self.reset_policy_cache_stats()

        started_at = time.perf_counter()
        deadline = (
            started_at + self.config.time_limit_seconds
            if self.config.time_limit_seconds is not None
            else None
        )
        monitor = ResourceMonitor(memory_limit_mb=self.config.memory_limit_mb)
        self._resource_monitor = monitor.start()

        trials_completed = 0
        goal_reaching_trials = 0
        step_limited_trials = 0
        total_trial_steps = 0
        consecutive_stable_trials = 0
        maximum_consecutive_stable_trials = 0
        first_stability_trial: int | None = None
        first_stability_elapsed_seconds: float | None = None
        first_solved_trial: int | None = None
        first_solved_elapsed_seconds: float | None = None
        final_trial_residual = math.inf
        final_trial_scaled_residual = math.inf
        stop_reason = "max_trials"

        try:
            for trial_number in self._trial_numbers():
                try:
                    trial_result = self.run_trial(deadline=deadline)
                except _DeadlineReached:
                    stop_reason = "time_limit"
                    break
                except _MemoryLimitReached:
                    stop_reason = "memory_limit"
                    break

                trials_completed += 1
                total_trial_steps += trial_result.steps
                final_trial_residual = trial_result.maximum_residual
                final_trial_scaled_residual = (
                    trial_result.maximum_scaled_residual
                )
                if trial_result.reached_goal:
                    goal_reaching_trials += 1
                if trial_result.hit_step_limit:
                    step_limited_trials += 1

                stable = (
                    trial_result.maximum_scaled_residual <= 1.0
                    and (
                        trial_result.reached_goal
                        or not self.config.require_goal_for_stability
                    )
                )
                consecutive_stable_trials = (
                    consecutive_stable_trials + 1 if stable else 0
                )
                maximum_consecutive_stable_trials = max(
                    maximum_consecutive_stable_trials,
                    consecutive_stable_trials,
                )
                if (
                    first_stability_trial is None
                    and consecutive_stable_trials
                    >= self.config.stable_trials_required
                ):
                    first_stability_trial = trial_number
                    first_stability_elapsed_seconds = (
                        time.perf_counter() - started_at
                    )
                if self.config.stop_when_solved:
                    try:
                        self._label_trial_path(
                            trial_result.visited_states,
                            deadline=deadline,
                        )
                    except _DeadlineReached:
                        stop_reason = "time_limit"
                        break
                    except _MemoryLimitReached:
                        stop_reason = "memory_limit"
                        break

                    initial_state = self.mdp.initial_state()
                    if (
                        self.mdp.is_terminal(initial_state)
                        or initial_state in self._solved_states
                    ):
                        if first_solved_trial is None:
                            first_solved_trial = trial_number
                            first_solved_elapsed_seconds = (
                                time.perf_counter() - started_at
                            )
                        stop_reason = "initial_state_solved"
                        break

                if (
                    self.config.stop_when_stable
                    and consecutive_stable_trials
                    >= self.config.stable_trials_required
                ):
                    stop_reason = "stable_trials"
                    break
                    
                if deadline is not None and time.perf_counter() >= deadline:
                    stop_reason = "time_limit"
                    break
                if monitor.limit_reached():
                    stop_reason = "memory_limit"
                    break
        finally:
            snapshot = monitor.stop()
            self._resource_monitor = None

        elapsed_seconds = time.perf_counter() - started_at
        return RTDPPlanningResult(
            stop_reason=stop_reason,
            trials_completed=trials_completed,
            goal_reaching_trials=goal_reaching_trials,
            step_limited_trials=step_limited_trials,
            total_trial_steps=total_trial_steps,
            elapsed_seconds=elapsed_seconds,
            bellman_backups=self.bellman_backups,
            planning_action_evaluations=self.planning_action_evaluations,
            transition_outcomes_evaluated=self.transition_outcomes_evaluated,
            visited_states=len(self.V),
            final_trial_residual=final_trial_residual,
            final_trial_scaled_residual=final_trial_scaled_residual,
            consecutive_stable_trials=consecutive_stable_trials,
            maximum_consecutive_stable_trials=(
                maximum_consecutive_stable_trials
            ),
            stability_criterion_reached=first_stability_trial is not None,
            first_stability_trial=first_stability_trial,
            first_stability_elapsed_seconds=first_stability_elapsed_seconds,
            initial_state_solved=(
                self.mdp.is_terminal(self.mdp.initial_state())
                or self.mdp.initial_state() in self._solved_states
            ),
            solved_states=len(self._solved_states),
            solved_checks=self.solved_checks,
            first_solved_trial=first_solved_trial,
            first_solved_elapsed_seconds=first_solved_elapsed_seconds,
            resolved_max_steps_per_trial=self.resolved_max_steps_per_trial,
            memory_limit_mb=self.config.memory_limit_mb,
            memory_limit_reached=snapshot.memory_limit_reached,
            baseline_rss_mb=snapshot.baseline_rss_mb,
            peak_rss_mb=snapshot.peak_rss_mb,
            peak_rss_delta_mb=snapshot.peak_rss_delta_mb,
        )

    def reset_policy_cache_stats(self) -> None:
        """Reset policy-candidate cache hit/miss counters."""
        self._policy_cache_hits = 0
        self._policy_cache_misses = 0

    def policy_cache_stats(self) -> dict[str, int | float]:
        """Return candidate-cache performance statistics."""
        total = self._policy_cache_hits + self._policy_cache_misses
        return {
            "hits": self._policy_cache_hits,
            "misses": self._policy_cache_misses,
            "entries": len(self._policy_candidate_cache),
            "hit_rate": (self._policy_cache_hits / total if total else 0.0),
        }

    def greedy_action_candidates(
        self,
        state: State,
    ) -> tuple[JointAction, ...]:
        """
        Return every joint action tied for the minimum current Q-value.

        The candidate set is memoized after planning because V is fixed during
        evaluation. Unlike the previous single-action cache, this preserves
        all greedy choices and allows reproducible stochastic tie breaking on
        repeated visits to the same state.
        """
        self.mdp.validate_state(state)

        cached = self._policy_candidate_cache.get(state)
        if cached is not None:
            self._policy_cache_hits += 1
            return cached

        self._policy_cache_misses += 1

        if self.mdp.is_terminal(state):
            terminal_candidates = (
                tuple("stay" for _ in range(self.mdp.n_agents)),
            )
            self._policy_candidate_cache[state] = terminal_candidates
            return terminal_candidates

        best_value = math.inf
        best_actions: list[JointAction] = []

        # Different action labels can induce exactly the same physical one-step
        # dynamics (blocked moves, explicit stay, or frozen goal agents).  The
        # Baseline policy still considers every labelled joint action so tie
        # behavior is preserved, but Q is computed only once per distinct
        # intended joint state during fixed-policy evaluation.
        intended_by_agent = tuple(
            {
                action: self.mdp.move_one(agent_index, position, action)
                for action in ACTIONS
            }
            for agent_index, position in enumerate(state)
        )
        physical_q_cache: dict[State, float] = {}

        for joint_action in self.mdp.all_joint_actions():
            intended_state = tuple(
                intended_by_agent[agent_index][action]
                for agent_index, action in enumerate(joint_action)
            )
            action_value = physical_q_cache.get(intended_state)
            if action_value is None:
                action_value = self.q_value(
                    state,
                    joint_action,
                    count_metrics=False,
                    deadline=None,
                )
                physical_q_cache[intended_state] = action_value

            if action_value < best_value and not self._values_tied(action_value, best_value):
                best_value = action_value
                best_actions = [joint_action]
            elif self._values_tied(action_value, best_value):
                best_actions.append(joint_action)

        if not best_actions:
            raise RuntimeError("No joint action was generated")

        best_actions = self._guided_joint_actions(
            state,
            best_actions,
        )

        candidates = tuple(best_actions)
        self._policy_candidate_cache[state] = candidates
        return candidates

    def policy_action_with_info(
        self,
        state: State,
        *,
        tie_rng: random.Random | None = None,
    ) -> tuple[JointAction, int]:
        """
        Select a greedy action and report whether a tie had to be broken.

        When ``tie_rng`` is supplied, equal-valued actions are sampled on every
        visit. Without it, a stable hash provides deterministic backward-
        compatible behavior.
        """
        candidates = self.greedy_action_candidates(state)
        tie_decisions = int(len(candidates) > 1)

        if len(candidates) == 1:
            return candidates[0], tie_decisions

        if tie_rng is not None:
            return tie_rng.choice(candidates), tie_decisions

        return (
            self._deterministic_tie_choice(list(candidates), state),
            tie_decisions,
        )

    def policy_action(
        self,
        state: State,
        *,
        tie_rng: random.Random | None = None,
    ) -> JointAction:
        """
        Return a current greedy policy action without updating V or counters.

        Supplying ``tie_rng`` creates a stochastic greedy policy over actions
        that are equal within ``tie_tolerance``. This prevents a fixed cached
        tie choice from trapping evaluation in an arbitrary cycle.
        """
        action, _ = self.policy_action_with_info(
            state,
            tie_rng=tie_rng,
        )
        return action
