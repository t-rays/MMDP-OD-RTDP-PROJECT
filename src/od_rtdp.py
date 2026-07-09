from __future__ import annotations

"""
Operator-Decomposition RTDP for the stochastic cooperative grid MMDP.

The planner solves the same cost-minimization Stochastic Shortest Path problem
as Baseline RTDP:

    V(s) = min_a sum_s' P(s' | s, a)
           [c(s, a, s') + V(s')]

The difference is how a complete joint action is selected.

Baseline RTDP evaluates every complete joint action at once. With five actions
per agent, one Bellman backup has branching factor 5 ** number_of_agents.

OD-RTDP represents an intermediate planning state as:

    (real_state, action_prefix)

and chooses one agent action at a time. Every intermediate OD state has only
five outgoing operators. No real time, cost, stochastic transition, or slip is
applied while the prefix is being built. Only after the prefix contains one
action for every agent is the complete joint action executed in the real MMDP.
"""

from dataclasses import dataclass
import hashlib
import math
import random
import time
from typing import Iterator

from baseline_rtdp import RTDPConfig
from grid_mmdp import ACTIONS, Action, GridMMDP, JointAction, State
from heuristic import ShortestPathHeuristic
from limits import sequential_multi_agent_step_bound
from numerics import scaled_residual_ratio, tied_by_ulp
from resource_monitor import ResourceMonitor


# A stored OD state always has an incomplete action prefix. A complete prefix is
# executed immediately in the real environment and is not stored as an OD state.
ODState = tuple[State, JointAction]


class _DeadlineReached(RuntimeError):
    """Internal signal used to stop planning when the deadline is reached."""


class _MemoryLimitReached(RuntimeError):
    """Internal signal used to stop planning at the RSS-delta limit."""


@dataclass(frozen=True)
class ODTrialResult:
    maximum_residual: float
    maximum_scaled_residual: float
    real_steps: int
    reached_goal: bool
    reached_solved_state: bool
    hit_step_limit: bool
    visited_od_states: tuple[ODState, ...]


@dataclass(frozen=True)
class ODRTDPPlanningResult:
    stop_reason: str
    trials_completed: int
    goal_reaching_trials: int
    step_limited_trials: int
    total_real_steps: int
    elapsed_seconds: float
    bellman_backups: int
    planning_operator_evaluations: int
    complete_joint_actions_evaluated: int
    transition_outcomes_evaluated: int
    visited_od_states: int
    visited_real_states: int
    final_trial_residual: float
    final_trial_scaled_residual: float
    consecutive_stable_trials: int
    maximum_consecutive_stable_trials: int
    stability_criterion_reached: bool
    first_stability_trial: int | None
    first_stability_elapsed_seconds: float | None
    initial_state_solved: bool
    solved_od_states: int
    solved_real_states: int
    solved_checks: int
    first_solved_trial: int | None
    first_solved_elapsed_seconds: float | None
    resolved_max_steps_per_trial: int | None
    memory_limit_mb: float | None
    memory_limit_reached: bool
    baseline_rss_mb: float
    peak_rss_mb: float
    peak_rss_delta_mb: float


class OperatorDecompositionRTDP:
    """
    RTDP over operator-decomposition states.

    For an incomplete prefix alpha, the Bellman equation is:

        V_OD(s, alpha) = min_a_i V_OD(s, alpha + a_i)

    because choosing another component of the joint action does not yet consume
    a real environment step.

    When alpha + a_i is a complete joint action, the real MMDP equation is used:

        Q_OD(s, alpha, a_i)
            = sum_s' P(s' | s, alpha + a_i)
              [c(s, alpha + a_i, s') + V_OD(s', empty_prefix)]

    Thus Baseline RTDP and OD-RTDP optimize exactly the same objective and use
    exactly the same environment model. Only the action-selection structure is
    different.
    """

    def __init__(
        self,
        mdp: GridMMDP,
        heuristic: ShortestPathHeuristic | None = None,
        config: RTDPConfig | None = None,
    ) -> None:
        self.mdp = mdp
        self.heuristic = (
            heuristic
            if heuristic is not None
            else ShortestPathHeuristic(mdp)
        )
        self.config = config or RTDPConfig()

        self.resolved_max_steps_per_trial = (
            self._resolve_max_steps_per_trial()
        )

        # Values are stored for real states paired with incomplete prefixes.
        self.V: dict[ODState, float] = {}

        # LRTDP solved labels on the expanded OD state space. Terminal real
        # states with an empty prefix are treated as solved implicitly.
        self._solved_od_states: set[ODState] = set()
        self.solved_checks = 0

        # Evaluation uses three immutable caches after planning:
        #
        # 1. Bellman-tied local operators before secondary guidance.
        # 2. Locally guided operators for the public diagnostic method.
        # 3. Globally guided complete joint actions for the actual policy.
        #
        # The third cache avoids rebuilding a complete action one prefix at a
        # time on every visit.  More importantly, complete-action guidance is
        # applied only after all Bellman-tied prefixes are expanded, so an
        # early local tie cannot accidentally force a high-self-loop joint
        # action.
        self._raw_operator_candidate_cache: dict[
            ODState,
            tuple[Action, ...],
        ] = {}
        self._guided_operator_candidate_cache: dict[
            ODState,
            tuple[Action, ...],
        ] = {}
        self._joint_policy_candidate_cache: dict[
            State,
            tuple[JointAction, ...],
        ] = {}
        self._global_real_policy_candidate_cache: dict[
            State,
            tuple[JointAction, ...],
        ] = {}
        self._policy_cache_hits = 0
        self._policy_cache_misses = 0

        self.bellman_backups = 0
        self.planning_operator_evaluations = 0
        self.complete_joint_actions_evaluated = 0
        self.transition_outcomes_evaluated = 0

        # Transition sampling is deliberately independent from tie breaking.
        self.transition_rng = random.Random(self.config.seed)
        self.tie_rng = random.Random(self.config.seed + 1)
        self._resource_monitor: ResourceMonitor | None = None

    def _resolve_max_steps_per_trial(self) -> int | None:
        if self.config.max_steps_per_trial is not None:
            return self.config.max_steps_per_trial
        distances = self.heuristic.distance_summary(self.mdp.initial_state())
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
        """Remove the current solution and reset counters and random states."""
        self.V.clear()
        self._solved_od_states.clear()
        self.solved_checks = 0
        self._raw_operator_candidate_cache.clear()
        self._guided_operator_candidate_cache.clear()
        self._joint_policy_candidate_cache.clear()
        self._global_real_policy_candidate_cache.clear()
        self.reset_policy_cache_stats()

        self.bellman_backups = 0
        self.planning_operator_evaluations = 0
        self.complete_joint_actions_evaluated = 0
        self.transition_outcomes_evaluated = 0

        self.transition_rng.seed(self.config.seed)
        self.tie_rng.seed(self.config.seed + 1)

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

    def validate_od_state(self, od_state: ODState) -> None:
        """Validate a real state paired with an incomplete action prefix."""
        state, prefix = od_state
        self.mdp.validate_state(state)

        if len(prefix) >= self.mdp.n_agents:
            raise ValueError(
                "A stored OD prefix must be incomplete; complete joint "
                "actions are executed immediately."
            )

        invalid_actions = [
            action for action in prefix if action not in ACTIONS
        ]
        if invalid_actions:
            raise ValueError(
                f"OD prefix contains unknown actions: {invalid_actions}"
            )

        if self.mdp.is_terminal(state) and prefix:
            raise ValueError(
                "A terminal real state must use an empty OD prefix."
            )

    def value(self, od_state: ODState) -> float:
        """
        Return the current value estimate of an OD state.

        Terminal real state:
            0

        Previously updated OD state:
            Stored Bellman value

        New OD state:
            Operator-decomposition shortest-path heuristic
        """
        self.validate_od_state(od_state)
        state, prefix = od_state

        if self.mdp.is_terminal(state):
            return 0.0

        if od_state in self.V:
            return self.V[od_state]

        return self.heuristic.od_value(state, prefix)

    def real_state_value(self, state: State) -> float:
        """Return the value of a real state with an empty action prefix."""
        return self.value((state, ()))

    def complete_joint_action_value(
        self,
        state: State,
        joint_action: JointAction,
        *,
        count_metrics: bool = True,
        deadline: float | None = None,
    ) -> float:
        """
        Evaluate a complete joint action using the real stochastic MMDP.

        Q(s,a) = sum_s' P(s'|s,a) [c(s,a,s') + V_OD(s', empty)]
        """
        self._check_deadline(deadline)
        self.mdp.validate_state(state)
        self.mdp.validate_joint_action(joint_action)

        transitions = self.mdp.joint_transitions(
            state,
            joint_action,
        )

        self._check_deadline(deadline)

        if count_metrics:
            self.complete_joint_actions_evaluated += 1
            self.transition_outcomes_evaluated += len(transitions)

        expected_cost = 0.0

        for next_state, probability in transitions:
            self._check_deadline(deadline)

            immediate_cost = self.mdp.transition_cost(
                state,
                joint_action,
                next_state,
            )
            future_cost = self.real_state_value(next_state)

            expected_cost += probability * (
                immediate_cost + future_cost
            )

        return expected_cost

    def operator_value(
        self,
        od_state: ODState,
        action: Action,
        *,
        count_metrics: bool = True,
        deadline: float | None = None,
    ) -> float:
        """
        Evaluate one local OD operator for the next agent.

        For an incomplete resulting prefix, no real transition occurs and the
        value is simply the value of the child OD state.

        For a complete resulting prefix, execute the complete joint action in
        the real stochastic MMDP and include the real transition cost.
        """
        self._check_deadline(deadline)
        self.validate_od_state(od_state)

        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action!r}")

        if count_metrics:
            self.planning_operator_evaluations += 1

        state, prefix = od_state
        extended_prefix = prefix + (action,)

        if len(extended_prefix) < self.mdp.n_agents:
            # Pure planning transition: no time, cost, slip, or collision.
            return self.value((state, extended_prefix))

        # The prefix is now a complete joint action and is executed for real.
        return self.complete_joint_action_value(
            state,
            extended_prefix,
            count_metrics=count_metrics,
            deadline=deadline,
        )

    def _deterministic_tie_choice(
        self,
        candidates: list[Action],
        od_state: ODState,
    ) -> Action:
        """
        Choose reproducibly among equal-valued OD operators.

        The real state, current prefix, planning seed, and tied actions define
        the choice. This prevents a fixed preference for the first action in
        ACTIONS while keeping policy evaluation deterministic.
        """
        if not candidates:
            raise ValueError(
                "candidates cannot be empty"
            )

        payload = repr(
            (
                self.config.seed,
                od_state,
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

    def _guided_operators(
        self,
        od_state: ODState,
        candidates: list[Action],
    ) -> list[Action]:
        """
        Refine Bellman-tied OD operators using structured heuristic guidance.

        The operator value remains the primary criterion. Guidance is applied
        only within the tie_tolerance set, preserving the OD objective while
        distinguishing equally short branches by conflict risk, path diversity,
        future progress options, and local mobility.
        """
        if len(candidates) <= 1:
            return candidates

        guidance_method = getattr(
            self.heuristic,
            "od_operator_guidance_key",
            None,
        )

        if not callable(guidance_method):
            return candidates

        state, prefix = od_state
        keyed_candidates = [
            (guidance_method(state, prefix, action), action)
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

    def best_operator(
        self,
        od_state: ODState,
        *,
        count_metrics: bool = True,
        random_ties: bool = True,
        deadline: float | None = None,
    ) -> tuple[Action, float]:
        """Return the minimum-cost action for the next agent in the prefix."""
        self.validate_od_state(od_state)
        state, _ = od_state

        if self.mdp.is_terminal(state):
            return "stay", 0.0

        best_value = math.inf
        best_actions: list[Action] = []

        for action in ACTIONS:
            self._check_deadline(deadline)

            candidate_value = self.operator_value(
                od_state,
                action,
                count_metrics=count_metrics,
                deadline=deadline,
            )

            if candidate_value < best_value and not self._values_tied(
                candidate_value, best_value
            ):
                best_value = candidate_value
                best_actions = [action]

            elif self._values_tied(candidate_value, best_value):
                best_actions.append(action)

        if not best_actions:
            raise RuntimeError("No OD operator was generated")

        best_actions = self._guided_operators(
            od_state,
            best_actions,
        )

        if random_ties:
            selected_action = self.tie_rng.choice(
                best_actions
            )
        else:
            selected_action = self._deterministic_tie_choice(
                best_actions,
                od_state,
            )

        return selected_action, best_value

    def backup(
        self,
        od_state: ODState,
        *,
        deadline: float | None = None,
    ) -> tuple[Action, float, float]:
        old_value = self.value(od_state)
        selected_action, new_value = self.best_operator(
            od_state, count_metrics=True, random_ties=True, deadline=deadline
        )
        self.V[od_state] = new_value
        self.bellman_backups += 1
        residual = abs(new_value - old_value)
        scaled = scaled_residual_ratio(
            old_value,
            new_value,
            absolute_tolerance=self.config.epsilon,
            relative_tolerance=self.config.relative_epsilon,
        )
        return selected_action, residual, scaled

    def _trial_step_numbers(self) -> Iterator[int]:
        """Generate bounded or unbounded real environment step indices."""
        step_number = 0

        while (
            self.resolved_max_steps_per_trial is None
            or step_number < self.resolved_max_steps_per_trial
        ):
            yield step_number
            step_number += 1

    def run_trial(
        self,
        *,
        deadline: float | None = None,
    ) -> ODTrialResult:
        state = self.mdp.initial_state()
        maximum_residual = 0.0
        maximum_scaled_residual = 0.0
        real_steps = 0
        visited_od_states: list[ODState] = []
        if self.mdp.is_terminal(state):
            return ODTrialResult(0.0, 0.0, 0, True, False, False, ())

        for _ in self._trial_step_numbers():
            self._check_deadline(deadline)
            prefix: JointAction = ()
            while len(prefix) < self.mdp.n_agents:
                od_state = (state, prefix)
                if (
                    self.config.stop_when_solved
                    and od_state in self._solved_od_states
                ):
                    return ODTrialResult(
                        maximum_residual,
                        maximum_scaled_residual,
                        real_steps,
                        False,
                        True,
                        False,
                        tuple(visited_od_states),
                    )

                visited_od_states.append(od_state)
                selected_action, residual, scaled = self.backup(
                    od_state, deadline=deadline
                )
                maximum_residual = max(maximum_residual, residual)
                maximum_scaled_residual = max(
                    maximum_scaled_residual, scaled
                )
                prefix = prefix + (selected_action,)

            state = self.mdp.sample_next(state, prefix, self.transition_rng)
            real_steps += 1
            if self.mdp.is_terminal(state):
                return ODTrialResult(
                    maximum_residual,
                    maximum_scaled_residual,
                    real_steps,
                    True,
                    False,
                    False,
                    tuple(visited_od_states),
                )
        return ODTrialResult(
            maximum_residual,
            maximum_scaled_residual,
            real_steps,
            False,
            False,
            True,
            tuple(visited_od_states),
        )

    def _operator_successors_for_solved_check(
        self,
        od_state: ODState,
        action: Action,
    ) -> tuple[ODState, ...]:
        """Return every positive-probability OD successor of one operator."""
        state, prefix = od_state
        extended_prefix = prefix + (action,)
        if len(extended_prefix) < self.mdp.n_agents:
            return ((state, extended_prefix),)
        return tuple(
            (next_state, ())
            for next_state, probability in self.mdp.joint_transitions(
                state, extended_prefix
            )
            if probability > 0.0
        )

    def check_solved(
        self,
        root: ODState,
        *,
        deadline: float | None = None,
    ) -> bool:
        """LRTDP-style solved-state check on the expanded OD state space."""
        self.solved_checks += 1
        root_state, root_prefix = root
        if self.mdp.is_terminal(root_state):
            if root_prefix:
                raise ValueError("A terminal OD state must have an empty prefix")
            return True
        if root in self._solved_od_states:
            return True

        open_stack: list[ODState] = [root]
        open_set: set[ODState] = {root}
        closed: list[ODState] = []
        closed_set: set[ODState] = set()
        envelope_is_solved = True

        while open_stack:
            self._check_deadline(deadline)
            od_state = open_stack.pop()
            open_set.discard(od_state)
            state, prefix = od_state
            if self.mdp.is_terminal(state):
                if prefix:
                    raise ValueError(
                        "A terminal OD state must have an empty prefix"
                    )
                continue
            if od_state in self._solved_od_states or od_state in closed_set:
                continue

            closed.append(od_state)
            closed_set.add(od_state)

            old_value = self.value(od_state)
            action, bellman_value = self.best_operator(
                od_state,
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

            for successor in self._operator_successors_for_solved_check(
                od_state, action
            ):
                successor_state, successor_prefix = successor
                if self.mdp.is_terminal(successor_state):
                    if successor_prefix:
                        raise ValueError(
                            "A terminal OD state must have an empty prefix"
                        )
                    continue
                if (
                    successor in self._solved_od_states
                    or successor in closed_set
                    or successor in open_set
                ):
                    continue
                open_stack.append(successor)
                open_set.add(successor)

        if envelope_is_solved:
            self._solved_od_states.update(closed_set)
            return True

        for od_state in reversed(closed):
            self._check_deadline(deadline)
            if od_state not in self._solved_od_states:
                self.backup(od_state, deadline=deadline)
        return False

    def _label_trial_path(
        self,
        visited_od_states: tuple[ODState, ...],
        *,
        deadline: float | None = None,
    ) -> None:
        """Try to label OD states on a sampled path, from end to start."""
        for od_state in reversed(visited_od_states):
            self._check_deadline(deadline)
            if not self.check_solved(od_state, deadline=deadline):
                break

    def _trial_numbers(self) -> Iterator[int]:
        """Generate bounded or unbounded internal trial numbers."""
        trial_number = 1

        while (
            self.config.max_trials is None
            or trial_number <= self.config.max_trials
        ):
            yield trial_number
            trial_number += 1

    def solve(self, *, reset: bool = True) -> ODRTDPPlanningResult:
        if reset:
            self.reset()
        self._raw_operator_candidate_cache.clear()
        self._guided_operator_candidate_cache.clear()
        self._joint_policy_candidate_cache.clear()
        self._global_real_policy_candidate_cache.clear()
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
        total_real_steps = 0
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
                total_real_steps += trial_result.real_steps
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
                            trial_result.visited_od_states,
                            deadline=deadline,
                        )
                    except _DeadlineReached:
                        stop_reason = "time_limit"
                        break
                    except _MemoryLimitReached:
                        stop_reason = "memory_limit"
                        break

                    initial_od_state: ODState = (
                        self.mdp.initial_state(),
                        (),
                    )
                    if (
                        self.mdp.is_terminal(initial_od_state[0])
                        or initial_od_state in self._solved_od_states
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

        visited_real_states = len({state for state, _ in self.V})
        return ODRTDPPlanningResult(
            stop_reason=stop_reason,
            trials_completed=trials_completed,
            goal_reaching_trials=goal_reaching_trials,
            step_limited_trials=step_limited_trials,
            total_real_steps=total_real_steps,
            elapsed_seconds=time.perf_counter() - started_at,
            bellman_backups=self.bellman_backups,
            planning_operator_evaluations=self.planning_operator_evaluations,
            complete_joint_actions_evaluated=(
                self.complete_joint_actions_evaluated
            ),
            transition_outcomes_evaluated=self.transition_outcomes_evaluated,
            visited_od_states=len(self.V),
            visited_real_states=visited_real_states,
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
                or (self.mdp.initial_state(), ()) in self._solved_od_states
            ),
            solved_od_states=len(self._solved_od_states),
            solved_real_states=len({
                state
                for state, prefix in self._solved_od_states
                if not prefix
            }),
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
        """Reset OD policy-candidate cache hit/miss counters."""
        self._policy_cache_hits = 0
        self._policy_cache_misses = 0

    def policy_cache_stats(self) -> dict[str, int | float]:
        """Return complete-policy candidate-cache statistics."""
        total = self._policy_cache_hits + self._policy_cache_misses
        return {
            "hits": self._policy_cache_hits,
            "misses": self._policy_cache_misses,
            "entries": len(self._joint_policy_candidate_cache),
            "hit_rate": (self._policy_cache_hits / total if total else 0.0),
        }

    def _bellman_operator_candidates(
        self,
        od_state: ODState,
    ) -> tuple[Action, ...]:
        """Return every locally Bellman-tied operator before guidance.

        These candidates preserve the primary OD value ordering.  Secondary
        guidance is deliberately postponed when a complete evaluation action
        is built, allowing the final joint action to be compared as a whole.
        """
        self.validate_od_state(od_state)

        cached = self._raw_operator_candidate_cache.get(od_state)
        if cached is not None:
            return cached

        state, prefix = od_state
        if self.mdp.is_terminal(state):
            terminal_candidates = ("stay",)
            self._raw_operator_candidate_cache[od_state] = (
                terminal_candidates
            )
            return terminal_candidates

        current_agent = len(prefix)
        if (
            self.mdp.config.freeze_agents_at_goal
            and state[current_agent] == self.mdp.goals[current_agent]
        ):
            # Every label has exactly the same physical effect for a frozen
            # agent. Canonicalizing it to stay removes 5-way artificial ties.
            frozen_candidates = ("stay",)
            self._raw_operator_candidate_cache[od_state] = frozen_candidates
            return frozen_candidates

        best_value = math.inf
        best_actions: list[Action] = []

        for action in ACTIONS:
            candidate_value = self.operator_value(
                od_state,
                action,
                count_metrics=False,
                deadline=None,
            )

            if candidate_value < best_value and not self._values_tied(candidate_value, best_value):
                best_value = candidate_value
                best_actions = [action]
            elif self._values_tied(candidate_value, best_value):
                best_actions.append(action)

        if not best_actions:
            raise RuntimeError("No OD operator was generated")

        candidates = tuple(best_actions)
        self._raw_operator_candidate_cache[od_state] = candidates
        return candidates

    def greedy_operator_candidates(
        self,
        od_state: ODState,
    ) -> tuple[Action, ...]:
        """
        Return all locally greedy operators for one incomplete OD state.

        Candidate sets are memoized because V is fixed during evaluation. The
        selected operator itself is not cached, allowing stochastic tie
        breaking whenever the same real state and prefix are revisited.
        """
        self.validate_od_state(od_state)

        cached = self._guided_operator_candidate_cache.get(od_state)
        if cached is not None:
            return cached

        best_actions = self._guided_operators(
            od_state,
            list(self._bellman_operator_candidates(od_state)),
        )

        candidates = tuple(best_actions)
        self._guided_operator_candidate_cache[od_state] = candidates
        return candidates

    def _guided_complete_joint_actions(
        self,
        state: State,
        candidates: list[JointAction],
    ) -> list[JointAction]:
        """Refine complete OD actions using one global secondary key."""
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
        best_key = min(key for key, _ in keyed_candidates)
        return [
            action
            for key, action in keyed_candidates
            if key == best_key
        ]

    def greedy_joint_action_candidates(
        self,
        state: State,
    ) -> tuple[JointAction, ...]:
        """Return globally guided complete actions induced by OD ties.

        The search expands every Bellman-tied local branch and compares the
        resulting complete joint actions with one global secondary key.
        """
        self.mdp.validate_state(state)

        cached = self._joint_policy_candidate_cache.get(state)
        if cached is not None:
            self._policy_cache_hits += 1
            return cached

        self._policy_cache_misses += 1

        if self.mdp.is_terminal(state):
            terminal_candidates = (
                tuple("stay" for _ in range(self.mdp.n_agents)),
            )
            self._joint_policy_candidate_cache[state] = terminal_candidates
            return terminal_candidates

        def expand(*, locally_guided: bool) -> list[JointAction]:
            partial_prefixes: list[JointAction] = [()]

            while (
                partial_prefixes
                and len(partial_prefixes[0]) < self.mdp.n_agents
            ):
                expanded: list[JointAction] = []
                for prefix in partial_prefixes:
                    od_state = (state, prefix)
                    local_candidates = (
                        self.greedy_operator_candidates(od_state)
                        if locally_guided
                        else self._bellman_operator_candidates(od_state)
                    )
                    expanded.extend(
                        prefix + (action,)
                        for action in local_candidates
                    )

                partial_prefixes = expanded

            return partial_prefixes

        partial_prefixes = expand(locally_guided=False)

        if not partial_prefixes:
            raise RuntimeError("No complete OD joint action was generated")

        guided = self._guided_complete_joint_actions(
            state,
            partial_prefixes,
        )
        candidates = tuple(guided)
        self._joint_policy_candidate_cache[state] = candidates
        return candidates

    def _deterministic_joint_tie_choice(
        self,
        candidates: list[JointAction],
        state: State,
    ) -> JointAction:
        """Choose reproducibly among complete globally guided actions."""
        if not candidates:
            raise ValueError("candidates cannot be empty")

        payload = repr(
            (
                self.config.seed,
                state,
                tuple(candidates),
                "complete-od-policy",
            )
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        index = int.from_bytes(digest[:8], byteorder="big") % len(candidates)
        return candidates[index]


    def global_real_greedy_action_candidates(
        self,
        state: State,
    ) -> tuple[JointAction, ...]:
        """Diagnostic policy extracted directly from real-state OD values.

        This enumerates all complete joint actions and evaluates them with
        ``V_OD(next_state, empty_prefix)``.  Comparing it with the ordinary
        prefix-induced policy separates value-learning errors from policy-
        extraction errors.
        """
        self.mdp.validate_state(state)
        cached = self._global_real_policy_candidate_cache.get(state)
        if cached is not None:
            return cached
        if self.mdp.is_terminal(state):
            result = (tuple("stay" for _ in range(self.mdp.n_agents)),)
            self._global_real_policy_candidate_cache[state] = result
            return result

        best_value = math.inf
        best_actions: list[JointAction] = []
        for joint_action in self.mdp.all_joint_actions():
            value = self.complete_joint_action_value(
                state, joint_action, count_metrics=False, deadline=None
            )
            if value < best_value and not self._values_tied(value, best_value):
                best_value = value
                best_actions = [joint_action]
            elif self._values_tied(value, best_value):
                best_actions.append(joint_action)
        guided = self._guided_complete_joint_actions(state, best_actions)
        result = tuple(guided)
        self._global_real_policy_candidate_cache[state] = result
        return result

    def global_policy_action_with_info(
        self,
        state: State,
        *,
        tie_rng: random.Random | None = None,
    ) -> tuple[JointAction, int]:
        candidates = self.global_real_greedy_action_candidates(state)
        if len(candidates) == 1:
            return candidates[0], 0
        if tie_rng is not None:
            return tie_rng.choice(candidates), 1
        return self._deterministic_joint_tie_choice(
            list(candidates), state
        ), 1

    def global_policy_action(
        self,
        state: State,
        *,
        tie_rng: random.Random | None = None,
    ) -> JointAction:
        return self.global_policy_action_with_info(
            state, tie_rng=tie_rng
        )[0]

    def policy_action_with_info(
        self,
        state: State,
        *,
        tie_rng: random.Random | None = None,
    ) -> tuple[JointAction, int]:
        """
        Select one globally guided complete OD action.

        The returned tie count retains the old interpretation: it counts the
        number of prefixes on the selected path that had multiple Bellman-tied
        local operators before secondary guidance.
        """
        self.mdp.validate_state(state)

        if self.mdp.is_terminal(state):
            return (
                tuple("stay" for _ in range(self.mdp.n_agents)),
                0,
            )

        candidates = self.greedy_joint_action_candidates(state)

        if len(candidates) == 1:
            selected = candidates[0]
        elif tie_rng is not None:
            selected = tie_rng.choice(candidates)
        else:
            selected = self._deterministic_joint_tie_choice(
                list(candidates),
                state,
            )

        tie_decisions = 0
        for prefix_length in range(self.mdp.n_agents):
            prefix = selected[:prefix_length]
            if len(
                self._bellman_operator_candidates((state, prefix))
            ) > 1:
                tie_decisions += 1

        return selected, tie_decisions

    def policy_action(
        self,
        state: State,
        *,
        tie_rng: random.Random | None = None,
    ) -> JointAction:
        """
        Construct a globally guided current greedy joint action without backups.

        Supplying ``tie_rng`` samples among complete actions that remain tied
        after global guidance. Candidate sets are cached per real state.
        """
        action, _ = self.policy_action_with_info(
            state,
            tie_rng=tie_rng,
        )
        return action
