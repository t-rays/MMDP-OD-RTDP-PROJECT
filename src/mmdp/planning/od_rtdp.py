from __future__ import annotations

"""OD-RTDP: RTDP over operator-decomposition states.

A stored OD state always has an incomplete action prefix. A complete prefix is
executed immediately in the real environment and is not stored as an OD state.
"""

import math
import random
from typing import Any

from mmdp.planning.components import SolvedTracker, TieBreaker, ValueStore
from mmdp.planning.config import RTDPConfig
from mmdp.planning.domain_base import RTDPDomainBase
from mmdp.domain.grid_mmdp import ACTIONS, Action, GridMMDP, JointAction, State
from mmdp.domain.heuristic import ShortestPathHeuristic

ODState = tuple[State, JointAction]


class OperatorDecompositionDomain(RTDPDomainBase[ODState, Action]):
    """RTDP planning domain over operator-decomposition states."""

    def __init__(
        self,
        mdp: GridMMDP,
        heuristic: ShortestPathHeuristic | None,
        config: RTDPConfig,
        value_store: ValueStore[ODState],
        solved_tracker: SolvedTracker[ODState],
        tie_breaker: TieBreaker[Action],
        joint_tie_breaker: TieBreaker[JointAction],
    ) -> None:
        self.joint_tie_breaker = joint_tie_breaker

        self._raw_operator_candidate_cache: dict[ODState, tuple[Action, ...]] = {}
        self._guided_operator_candidate_cache: dict[ODState, tuple[Action, ...]] = {}
        self._joint_policy_candidate_cache: dict[State, tuple[JointAction, ...]] = {}
        self._global_real_policy_candidate_cache: dict[State, tuple[JointAction, ...]] = {}

        self.planning_operator_evaluations = 0
        self.complete_joint_actions_evaluated = 0

        super().__init__(
            mdp=mdp,
            heuristic=heuristic if heuristic is not None else ShortestPathHeuristic(mdp),
            config=config,
            value_store=value_store,
            solved_tracker=solved_tracker,
            tie_breaker=tie_breaker,
        )

    def _reset_domain_metrics(self) -> None:
        self.planning_operator_evaluations = 0
        self.complete_joint_actions_evaluated = 0

    def reset_caches(self) -> None:
        self._raw_operator_candidate_cache.clear()
        self._guided_operator_candidate_cache.clear()
        self._joint_policy_candidate_cache.clear()
        self._global_real_policy_candidate_cache.clear()
        self.reset_policy_cache_stats()

    def _policy_cache_entries(self) -> int:
        return len(self._joint_policy_candidate_cache)

    # ------------------------------------------------------------------
    # State space
    # ------------------------------------------------------------------

    def initial_state(self) -> ODState:
        return (self.mdp.initial_state(), ())

    def is_terminal(self, od_state: ODState) -> bool:
        state, prefix = od_state
        if self.mdp.is_terminal(state):
            if prefix:
                raise ValueError("A terminal OD state must have an empty prefix")
            return True
        return False

    def validate_od_state(self, od_state: ODState) -> None:
        state, prefix = od_state
        self.mdp.validate_state(state)

        if len(prefix) >= self.mdp.n_agents:
            raise ValueError(
                "A stored OD prefix must be incomplete; complete joint "
                "actions are executed immediately."
            )

        invalid_actions = [action for action in prefix if action not in ACTIONS]
        if invalid_actions:
            raise ValueError(f"OD prefix contains unknown actions: {invalid_actions}")

        if self.mdp.is_terminal(state) and prefix:
            raise ValueError("A terminal real state must use an empty OD prefix.")

    def sample_next(self, od_state: ODState, action: Action) -> ODState:
        state, prefix = od_state
        extended_prefix = prefix + (action,)
        if len(extended_prefix) < self.mdp.n_agents:
            return (state, extended_prefix)
        next_state = self.mdp.sample_next(state, extended_prefix, self.transition_rng)
        return (next_state, ())

    def _completes_real_step(self, od_state: ODState, next_od_state: ODState) -> bool:
        # A real environment step completes when the assembled prefix is
        # executed, which resets the successor's prefix to empty.
        return not next_od_state[1]

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def get_value(self, od_state: ODState) -> float:
        self.validate_od_state(od_state)
        state, prefix = od_state

        if self.mdp.is_terminal(state):
            return 0.0

        stored = self.value_store.get(od_state)
        if stored is not None:
            return stored

        return self.heuristic.od_value(state, prefix)

    def real_state_value(self, state: State) -> float:
        return self.get_value((state, ()))

    def complete_joint_action_value(
        self,
        state: State,
        joint_action: JointAction,
        *,
        count_metrics: bool = True,
        deadline: float | None = None,
    ) -> float:
        self._check_deadline(deadline)
        self.mdp.validate_state(state)
        self.mdp.validate_joint_action(joint_action)

        transitions = self.mdp.joint_transitions(state, joint_action)
        self._check_deadline(deadline)

        if count_metrics:
            self.complete_joint_actions_evaluated += 1
            self.transition_outcomes_evaluated += len(transitions)

        expected_cost = 0.0

        for next_state, probability in transitions:
            self._check_deadline(deadline)
            immediate_cost = self.mdp.transition_cost(state, joint_action, next_state)
            future_cost = self.real_state_value(next_state)
            expected_cost += probability * (immediate_cost + future_cost)

        return expected_cost

    def operator_value(
        self,
        od_state: ODState,
        action: Action,
        *,
        count_metrics: bool = True,
        deadline: float | None = None,
    ) -> float:
        self._check_deadline(deadline)
        self.validate_od_state(od_state)

        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action!r}")

        if count_metrics:
            self.planning_operator_evaluations += 1

        state, prefix = od_state
        extended_prefix = prefix + (action,)

        if len(extended_prefix) < self.mdp.n_agents:
            return self.get_value((state, extended_prefix))

        return self.complete_joint_action_value(
            state,
            extended_prefix,
            count_metrics=count_metrics,
            deadline=deadline,
        )

    # ------------------------------------------------------------------
    # Greedy operator selection
    # ------------------------------------------------------------------

    def _guided_operators(
        self,
        od_state: ODState,
        candidates: list[Action],
    ) -> list[Action]:
        if len(candidates) <= 1:
            return candidates

        guidance_method = getattr(self.heuristic, "od_operator_guidance_key", None)
        if not callable(guidance_method):
            return candidates

        state, prefix = od_state
        keyed_candidates = [
            (guidance_method(state, prefix, action), action)
            for action in candidates
        ]
        best_key = min(key for key, _ in keyed_candidates)

        return [
            action
            for key, action in keyed_candidates
            if key == best_key
        ]

    def best_action(
        self,
        od_state: ODState,
        *,
        count_metrics: bool = True,
        random_ties: bool = True,
        deadline: float | None = None,
    ) -> tuple[Action, float]:
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

            if candidate_value < best_value and not self._values_tied(candidate_value, best_value):
                best_value = candidate_value
                best_actions = [action]
            elif self._values_tied(candidate_value, best_value):
                best_actions.append(action)

        if not best_actions:
            raise RuntimeError("No OD operator was generated")

        best_actions = self._guided_operators(od_state, best_actions)

        if random_ties:
            selected_action = self.tie_rng.choice(best_actions)
        else:
            selected_action = self.tie_breaker.choose(best_actions, od_state)

        return selected_action, best_value

    # ------------------------------------------------------------------
    # LRTDP solved check
    # ------------------------------------------------------------------

    def get_successors_for_solved_check(
        self, od_state: ODState, action: Action
    ) -> tuple[ODState, ...]:
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

    # ------------------------------------------------------------------
    # Planning metrics
    # ------------------------------------------------------------------

    def build_result_kwargs(self) -> dict[str, Any]:
        return {
            "planning_operator_evaluations": self.planning_operator_evaluations,
            "complete_joint_actions_evaluated": self.complete_joint_actions_evaluated,
            "visited_od_states": len(self.value_store),
            "visited_real_states": len(
                {state for state, _ in self.value_store.states()}
            ),
            "solved_od_states": len(self.solved_tracker),
            "solved_real_states": len(
                {state for state, prefix in self.solved_tracker.states() if not prefix}
            ),
        }

    # ------------------------------------------------------------------
    # Fixed-policy extraction (used by evaluation)
    # ------------------------------------------------------------------

    def _bellman_operator_candidates(self, od_state: ODState) -> tuple[Action, ...]:
        self.validate_od_state(od_state)
        cached = self._raw_operator_candidate_cache.get(od_state)
        if cached is not None:
            return cached

        state, prefix = od_state
        if self.mdp.is_terminal(state):
            terminal_candidates = ("stay",)
            self._raw_operator_candidate_cache[od_state] = terminal_candidates
            return terminal_candidates

        current_agent = len(prefix)
        if (
            self.mdp.config.freeze_agents_at_goal
            and state[current_agent] == self.mdp.goals[current_agent]
        ):
            frozen_candidates = ("stay",)
            self._raw_operator_candidate_cache[od_state] = frozen_candidates
            return frozen_candidates

        best_value = math.inf
        best_actions: list[Action] = []

        for action in ACTIONS:
            candidate_value = self.operator_value(
                od_state, action, count_metrics=False, deadline=None
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

    def greedy_operator_candidates(self, od_state: ODState) -> tuple[Action, ...]:
        self.validate_od_state(od_state)
        cached = self._guided_operator_candidate_cache.get(od_state)
        if cached is not None:
            return cached

        best_actions = self._guided_operators(
            od_state, list(self._bellman_operator_candidates(od_state))
        )

        candidates = tuple(best_actions)
        self._guided_operator_candidate_cache[od_state] = candidates
        return candidates

    def _guided_complete_joint_actions(
        self, state: State, candidates: list[JointAction]
    ) -> list[JointAction]:
        if len(candidates) <= 1:
            return candidates

        guidance_method = getattr(self.heuristic, "joint_action_guidance_key", None)
        if not callable(guidance_method):
            return candidates

        keyed_candidates = [
            (guidance_method(state, action), action)
            for action in candidates
        ]
        best_key = min(key for key, _ in keyed_candidates)
        return [action for key, action in keyed_candidates if key == best_key]

    def greedy_joint_action_candidates(self, state: State) -> tuple[JointAction, ...]:
        self.mdp.validate_state(state)
        cached = self._joint_policy_candidate_cache.get(state)
        if cached is not None:
            self._policy_cache_hits += 1
            return cached

        self._policy_cache_misses += 1

        if self.mdp.is_terminal(state):
            terminal_candidates = (tuple("stay" for _ in range(self.mdp.n_agents)),)
            self._joint_policy_candidate_cache[state] = terminal_candidates
            return terminal_candidates

        def expand(*, locally_guided: bool) -> list[JointAction]:
            partial_prefixes: list[JointAction] = [()]
            while partial_prefixes and len(partial_prefixes[0]) < self.mdp.n_agents:
                expanded: list[JointAction] = []
                for prefix in partial_prefixes:
                    od_state = (state, prefix)
                    local_candidates = (
                        self.greedy_operator_candidates(od_state)
                        if locally_guided
                        else self._bellman_operator_candidates(od_state)
                    )
                    expanded.extend(prefix + (action,) for action in local_candidates)
                partial_prefixes = expanded
            return partial_prefixes

        partial_prefixes = expand(locally_guided=False)

        if not partial_prefixes:
            raise RuntimeError("No complete OD joint action was generated")

        guided = self._guided_complete_joint_actions(state, partial_prefixes)
        candidates = tuple(guided)
        self._joint_policy_candidate_cache[state] = candidates
        return candidates

    def global_real_greedy_action_candidates(self, state: State) -> tuple[JointAction, ...]:
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
        self, state: State, *, tie_rng: random.Random | None = None
    ) -> tuple[JointAction, int]:
        candidates = self.global_real_greedy_action_candidates(state)
        if len(candidates) == 1:
            return candidates[0], 0
        if tie_rng is not None:
            return tie_rng.choice(candidates), 1
        return self.joint_tie_breaker.choose(list(candidates), state), 1

    def global_policy_action(
        self, state: State, *, tie_rng: random.Random | None = None
    ) -> JointAction:
        return self.global_policy_action_with_info(state, tie_rng=tie_rng)[0]

    def policy_action_with_info(
        self, state: State, *, tie_rng: random.Random | None = None
    ) -> tuple[JointAction, int]:
        self.mdp.validate_state(state)
        if self.mdp.is_terminal(state):
            return (tuple("stay" for _ in range(self.mdp.n_agents)), 0)

        candidates = self.greedy_joint_action_candidates(state)

        if len(candidates) == 1:
            selected = candidates[0]
        elif tie_rng is not None:
            selected = tie_rng.choice(candidates)
        else:
            selected = self.joint_tie_breaker.choose(list(candidates), state)

        tie_decisions = 0
        for prefix_length in range(self.mdp.n_agents):
            prefix = selected[:prefix_length]
            if len(self._bellman_operator_candidates((state, prefix))) > 1:
                tie_decisions += 1

        return selected, tie_decisions

    def policy_action(
        self, state: State, *, tie_rng: random.Random | None = None
    ) -> JointAction:
        action, _ = self.policy_action_with_info(state, tie_rng=tie_rng)
        return action
