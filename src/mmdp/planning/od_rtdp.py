from __future__ import annotations

"""OD-RTDP over states that pair a real state with an action prefix."""

import math

from mmdp.domain.grid_mmdp import ACTIONS, Action, GridMMDP, JointAction, State
from mmdp.domain.heuristic import ShortestPathHeuristic
from mmdp.planning.components import SolvedTracker, TieBreaker, ValueStore
from mmdp.planning.config import RTDPConfig
from mmdp.planning.domain_base import RTDPDomainBase

ODState = tuple[State, JointAction]


class OperatorDecompositionDomain(RTDPDomainBase[ODState, Action]):
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
        super().__init__(
            mdp=mdp,
            heuristic=heuristic or ShortestPathHeuristic(mdp),
            config=config,
            value_store=value_store,
            solved_tracker=solved_tracker,
            tie_breaker=tie_breaker,
        )

    def reset_caches(self) -> None:
        self._raw_operator_candidate_cache.clear()
        self._guided_operator_candidate_cache.clear()
        self._joint_policy_candidate_cache.clear()

    def initial_state(self) -> ODState:
        return self.mdp.initial_state(), ()

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
            raise ValueError("Stored OD prefixes must be incomplete")
        if any(action not in ACTIONS for action in prefix):
            raise ValueError("OD prefix contains an unknown action")
        if self.mdp.is_terminal(state) and prefix:
            raise ValueError("A terminal real state must have an empty prefix")

    def sample_next(self, od_state: ODState, action: Action) -> ODState:
        state, prefix = od_state
        extended = prefix + (action,)
        if len(extended) < self.mdp.n_agents:
            return state, extended
        return self.mdp.sample_next(state, extended, self.transition_rng), ()

    def _completes_real_step(self, od_state: ODState, next_od_state: ODState) -> bool:
        return not next_od_state[1]

    def get_value(self, od_state: ODState) -> float:
        self.validate_od_state(od_state)
        state, prefix = od_state
        if self.mdp.is_terminal(state):
            return 0.0
        stored = self.value_store.get(od_state)
        return self.heuristic.od_value(state, prefix) if stored is None else stored

    def real_state_value(self, state: State) -> float:
        return self.get_value((state, ()))

    def complete_joint_action_value(
        self,
        state: State,
        joint_action: JointAction,
        *,
        deadline: float | None = None,
    ) -> float:
        self._check_deadline(deadline)
        transitions = self.mdp.joint_transitions(state, joint_action)
        expected_cost = 0.0
        for next_state, probability in transitions:
            self._check_deadline(deadline)
            expected_cost += probability * (
                self.mdp.transition_cost(state, joint_action, next_state)
                + self.real_state_value(next_state)
            )
        return expected_cost

    def operator_value(
        self,
        od_state: ODState,
        action: Action,
        *,
        deadline: float | None = None,
    ) -> float:
        self._check_deadline(deadline)
        self.validate_od_state(od_state)
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action!r}")
        state, prefix = od_state
        extended = prefix + (action,)
        if len(extended) < self.mdp.n_agents:
            return self.get_value((state, extended))
        return self.complete_joint_action_value(state, extended, deadline=deadline)

    def _guided_operators(
        self,
        od_state: ODState,
        candidates: list[Action],
    ) -> list[Action]:
        if len(candidates) <= 1:
            return candidates
        state, prefix = od_state
        keyed = [
            (self.heuristic.od_operator_guidance_key(state, prefix, action), action)
            for action in candidates
        ]
        best_key = min(key for key, _ in keyed)
        return [action for key, action in keyed if key == best_key]

    def best_action(
        self,
        od_state: ODState,
        *,
        random_ties: bool,
        deadline: float | None = None,
    ) -> tuple[Action, float]:
        self.validate_od_state(od_state)
        state, _ = od_state
        if self.mdp.is_terminal(state):
            return "stay", 0.0

        best_value = math.inf
        best_actions: list[Action] = []
        for action in ACTIONS:
            value = self.operator_value(od_state, action, deadline=deadline)
            if value < best_value and not self._values_tied(value, best_value):
                best_value = value
                best_actions = [action]
            elif self._values_tied(value, best_value):
                best_actions.append(action)

        if not best_actions:
            raise RuntimeError("No OD operator was generated")
        best_actions = self._guided_operators(od_state, best_actions)
        selected = (
            self.tie_rng.choice(best_actions)
            if random_ties
            else self.tie_breaker.choose(best_actions, od_state)
        )
        return selected, best_value

    def get_successors_for_solved_check(
        self,
        od_state: ODState,
        action: Action,
    ) -> tuple[ODState, ...]:
        state, prefix = od_state
        extended = prefix + (action,)
        if len(extended) < self.mdp.n_agents:
            return ((state, extended),)
        return tuple(
            (next_state, ())
            for next_state, probability in self.mdp.joint_transitions(state, extended)
            if probability > 0.0
        )

    def _bellman_operator_candidates(self, od_state: ODState) -> tuple[Action, ...]:
        self.validate_od_state(od_state)
        cached = self._raw_operator_candidate_cache.get(od_state)
        if cached is not None:
            return cached

        state, prefix = od_state
        if self.mdp.is_terminal(state):
            result = ("stay",)
        elif state[len(prefix)] == self.mdp.goals[len(prefix)]:
            result = ("stay",)
        else:
            best_value = math.inf
            best_actions: list[Action] = []
            for action in ACTIONS:
                value = self.operator_value(od_state, action)
                if value < best_value and not self._values_tied(value, best_value):
                    best_value = value
                    best_actions = [action]
                elif self._values_tied(value, best_value):
                    best_actions.append(action)
            if not best_actions:
                raise RuntimeError("No OD operator was generated")
            result = tuple(best_actions)

        self._raw_operator_candidate_cache[od_state] = result
        return result


    def _guided_complete_joint_actions(
        self,
        state: State,
        candidates: list[JointAction],
    ) -> list[JointAction]:
        if len(candidates) <= 1:
            return candidates
        keyed = [
            (self.heuristic.joint_action_guidance_key(state, action), action)
            for action in candidates
        ]
        best_key = min(key for key, _ in keyed)
        return [action for key, action in keyed if key == best_key]

    def greedy_joint_action_candidates(self, state: State) -> tuple[JointAction, ...]:
        self.mdp.validate_state(state)
        cached = self._joint_policy_candidate_cache.get(state)
        if cached is not None:
            return cached
        if self.mdp.is_terminal(state):
            result = (tuple("stay" for _ in range(self.mdp.n_agents)),)
            self._joint_policy_candidate_cache[state] = result
            return result

        prefixes: list[JointAction] = [()]
        while prefixes and len(prefixes[0]) < self.mdp.n_agents:
            expanded: list[JointAction] = []
            for prefix in prefixes:
                candidates = self._bellman_operator_candidates((state, prefix))
                expanded.extend(prefix + (action,) for action in candidates)
            prefixes = expanded

        if not prefixes:
            raise RuntimeError("No complete OD joint action was generated")
        result = tuple(self._guided_complete_joint_actions(state, prefixes))
        self._joint_policy_candidate_cache[state] = result
        return result

    def policy_action(self, state: State) -> JointAction:
        candidates = self.greedy_joint_action_candidates(state)
        if len(candidates) == 1:
            return candidates[0]
        return self.joint_tie_breaker.choose(candidates, state)
