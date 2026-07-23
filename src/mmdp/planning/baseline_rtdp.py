from __future__ import annotations

"""Baseline RTDP over complete joint states and complete joint actions."""

import math
from collections.abc import Callable

from mmdp.domain.grid_mmdp import ACTIONS, GridMMDP, JointAction, State
from mmdp.domain.heuristic import ShortestPathHeuristic
from mmdp.planning.components import SolvedTracker, TieBreaker, ValueStore
from mmdp.planning.config import RTDPConfig
from mmdp.planning.domain_base import RTDPDomainBase

StateHeuristic = Callable[[State], float]


class BaselineDomain(RTDPDomainBase[State, JointAction]):
    def __init__(
        self,
        mdp: GridMMDP,
        heuristic: StateHeuristic | None,
        config: RTDPConfig,
        value_store: ValueStore[State],
        solved_tracker: SolvedTracker[State],
        tie_breaker: TieBreaker[JointAction],
    ) -> None:
        self._policy_candidate_cache: dict[State, tuple[JointAction, ...]] = {}
        self._state_step_cost_cache: dict[State, float] = {}
        super().__init__(
            mdp=mdp,
            heuristic=heuristic or ShortestPathHeuristic(mdp),
            config=config,
            value_store=value_store,
            solved_tracker=solved_tracker,
            tie_breaker=tie_breaker,
        )

    def reset_caches(self) -> None:
        self._policy_candidate_cache.clear()
        self._state_step_cost_cache.clear()

    def initial_state(self) -> State:
        return self.mdp.initial_state()

    def is_terminal(self, state: State) -> bool:
        return self.mdp.is_terminal(state)

    def sample_next(self, state: State, action: JointAction) -> State:
        return self.mdp.sample_next(state, action, self.transition_rng)

    def get_value(self, state: State) -> float:
        if self.mdp.is_terminal(state):
            return 0.0
        stored = self.value_store.get(state)
        return self.heuristic(state) if stored is None else stored

    def _value_unchecked(self, state: State) -> float:
        if state == self.mdp.goals:
            return 0.0
        stored = self.value_store.get(state)
        return self.heuristic(state) if stored is None else stored

    def _step_cost(self, state: State) -> float:
        cached = self._state_step_cost_cache.get(state)
        if cached is None:
            cached = float(
                sum(position != goal for position, goal in zip(state, self.mdp.goals))
            )
            self._state_step_cost_cache[state] = cached
        return cached

    def q_value(
        self,
        state: State,
        joint_action: JointAction,
        *,
        deadline: float | None = None,
    ) -> float:
        self._check_deadline(deadline)
        transitions = self.mdp.joint_transitions(state, joint_action)
        immediate_cost = self._step_cost(state)
        expected_cost = 0.0
        for next_state, probability in transitions:
            self._check_deadline(deadline)
            expected_cost += probability * (immediate_cost + self._value_unchecked(next_state))
        return expected_cost

    def _guided_joint_actions(
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

    def best_action(
        self,
        state: State,
        *,
        random_ties: bool,
        deadline: float | None = None,
    ) -> tuple[JointAction, float]:
        if self.mdp.is_terminal(state):
            return tuple("stay" for _ in range(self.mdp.n_agents)), 0.0

        best_value = math.inf
        best_actions: list[JointAction] = []
        for joint_action in self.mdp.all_joint_actions():
            value = self.q_value(state, joint_action, deadline=deadline)
            if value < best_value and not self._values_tied(value, best_value):
                best_value = value
                best_actions = [joint_action]
            elif self._values_tied(value, best_value):
                best_actions.append(joint_action)

        if not best_actions:
            raise RuntimeError("No joint action was generated")
        best_actions = self._guided_joint_actions(state, best_actions)
        selected = (
            self.tie_rng.choice(best_actions)
            if random_ties
            else self.tie_breaker.choose(best_actions, state)
        )
        return selected, best_value

    def get_successors_for_solved_check(
        self,
        state: State,
        action: JointAction,
    ) -> tuple[State, ...]:
        return tuple(
            next_state
            for next_state, probability in self.mdp.joint_transitions(state, action)
            if probability > 0.0
        )

    def greedy_action_candidates(self, state: State) -> tuple[JointAction, ...]:
        self.mdp.validate_state(state)
        cached = self._policy_candidate_cache.get(state)
        if cached is not None:
            return cached
        if self.mdp.is_terminal(state):
            result = (tuple("stay" for _ in range(self.mdp.n_agents)),)
            self._policy_candidate_cache[state] = result
            return result

        intended_by_agent = tuple(
            {
                action: self.mdp.move_one(agent_index, position, action)
                for action in ACTIONS
            }
            for agent_index, position in enumerate(state)
        )
        physical_q_cache: dict[State, float] = {}
        best_value = math.inf
        best_actions: list[JointAction] = []

        for joint_action in self.mdp.all_joint_actions():
            intended_state = tuple(
                intended_by_agent[index][action]
                for index, action in enumerate(joint_action)
            )
            value = physical_q_cache.get(intended_state)
            if value is None:
                value = self.q_value(state, joint_action)
                physical_q_cache[intended_state] = value
            if value < best_value and not self._values_tied(value, best_value):
                best_value = value
                best_actions = [joint_action]
            elif self._values_tied(value, best_value):
                best_actions.append(joint_action)

        if not best_actions:
            raise RuntimeError("No joint action was generated")
        result = tuple(self._guided_joint_actions(state, best_actions))
        self._policy_candidate_cache[state] = result
        return result

    def policy_action(self, state: State) -> JointAction:
        candidates = self.greedy_action_candidates(state)
        if len(candidates) == 1:
            return candidates[0]
        return self.tie_breaker.choose(candidates, state)
