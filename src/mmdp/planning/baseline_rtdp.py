from __future__ import annotations

"""Baseline RTDP: value iteration trials over complete joint states/actions."""

import math
import random
from typing import Any, Callable

from mmdp.planning.components import SolvedTracker, TieBreaker, ValueStore
from mmdp.planning.config import RTDPConfig
from mmdp.planning.domain_base import RTDPDomainBase
from mmdp.domain.grid_mmdp import ACTIONS, GridMMDP, JointAction, State
from mmdp.domain.heuristic import ShortestPathHeuristic

StateHeuristic = Callable[[State], float]


class BaselineDomain(RTDPDomainBase[State, JointAction]):
    """Standard RTDP over complete states and complete joint actions."""

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
        self.planning_action_evaluations = 0

        super().__init__(
            mdp=mdp,
            heuristic=heuristic if heuristic is not None else ShortestPathHeuristic(mdp),
            config=config,
            value_store=value_store,
            solved_tracker=solved_tracker,
            tie_breaker=tie_breaker,
        )

    def _reset_domain_metrics(self) -> None:
        self.planning_action_evaluations = 0
        self._state_step_cost_cache.clear()

    def reset_caches(self) -> None:
        self._policy_candidate_cache.clear()
        self.reset_policy_cache_stats()

    def _policy_cache_entries(self) -> int:
        return len(self._policy_candidate_cache)

    # ------------------------------------------------------------------
    # State space
    # ------------------------------------------------------------------

    def initial_state(self) -> State:
        return self.mdp.initial_state()

    def is_terminal(self, state: State) -> bool:
        return self.mdp.is_terminal(state)

    def sample_next(self, state: State, action: JointAction) -> State:
        return self.mdp.sample_next(state, action, self.transition_rng)

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def get_value(self, state: State) -> float:
        if self.mdp.is_terminal(state):
            return 0.0
        stored = self.value_store.get(state)
        if stored is not None:
            return stored
        return self.heuristic(state)

    def _value_unchecked(self, state: State) -> float:
        if state == self.mdp.goals:
            return 0.0
        stored = self.value_store.get(state)
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
        self._check_deadline(deadline)
        transitions = self.mdp.joint_transitions(state, joint_action)
        self._check_deadline(deadline)

        if count_metrics:
            self.planning_action_evaluations += 1
            self.transition_outcomes_evaluated += len(transitions)

        immediate_cost = self._step_cost_unchecked(state)
        expected_cost = 0.0

        for next_state, probability in transitions:
            self._check_deadline(deadline)
            future_cost = self._value_unchecked(next_state)
            expected_cost += probability * (immediate_cost + future_cost)

        return expected_cost

    # ------------------------------------------------------------------
    # Greedy action selection
    # ------------------------------------------------------------------

    def _guided_joint_actions(
        self,
        state: State,
        candidates: list[JointAction],
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
        if self.mdp.is_terminal(state):
            stay_action: JointAction = tuple("stay" for _ in range(self.mdp.n_agents))
            return stay_action, 0.0

        best_value = math.inf
        best_actions: list[JointAction] = []

        for joint_action in self.mdp.all_joint_actions():
            self._check_deadline(deadline)
            action_value = self.q_value(
                state,
                joint_action,
                count_metrics=count_metrics,
                deadline=deadline,
            )

            if action_value < best_value and not self._values_tied(action_value, best_value):
                best_value = action_value
                best_actions = [joint_action]
            elif self._values_tied(action_value, best_value):
                best_actions.append(joint_action)

        if not best_actions:
            raise RuntimeError("No joint action was generated")

        best_actions = self._guided_joint_actions(state, best_actions)

        if random_ties:
            selected_action = self.tie_rng.choice(best_actions)
        else:
            selected_action = self.tie_breaker.choose(best_actions, state)

        return selected_action, best_value

    # ------------------------------------------------------------------
    # LRTDP solved check
    # ------------------------------------------------------------------

    def get_successors_for_solved_check(
        self, state: State, action: JointAction
    ) -> tuple[State, ...]:
        transitions = self.mdp.joint_transitions(state, action)
        return tuple(
            next_state
            for next_state, probability in transitions
            if probability > 0.0
        )

    # ------------------------------------------------------------------
    # Planning metrics
    # ------------------------------------------------------------------

    def build_result_kwargs(self) -> dict[str, Any]:
        return {
            "planning_action_evaluations": self.planning_action_evaluations,
            "visited_states": len(self.value_store),
            "solved_states": len(self.solved_tracker),
        }

    # ------------------------------------------------------------------
    # Fixed-policy extraction (used by evaluation)
    # ------------------------------------------------------------------

    def greedy_action_candidates(self, state: State) -> tuple[JointAction, ...]:
        self.mdp.validate_state(state)
        cached = self._policy_candidate_cache.get(state)
        if cached is not None:
            self._policy_cache_hits += 1
            return cached

        self._policy_cache_misses += 1

        if self.mdp.is_terminal(state):
            terminal_candidates = (tuple("stay" for _ in range(self.mdp.n_agents)),)
            self._policy_candidate_cache[state] = terminal_candidates
            return terminal_candidates

        best_value = math.inf
        best_actions: list[JointAction] = []

        intended_by_agent = tuple(
            {action: self.mdp.move_one(agent_index, position, action) for action in ACTIONS}
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
                    state, joint_action, count_metrics=False, deadline=None
                )
                physical_q_cache[intended_state] = action_value

            if action_value < best_value and not self._values_tied(action_value, best_value):
                best_value = action_value
                best_actions = [joint_action]
            elif self._values_tied(action_value, best_value):
                best_actions.append(joint_action)

        if not best_actions:
            raise RuntimeError("No joint action was generated")

        best_actions = self._guided_joint_actions(state, best_actions)
        candidates = tuple(best_actions)
        self._policy_candidate_cache[state] = candidates
        return candidates

    def policy_action_with_info(
        self, state: State, *, tie_rng: random.Random | None = None
    ) -> tuple[JointAction, int]:
        candidates = self.greedy_action_candidates(state)
        tie_decisions = int(len(candidates) > 1)

        if len(candidates) == 1:
            return candidates[0], tie_decisions

        if tie_rng is not None:
            return tie_rng.choice(candidates), tie_decisions

        return (
            self.tie_breaker.choose(list(candidates), state),
            tie_decisions,
        )

    def policy_action(
        self, state: State, *, tie_rng: random.Random | None = None
    ) -> JointAction:
        action, _ = self.policy_action_with_info(state, tie_rng=tie_rng)
        return action
