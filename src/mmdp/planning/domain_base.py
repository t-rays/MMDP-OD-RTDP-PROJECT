from __future__ import annotations

"""Shared RTDP trial logic for Baseline RTDP and OD-RTDP."""

import math
import random
import time
from abc import abstractmethod
from typing import Any, Generic, TypeVar

from mmdp.domain.grid_mmdp import GridMMDP
from mmdp.planning.components import PlanningDomain, SolvedTracker, TieBreaker, ValueStore
from mmdp.planning.config import (
    DeadlineReached,
    RTDPConfig,
    sequential_multi_agent_step_bound,
    tied_by_ulp,
)

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")


class RTDPDomainBase(PlanningDomain[StateType, ActionType], Generic[StateType, ActionType]):
    def __init__(
        self,
        mdp: GridMMDP,
        heuristic: Any,
        config: RTDPConfig,
        value_store: ValueStore[StateType],
        solved_tracker: SolvedTracker[StateType],
        tie_breaker: TieBreaker[ActionType],
    ) -> None:
        self.mdp = mdp
        self.heuristic = heuristic
        self.config = config
        self.value_store = value_store
        self.solved_tracker = solved_tracker
        self.tie_breaker = tie_breaker
        self.resolved_max_steps_per_trial = self._resolve_max_steps_per_trial()
        self.transition_rng = random.Random(config.seed)
        self.tie_rng = random.Random(config.seed + 1)

    def _resolve_max_steps_per_trial(self) -> int:
        distances = self.heuristic.distance_summary(self.mdp.initial_state())
        if any(math.isinf(distance) for distance in distances):
            raise ValueError("At least one start cannot reach its assigned goal")
        success_probability = 1.0 - self.mdp.config.slip_to_stay_probability
        if success_probability <= 0.0:
            raise ValueError("Movement success probability must be positive")
        return sequential_multi_agent_step_bound(
            distances,
            success_probability,
            self.config.step_tail_probability,
        )

    def _values_tied(self, first: float, second: float) -> bool:
        return tied_by_ulp(first, second, ulps=self.config.tie_ulps)

    @staticmethod
    def _check_deadline(deadline: float | None) -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineReached

    def is_solved(self, state: StateType) -> bool:
        return self.solved_tracker.is_solved(state)

    def is_initial_state_solved(self) -> bool:
        initial = self.initial_state()
        return self.is_terminal(initial) or self.is_solved(initial)

    def mark_solved(self, states: set[StateType]) -> None:
        self.solved_tracker.mark_solved(states)

    def reset(self) -> None:
        self.value_store.clear()
        self.solved_tracker.clear()
        self.reset_caches()
        self.transition_rng.seed(self.config.seed)
        self.tie_rng.seed(self.config.seed + 1)

    @abstractmethod
    def best_action(
        self,
        state: StateType,
        *,
        random_ties: bool,
        deadline: float | None = None,
    ) -> tuple[ActionType, float]: ...

    def backup(self, state: StateType, *, deadline: float | None = None) -> ActionType:
        action, new_value = self.best_action(
            state,
            random_ties=True,
            deadline=deadline,
        )
        self.value_store.set(state, new_value)
        return action

    def get_best_action_and_value_for_solved_check(
        self, state: StateType, *, deadline: float | None = None
    ) -> tuple[ActionType, float]:
        return self.best_action(state, random_ties=False, deadline=deadline)

    def _completes_real_step(self, state: StateType, next_state: StateType) -> bool:
        return True

    def run_trial(self, *, deadline: float | None = None) -> tuple[StateType, ...]:
        state = self.initial_state()
        visited_states: list[StateType] = []
        real_steps = 0

        while real_steps < self.resolved_max_steps_per_trial:
            self._check_deadline(deadline)
            if self.is_terminal(state) or self.is_solved(state):
                break

            visited_states.append(state)
            action = self.backup(state, deadline=deadline)
            next_state = self.sample_next(state, action)
            if self._completes_real_step(state, next_state):
                real_steps += 1
            state = next_state

        return tuple(visited_states)
