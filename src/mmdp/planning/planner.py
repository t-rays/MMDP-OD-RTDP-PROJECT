from __future__ import annotations

"""RTDP trial loop with LRTDP solved-state termination."""

import time
from typing import Generic, TypeVar

from mmdp.planning.components import PlanningDomain
from mmdp.planning.config import DeadlineReached, RTDPConfig, scaled_residual_ratio
from mmdp.planning.results import PlanningResult
from mmdp.resource_monitor import ResourceMonitor

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")


class RTDPPlanner(Generic[StateType, ActionType]):
    def __init__(
        self,
        domain: PlanningDomain[StateType, ActionType],
        config: RTDPConfig,
    ) -> None:
        self.domain = domain
        self.config = config

    @property
    def mdp(self):
        return self.domain.mdp


    def policy_action(self, state):
        return self.domain.policy_action(state)

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        if time.perf_counter() >= deadline:
            raise DeadlineReached

    def _label_trial_path(
        self,
        visited_states: tuple[StateType, ...],
        *,
        deadline: float,
    ) -> None:
        for state in reversed(visited_states):
            self._check_deadline(deadline)
            if not self.check_solved(state, deadline=deadline):
                break

    def check_solved(self, root: StateType, *, deadline: float) -> bool:
        if self.domain.is_terminal(root) or self.domain.is_solved(root):
            return True

        open_stack: list[StateType] = [root]
        open_set: set[StateType] = {root}
        closed: list[StateType] = []
        closed_set: set[StateType] = set()
        envelope_is_solved = True

        while open_stack:
            self._check_deadline(deadline)
            state = open_stack.pop()
            open_set.discard(state)
            if self.domain.is_terminal(state) or self.domain.is_solved(state):
                continue
            if state in closed_set:
                continue

            closed.append(state)
            closed_set.add(state)
            old_value = self.domain.get_value(state)
            action, bellman_value = self.domain.get_best_action_and_value_for_solved_check(
                state,
                deadline=deadline,
            )
            if scaled_residual_ratio(
                old_value,
                bellman_value,
                absolute_tolerance=self.config.epsilon,
                relative_tolerance=self.config.relative_epsilon,
            ) > 1.0:
                envelope_is_solved = False
                continue

            for successor in self.domain.get_successors_for_solved_check(state, action):
                if (
                    self.domain.is_terminal(successor)
                    or self.domain.is_solved(successor)
                    or successor in closed_set
                    or successor in open_set
                ):
                    continue
                open_stack.append(successor)
                open_set.add(successor)

        if envelope_is_solved:
            self.domain.mark_solved(closed_set)
            return True

        for state in reversed(closed):
            self._check_deadline(deadline)
            if not self.domain.is_solved(state):
                self.domain.backup(state, deadline=deadline)
        return False

    def solve(self) -> PlanningResult:
        self.domain.reset()
        started_at = time.perf_counter()
        deadline = started_at + self.config.time_limit_seconds
        monitor = ResourceMonitor().start()
        stop_reason = "time_limit"

        try:
            while True:
                visited_states = self.domain.run_trial(deadline=deadline)
                self._label_trial_path(visited_states, deadline=deadline)
                if self.domain.is_initial_state_solved():
                    stop_reason = "initial_state_solved"
                    break
                self._check_deadline(deadline)
        except DeadlineReached:
            stop_reason = "time_limit"
        finally:
            snapshot = monitor.stop()

        return PlanningResult(
            stop_reason=stop_reason,
            elapsed_seconds=time.perf_counter() - started_at,
            peak_rss_delta_mb=snapshot.peak_rss_delta_mb,
        )
