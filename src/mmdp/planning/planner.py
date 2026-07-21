from __future__ import annotations

"""Generic RTDP/LRTDP planning engine.

``RTDPPlanner`` owns the outer trial loop, the LRTDP solved-labeling
procedure, the stability stopping rule, and resource monitoring.  Everything
state-space specific (transitions, backups, greedy policies) is delegated to
an injected ``PlanningDomain``.
"""

import math
import time
import random
from typing import Generic, Iterator, TypeVar

from mmdp.planning.components import PlanningDomain
from mmdp.planning.config import RTDPConfig
from mmdp.planning.exceptions import DeadlineReached, MemoryLimitReached
from mmdp.planning.numerics import scaled_residual_ratio
from mmdp.resource_monitor import ResourceMonitor

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")
PlanningResultType = TypeVar("PlanningResultType")


class RTDPPlanner(Generic[StateType, ActionType, PlanningResultType]):
    """
    The core RTDP algorithmic engine. It relies entirely on injected
    components (via PlanningDomain) to define the state space, transition model,
    and backup operations.
    """

    def __init__(
        self,
        domain: PlanningDomain[StateType, ActionType],
        config: RTDPConfig,
        result_builder: type[PlanningResultType],
    ) -> None:
        self.domain = domain
        self.config = config
        self.result_builder = result_builder

        self.solved_checks = 0
        self._resource_monitor: ResourceMonitor | None = None

    # ------------------------------------------------------------------
    # Facade: expose the domain surface that evaluation relies on
    # ------------------------------------------------------------------

    @property
    def mdp(self):
        return self.domain.mdp

    @property
    def heuristic(self):
        return getattr(self.domain, "heuristic", None)

    @property
    def resolved_max_steps_per_trial(self) -> int | None:
        return getattr(self.domain, "resolved_max_steps_per_trial", None)

    def policy_action(
        self, state: StateType, *, tie_rng: random.Random | None = None
    ) -> ActionType:
        return self.domain.policy_action(state, tie_rng=tie_rng)

    def policy_action_with_info(
        self, state: StateType, *, tie_rng: random.Random | None = None
    ) -> tuple[ActionType, int]:
        return self.domain.policy_action_with_info(state, tie_rng=tie_rng)

    def global_policy_action(
        self, state: StateType, *, tie_rng: random.Random | None = None
    ) -> ActionType:
        return self.domain.global_policy_action(state, tie_rng=tie_rng)

    def global_policy_action_with_info(
        self, state: StateType, *, tie_rng: random.Random | None = None
    ) -> tuple[ActionType, int]:
        return self.domain.global_policy_action_with_info(state, tie_rng=tie_rng)

    def reset_policy_cache_stats(self) -> None:
        self.domain.reset_policy_cache_stats()

    def policy_cache_stats(self) -> dict[str, int | float]:
        return self.domain.policy_cache_stats()

    # ------------------------------------------------------------------
    # LRTDP solved labeling
    # ------------------------------------------------------------------

    def _check_deadline(self, deadline: float | None) -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineReached
        if (
            self._resource_monitor is not None
            and self._resource_monitor.limit_reached()
        ):
            raise MemoryLimitReached

    def _trial_numbers(self) -> Iterator[int]:
        trial_number = 1
        while (
            self.config.max_trials is None
            or trial_number <= self.config.max_trials
        ):
            yield trial_number
            trial_number += 1

    def _label_trial_path(
        self,
        visited_states: tuple[StateType, ...],
        *,
        deadline: float | None = None,
    ) -> None:
        for state in reversed(visited_states):
            self._check_deadline(deadline)
            if not self.check_solved(state, deadline=deadline):
                break

    def check_solved(
        self,
        root: StateType,
        *,
        deadline: float | None = None,
    ) -> bool:
        self.solved_checks += 1
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
                state, deadline=deadline
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

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self, *, reset: bool = True) -> PlanningResultType:
        if reset:
            self.domain.reset()
        self.domain.reset_caches()

        started_at = time.perf_counter()
        deadline = (
            started_at + self.config.time_limit_seconds
            if self.config.time_limit_seconds is not None
            else None
        )
        monitor = ResourceMonitor(memory_limit_mb=self.config.memory_limit_mb)
        self._resource_monitor = monitor.start()
        self.domain.attach_resource_monitor(monitor)

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
                    trial_result = self.domain.run_trial(deadline=deadline)
                except DeadlineReached:
                    stop_reason = "time_limit"
                    break
                except MemoryLimitReached:
                    stop_reason = "memory_limit"
                    break

                trials_completed += 1
                total_trial_steps += trial_result.steps
                final_trial_residual = trial_result.maximum_residual
                final_trial_scaled_residual = trial_result.maximum_scaled_residual

                if trial_result.reached_goal:
                    goal_reaching_trials += 1
                if trial_result.hit_step_limit:
                    step_limited_trials += 1

                stable = (
                    final_trial_scaled_residual <= 1.0
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
                    except DeadlineReached:
                        stop_reason = "time_limit"
                        break
                    except MemoryLimitReached:
                        stop_reason = "memory_limit"
                        break

                    if self.domain.is_initial_state_solved():
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
            self.domain.detach_resource_monitor()

        elapsed_seconds = time.perf_counter() - started_at

        base_kwargs = dict(
            stop_reason=stop_reason,
            trials_completed=trials_completed,
            goal_reaching_trials=goal_reaching_trials,
            step_limited_trials=step_limited_trials,
            total_trial_steps=total_trial_steps,
            elapsed_seconds=elapsed_seconds,
            bellman_backups=self.domain.bellman_backups,
            transition_outcomes_evaluated=self.domain.transition_outcomes_evaluated,
            final_trial_residual=final_trial_residual,
            final_trial_scaled_residual=final_trial_scaled_residual,
            consecutive_stable_trials=consecutive_stable_trials,
            maximum_consecutive_stable_trials=maximum_consecutive_stable_trials,
            stability_criterion_reached=(first_stability_trial is not None),
            first_stability_trial=first_stability_trial,
            first_stability_elapsed_seconds=first_stability_elapsed_seconds,
            initial_state_solved=self.domain.is_initial_state_solved(),
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

        return self.result_builder(**base_kwargs, **self.domain.build_result_kwargs())
