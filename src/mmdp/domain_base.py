from __future__ import annotations

"""Shared plumbing for RTDP planning domains.

``RTDPDomainBase`` implements everything that is identical between Baseline
RTDP and OD-RTDP: step-limit resolution, numeric tie comparison, deadline and
memory-limit checks, RNG management, Bellman-backup residual bookkeeping,
policy-cache statistics, and the forward-simulation trial loop.  Subclasses
supply the state space (``initial_state`` / ``sample_next`` / ``is_terminal``),
the greedy action selection (``best_action``), and their algorithm-specific
metrics.
"""

import math
import random
import time
from abc import abstractmethod
from typing import Any, Generic, Iterator, TypeVar

from mmdp.components import PlanningDomain, SolvedTracker, TieBreaker, ValueStore
from mmdp.config import RTDPConfig
from mmdp.exceptions import DeadlineReached, MemoryLimitReached
from mmdp.grid_mmdp import GridMMDP
from mmdp.numerics import scaled_residual_ratio, tied_by_ulp
from mmdp.results import TrialResult

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")


class RTDPDomainBase(PlanningDomain[StateType, ActionType], Generic[StateType, ActionType]):
    """Common behavior for RTDP planning domains over a ``GridMMDP``."""

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

        self._policy_cache_hits = 0
        self._policy_cache_misses = 0

        self.bellman_backups = 0
        self.transition_outcomes_evaluated = 0

        self.transition_rng = random.Random(self.config.seed)
        self.tie_rng = random.Random(self.config.seed + 1)

    # ------------------------------------------------------------------
    # Limits and numeric comparisons
    # ------------------------------------------------------------------

    def _resolve_max_steps_per_trial(self) -> int | None:
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

        from mmdp.limits import sequential_multi_agent_step_bound

        return sequential_multi_agent_step_bound(
            distances,
            success_probability,
            self.config.step_tail_probability,
        )

    def _values_tied(self, first: float, second: float) -> bool:
        if self.config.tie_tolerance is not None:
            return math.isclose(
                first, second, rel_tol=0.0, abs_tol=self.config.tie_tolerance
            )
        return tied_by_ulp(first, second, ulps=self.config.tie_ulps)

    def _check_deadline(self, deadline: float | None) -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineReached
        monitor = self._resource_monitor
        if monitor is not None and monitor.limit_reached():
            raise MemoryLimitReached

    # ------------------------------------------------------------------
    # Solved-state tracking
    # ------------------------------------------------------------------

    def is_solved(self, state: StateType) -> bool:
        return self.solved_tracker.is_solved(state)

    def is_initial_state_solved(self) -> bool:
        initial = self.initial_state()
        return self.is_terminal(initial) or self.is_solved(initial)

    def mark_solved(self, states: set[StateType]) -> None:
        self.solved_tracker.mark_solved(states)

    # ------------------------------------------------------------------
    # Reset scaffolding
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.value_store.clear()
        self.solved_tracker.clear()
        self.reset_caches()

        self.bellman_backups = 0
        self.transition_outcomes_evaluated = 0
        self._reset_domain_metrics()

        self.transition_rng.seed(self.config.seed)
        self.tie_rng.seed(self.config.seed + 1)

    @abstractmethod
    def _reset_domain_metrics(self) -> None:
        """Reset algorithm-specific counters and caches."""
        ...

    # ------------------------------------------------------------------
    # Policy-cache statistics
    # ------------------------------------------------------------------

    def reset_policy_cache_stats(self) -> None:
        self._policy_cache_hits = 0
        self._policy_cache_misses = 0

    @abstractmethod
    def _policy_cache_entries(self) -> int:
        """Return the number of memoized greedy policy decisions."""
        ...

    def policy_cache_stats(self) -> dict[str, int | float]:
        total = self._policy_cache_hits + self._policy_cache_misses
        return {
            "hits": self._policy_cache_hits,
            "misses": self._policy_cache_misses,
            "entries": self._policy_cache_entries(),
            "hit_rate": (self._policy_cache_hits / total if total else 0.0),
        }

    # ------------------------------------------------------------------
    # Bellman backup
    # ------------------------------------------------------------------

    @abstractmethod
    def best_action(
        self,
        state: StateType,
        *,
        count_metrics: bool = True,
        random_ties: bool = True,
        deadline: float | None = None,
    ) -> tuple[ActionType, float]:
        """Return the greedy action and its Bellman value for one state."""
        ...

    def backup(
        self,
        state: StateType,
        *,
        deadline: float | None = None,
    ) -> tuple[ActionType, float, float]:
        old_value = self.get_value(state)
        action, new_value = self.best_action(
            state, count_metrics=True, random_ties=True, deadline=deadline
        )
        self.value_store.set(state, new_value)
        self.bellman_backups += 1
        residual = abs(new_value - old_value)
        scaled = scaled_residual_ratio(
            old_value,
            new_value,
            absolute_tolerance=self.config.epsilon,
            relative_tolerance=self.config.relative_epsilon,
        )
        return action, residual, scaled

    def get_best_action_and_value_for_solved_check(
        self, state: StateType, *, deadline: float | None = None
    ) -> tuple[ActionType, float]:
        return self.best_action(
            state, count_metrics=True, random_ties=False, deadline=deadline
        )

    # ------------------------------------------------------------------
    # Forward-simulation trial
    # ------------------------------------------------------------------

    def _completes_real_step(self, state: StateType, next_state: StateType) -> bool:
        """Return True when moving to ``next_state`` finishes one real
        environment step.  Baseline steps once per action; OD only completes a
        real step when a full joint action has been assembled and executed."""
        return True

    def _trial_step_numbers(self) -> Iterator[int]:
        step_number = 0
        while (
            self.resolved_max_steps_per_trial is None
            or step_number < self.resolved_max_steps_per_trial
        ):
            yield step_number
            step_number += 1

    def run_trial(self, *, deadline: float | None = None) -> TrialResult[StateType]:
        state = self.initial_state()
        maximum_residual = 0.0
        maximum_scaled_residual = 0.0
        steps = 0
        visited_states: list[StateType] = []

        if self.is_terminal(state):
            return TrialResult(0.0, 0.0, 0, True, False, False, ())

        step_cap = self.resolved_max_steps_per_trial
        while step_cap is None or steps < step_cap:
            self._check_deadline(deadline)

            if self.config.stop_when_solved and self.is_solved(state):
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
            action, residual, scaled = self.backup(state, deadline=deadline)
            maximum_residual = max(maximum_residual, residual)
            maximum_scaled_residual = max(maximum_scaled_residual, scaled)

            next_state = self.sample_next(state, action)
            if self._completes_real_step(state, next_state):
                steps += 1
            if self.is_terminal(next_state):
                return TrialResult(
                    maximum_residual,
                    maximum_scaled_residual,
                    steps,
                    True,
                    False,
                    False,
                    tuple(visited_states),
                )
            state = next_state

        return TrialResult(
            maximum_residual,
            maximum_scaled_residual,
            steps,
            False,
            False,
            True,
            tuple(visited_states),
        )
