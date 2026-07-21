from __future__ import annotations

"""
Stochastic cooperative grid MMDP used by Baseline RTDP and OD-RTDP.

The module defines the environment only. It does not choose actions.

Objective
---------
The problem is formulated as a cost-minimization Stochastic Shortest Path:

    transition cost = number of agents that have not yet reached their goals

Summing this cost over a complete execution equals the sum of individual
arrival times when agents remain frozen after reaching their goals.

Stochastic movement
-------------------
Each intended movement succeeds with probability ``1 - slip`` and otherwise
leaves that agent in place. Outcomes are independent before collision handling.

Collision rule
--------------
When a raw joint outcome contains a vertex collision or an edge swap, the
complete joint movement is rejected and the environment remains in the current
state. No arbitrary numerical collision penalty is used.
"""

from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
import math
import random
from typing import Iterator

from map_creator import MapInstance, Position


State = tuple[Position, ...]
Action = str
JointAction = tuple[Action, ...]
Transition = tuple[State, float]


ACTIONS: tuple[Action, ...] = (
    "stay",
    "up",
    "down",
    "left",
    "right",
)


MOVE: dict[Action, Position] = {
    "stay": (0, 0),
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(frozen=True)
class MMDPConfig:
    """Parameters defining the stochastic grid environment.

    ``transition_cache_max_entries`` bounds each of the raw and resolved LRU
    transition caches.  ``None`` keeps the old unbounded behavior and ``0``
    disables new cache entries.  Bounding the caches is important on large
    multi-agent instances, where the number of state-action pairs grows as
    5**n and an unbounded cache can dominate memory.
    """

    slip_to_stay_probability: float = 0.20
    freeze_agents_at_goal: bool = True
    reject_conflicting_transitions: bool = True
    transition_cache_max_entries: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.slip_to_stay_probability <= 1.0:
            raise ValueError(
                "slip_to_stay_probability must be between 0 and 1"
            )

        if (
            self.transition_cache_max_entries is not None
            and self.transition_cache_max_entries < 0
        ):
            raise ValueError(
                "transition_cache_max_entries must be non-negative or None"
            )


class GridMMDP:
    """Centralized cooperative MMDP built from a MapInstance."""

    def __init__(
        self,
        instance: MapInstance,
        config: MMDPConfig | None = None,
    ) -> None:
        self.instance = instance
        self.config = config or MMDPConfig()

        # Transition distributions are stationary for a fixed MMDP.
        # Cache both raw and collision-resolved distributions so repeated
        # Bellman backups do not rebuild the same Cartesian products.
        self._raw_transition_cache: OrderedDict[
            tuple[State, JointAction],
            tuple[Transition, ...],
        ] = OrderedDict()
        self._resolved_transition_cache: OrderedDict[
            tuple[State, JointAction],
            tuple[Transition, ...],
        ] = OrderedDict()

        # Planning benefits from full transition memoization. During policy
        # evaluation, however, evaluating all candidate actions for a new
        # state can insert 5**n distributions even though only one action is
        # executed. The write switch lets evaluation reuse existing planning
        # entries without filling the cache with every rejected candidate.
        self._transition_cache_write_enabled = True

        self._raw_transition_cache_hits = 0
        self._raw_transition_cache_misses = 0
        self._raw_transition_cache_writes = 0
        self._raw_transition_cache_evictions = 0
        self._resolved_transition_cache_hits = 0
        self._resolved_transition_cache_misses = 0
        self._resolved_transition_cache_writes = 0
        self._resolved_transition_cache_evictions = 0

        self._validate_instance()

    @property
    def map_name(self) -> str:
        return self.instance.grid_map.name

    @property
    def width(self) -> int:
        return self.instance.grid_map.width

    @property
    def height(self) -> int:
        return self.instance.grid_map.height

    @property
    def obstacles(self) -> frozenset[Position]:
        return self.instance.grid_map.obstacles

    @property
    def free_cells(self) -> frozenset[Position]:
        return self.instance.grid_map.free_cells

    @property
    def starts(self) -> State:
        return self.instance.starts

    @property
    def goals(self) -> State:
        return self.instance.goals

    @property
    def n_agents(self) -> int:
        return self.instance.n_agents

    def _validate_instance(self) -> None:
        if self.n_agents <= 0:
            raise ValueError(
                "MapInstance must contain at least one agent"
            )

        if len(self.starts) != len(self.goals):
            raise ValueError(
                "MapInstance must contain the same number of starts and goals"
            )

        if len(set(self.starts)) != len(self.starts):
            raise ValueError(
                "Agent start positions must be unique"
            )

        if len(set(self.goals)) != len(self.goals):
            raise ValueError(
                "Agent goal positions must be unique"
            )

        for label, positions in (
            ("start", self.starts),
            ("goal", self.goals),
        ):
            for agent_index, position in enumerate(positions):
                if position not in self.free_cells:
                    raise ValueError(
                        f"Agent {agent_index} {label} "
                        f"{position} is not a free cell"
                    )

    def initial_state(self) -> State:
        return self.starts

    def goal_state(self) -> State:
        return self.goals

    def validate_state(self, state: State) -> None:
        if len(state) != self.n_agents:
            raise ValueError(
                f"State has {len(state)} positions; "
                f"expected {self.n_agents}"
            )

        for agent_index, position in enumerate(state):
            if position not in self.free_cells:
                raise ValueError(
                    f"Agent {agent_index} is on an invalid cell: {position}"
                )

    def is_terminal(self, state: State) -> bool:
        self.validate_state(state)
        return state == self.goals

    def validate_joint_action(
        self,
        joint_action: JointAction,
    ) -> None:
        if len(joint_action) != self.n_agents:
            raise ValueError(
                f"Joint action has {len(joint_action)} actions; "
                f"expected {self.n_agents}"
            )

        invalid_actions = [
            action
            for action in joint_action
            if action not in MOVE
        ]

        if invalid_actions:
            raise ValueError(
                f"Unknown actions: {invalid_actions}"
            )

    def unfinished_agent_count(
        self,
        state: State,
    ) -> int:
        """Return the number of agents not yet at their assigned goals."""
        self.validate_state(state)

        return sum(
            position != goal
            for position, goal in zip(state, self.goals)
        )

    def transition_cost(
        self,
        state: State,
        joint_action: JointAction,
        next_state: State,
    ) -> float:
        """
        Return the non-negative cost of one real environment transition.

        The current model uses:

            c(s,a,s') = number of unfinished agents in s

        The action and successor are validated even though the numerical value
        currently depends only on the current state.
        """
        self.validate_state(state)
        self.validate_joint_action(joint_action)
        self.validate_state(next_state)

        if self.is_terminal(state):
            return 0.0

        return float(
            self.unfinished_agent_count(state)
        )

    def move_one(
        self,
        agent_index: int,
        position: Position,
        action: Action,
    ) -> Position:
        """Apply one intended deterministic movement."""
        if not 0 <= agent_index < self.n_agents:
            raise IndexError(
                f"Agent index {agent_index} is outside the valid range"
            )

        if action not in MOVE:
            raise ValueError(
                f"Unknown action: {action!r}"
            )

        if position not in self.free_cells:
            raise ValueError(
                f"Position {position} is not a free cell"
            )

        if (
            self.config.freeze_agents_at_goal
            and position == self.goals[agent_index]
        ):
            return position

        dx, dy = MOVE[action]
        candidate = (
            position[0] + dx,
            position[1] + dy,
        )

        if candidate in self.free_cells:
            return candidate

        return position

    def individual_outcomes(
        self,
        agent_index: int,
        position: Position,
        action: Action,
    ) -> tuple[tuple[Position, float], ...]:
        """Return all possible stochastic outcomes for one agent."""
        intended_position = self.move_one(
            agent_index,
            position,
            action,
        )

        success_probability = (
            1.0 - self.config.slip_to_stay_probability
        )

        outcomes: dict[Position, float] = defaultdict(float)
        outcomes[intended_position] += success_probability
        outcomes[position] += self.config.slip_to_stay_probability

        result = tuple(
            (next_position, probability)
            for next_position, probability in outcomes.items()
            if probability > 0.0
        )

        probability_sum = sum(
            probability
            for _, probability in result
        )

        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Individual transition probabilities sum to "
                f"{probability_sum}, not 1"
            )

        return result

    def _cache_lookup(
        self,
        cache: OrderedDict[
            tuple[State, JointAction],
            tuple[Transition, ...],
        ],
        key: tuple[State, JointAction],
    ) -> tuple[Transition, ...] | None:
        """Return and mark an LRU cache entry as recently used."""
        try:
            value = cache[key]
        except KeyError:
            return None
        cache.move_to_end(key)
        return value

    def _cache_store(
        self,
        cache: OrderedDict[
            tuple[State, JointAction],
            tuple[Transition, ...],
        ],
        key: tuple[State, JointAction],
        value: tuple[Transition, ...],
        *,
        raw: bool,
    ) -> bool:
        """Store one transition distribution and enforce the LRU bound."""
        if not self._transition_cache_write_enabled:
            return False

        limit = self.config.transition_cache_max_entries
        if limit == 0:
            return False

        cache[key] = value
        cache.move_to_end(key)

        if limit is not None and len(cache) > limit:
            cache.popitem(last=False)
            if raw:
                self._raw_transition_cache_evictions += 1
            else:
                self._resolved_transition_cache_evictions += 1

        return True

    def _raw_joint_transitions(
        self,
        state: State,
        joint_action: JointAction,
    ) -> tuple[Transition, ...]:
        """
        Return joint outcomes before applying the collision rejection rule.

        The distribution is cached because it depends only on the immutable
        MMDP definition, the current state, and the joint action.
        """
        # Callers validate once before entering this private hot path.
        cache_key = (
            state,
            joint_action,
        )

        cached = self._cache_lookup(
            self._raw_transition_cache,
            cache_key,
        )

        if cached is not None:
            self._raw_transition_cache_hits += 1
            return cached

        self._raw_transition_cache_misses += 1

        if self.is_terminal(state):
            terminal_transition = ((state, 1.0),)
            if self._cache_store(
                self._raw_transition_cache,
                cache_key,
                terminal_transition,
                raw=True,
            ):
                self._raw_transition_cache_writes += 1
            return terminal_transition

        per_agent_outcomes = [
            self.individual_outcomes(
                agent_index,
                position,
                action,
            )
            for agent_index, (position, action)
            in enumerate(zip(state, joint_action))
        ]

        merged: dict[State, float] = defaultdict(float)

        for combination in product(*per_agent_outcomes):
            next_state: State = tuple(
                position
                for position, _ in combination
            )

            probability = math.prod(
                outcome_probability
                for _, outcome_probability in combination
            )

            merged[next_state] += probability

        transitions = tuple(
            (next_state, probability)
            for next_state, probability in merged.items()
            if probability > 0.0
        )

        self._validate_probability_sum(
            transitions,
            label="Raw joint",
        )

        if self._cache_store(
            self._raw_transition_cache,
            cache_key,
            transitions,
            raw=True,
        ):
            self._raw_transition_cache_writes += 1

        return transitions

    @staticmethod
    def vertex_conflict_count(
        state: State,
    ) -> int:
        """Count pairs of agents occupying the same cell."""
        counts: dict[Position, int] = defaultdict(int)

        for position in state:
            counts[position] += 1

        return sum(
            count * (count - 1) // 2
            for count in counts.values()
            if count > 1
        )

    @staticmethod
    def edge_swap_conflict_count(
        state: State,
        next_state: State,
    ) -> int:
        """Count pairs of agents that swap positions in one step."""
        if len(state) != len(next_state):
            raise ValueError(
                "state and next_state must contain the same number of agents"
            )

        conflicts = 0

        for first_agent in range(len(state)):
            for second_agent in range(
                first_agent + 1,
                len(state),
            ):
                if (
                    state[first_agent] == next_state[second_agent]
                    and state[second_agent] == next_state[first_agent]
                    and state[first_agent] != state[second_agent]
                ):
                    conflicts += 1

        return conflicts

    def has_conflict(
        self,
        state: State,
        next_state: State,
    ) -> bool:
        """Return True for a vertex conflict or an edge-swap conflict.

        This boolean check is on the transition-generation hot path.  It uses
        early exits instead of constructing full conflict counts, while the
        public counting helpers remain available for diagnostics.
        """
        if len(set(next_state)) != len(next_state):
            return True

        n_agents = len(state)
        for first_agent in range(n_agents):
            first_start = state[first_agent]
            first_end = next_state[first_agent]
            for second_agent in range(first_agent + 1, n_agents):
                if (
                    first_start == next_state[second_agent]
                    and state[second_agent] == first_end
                    and first_start != state[second_agent]
                ):
                    return True
        return False

    def conflict_probability(
        self,
        state: State,
        joint_action: JointAction,
    ) -> float:
        """Return the probability mass of conflicting raw outcomes."""
        self.validate_state(state)
        self.validate_joint_action(joint_action)
        return sum(
            probability
            for next_state, probability
            in self._raw_joint_transitions(state, joint_action)
            if self.has_conflict(state, next_state)
        )

    def self_loop_probability(
        self,
        state: State,
        joint_action: JointAction,
    ) -> float:
        """Return ``P(next_state == state | state, joint_action)``.

        This includes every way a real step can leave the full joint state
        unchanged: independent slips, explicit or blocked ``stay`` actions,
        and rejection of a conflicting raw outcome.  It is therefore more
        informative for policy tie breaking than collision probability alone.
        """
        self.validate_state(state)
        self.validate_joint_action(joint_action)

        return sum(
            probability
            for next_state, probability in self.joint_transitions(
                state,
                joint_action,
            )
            if next_state == state
        )


    def action_risk_breakdown(
        self,
        state: State,
        joint_action: JointAction,
    ) -> dict[str, float | int]:
        """Return an interpretable breakdown of one joint action.

        Probabilities are computed from raw stochastic outcomes, before the
        collision-rejection rule merges them into the current state.  The
        categories are mutually interpretable but not all are mutually
        exclusive: ``self_loop_probability`` is the final resolved probability
        of staying in the same joint state, while vertex/edge probabilities
        identify the collision component and ``noncollision_no_motion``
        identifies raw outcomes that already equal the current state.
        """
        self.validate_state(state)
        self.validate_joint_action(joint_action)

        vertex_probability = 0.0
        edge_swap_probability = 0.0
        any_conflict_probability = 0.0
        noncollision_no_motion_probability = 0.0

        for raw_next_state, probability in self._raw_joint_transitions(
            state, joint_action
        ):
            vertex = self.vertex_conflict_count(raw_next_state) > 0
            edge = self.edge_swap_conflict_count(state, raw_next_state) > 0
            if vertex:
                vertex_probability += probability
            if edge:
                edge_swap_probability += probability
            if vertex or edge:
                any_conflict_probability += probability
            elif raw_next_state == state:
                noncollision_no_motion_probability += probability

        unfinished_stay_actions = 0
        unfinished_blocked_actions = 0
        movable_unfinished_actions = 0
        frozen_agents = 0

        for index, (position, action, goal) in enumerate(
            zip(state, joint_action, self.goals)
        ):
            if position == goal and self.config.freeze_agents_at_goal:
                frozen_agents += 1
                continue
            if action == "stay":
                unfinished_stay_actions += 1
                continue
            intended = self.move_one(index, position, action)
            if intended == position:
                unfinished_blocked_actions += 1
            else:
                movable_unfinished_actions += 1

        return {
            "self_loop_probability": self.self_loop_probability(
                state, joint_action
            ),
            "conflict_probability": any_conflict_probability,
            "vertex_conflict_probability": vertex_probability,
            "edge_swap_probability": edge_swap_probability,
            "noncollision_no_motion_probability": (
                noncollision_no_motion_probability
            ),
            "unfinished_stay_actions": unfinished_stay_actions,
            "unfinished_blocked_actions": unfinished_blocked_actions,
            "movable_unfinished_actions": movable_unfinished_actions,
            "frozen_agents": frozen_agents,
        }

    def joint_transitions(
        self,
        state: State,
        joint_action: JointAction,
    ) -> tuple[Transition, ...]:
        """
        Return every legal resolved successor and its probability.

        When collision rejection is enabled, each conflicting raw outcome is
        converted into a transition back to the current state. Duplicate
        successors are then merged. The resolved distribution is memoized.
        """
        self.validate_state(state)
        self.validate_joint_action(joint_action)

        cache_key = (
            state,
            joint_action,
        )

        cached = self._cache_lookup(
            self._resolved_transition_cache,
            cache_key,
        )

        if cached is not None:
            self._resolved_transition_cache_hits += 1
            return cached

        self._resolved_transition_cache_misses += 1

        raw_transitions = self._raw_joint_transitions(
            state,
            joint_action,
        )

        if not self.config.reject_conflicting_transitions:
            if self._cache_store(
                self._resolved_transition_cache,
                cache_key,
                raw_transitions,
                raw=False,
            ):
                self._resolved_transition_cache_writes += 1
            return raw_transitions

        merged: dict[State, float] = defaultdict(float)

        for raw_next_state, probability in raw_transitions:
            resolved_state = (
                state
                if self.has_conflict(state, raw_next_state)
                else raw_next_state
            )

            merged[resolved_state] += probability

        transitions = tuple(
            (next_state, probability)
            for next_state, probability in merged.items()
            if probability > 0.0
        )

        self._validate_probability_sum(
            transitions,
            label="Resolved joint",
        )

        if self._cache_store(
            self._resolved_transition_cache,
            cache_key,
            transitions,
            raw=False,
        ):
            self._resolved_transition_cache_writes += 1

        return transitions

    @contextmanager
    def transition_cache_writes(
        self,
        enabled: bool,
    ) -> Iterator[None]:
        """Temporarily enable or suppress new transition-cache writes.

        Cache reads remain active. Nested contexts are conservative: an inner
        context cannot re-enable writes while an outer context disabled them.
        This is used during greedy policy extraction in evaluation so only the
        action actually executed is cached afterwards.
        """
        previous = self._transition_cache_write_enabled
        self._transition_cache_write_enabled = previous and enabled
        try:
            yield
        finally:
            self._transition_cache_write_enabled = previous

    def reset_transition_cache_stats(self) -> None:
        """Reset hit/miss/write counters without clearing distributions."""
        self._raw_transition_cache_hits = 0
        self._raw_transition_cache_misses = 0
        self._raw_transition_cache_writes = 0
        self._raw_transition_cache_evictions = 0
        self._resolved_transition_cache_hits = 0
        self._resolved_transition_cache_misses = 0
        self._resolved_transition_cache_writes = 0
        self._resolved_transition_cache_evictions = 0

    def transition_cache_stats(self) -> dict[str, int | bool | None]:
        """Return cache sizes and counters for performance diagnostics."""
        return {
            "write_enabled": self._transition_cache_write_enabled,
            "max_entries_per_cache": self.config.transition_cache_max_entries,
            "raw_entries": len(self._raw_transition_cache),
            "resolved_entries": len(self._resolved_transition_cache),
            "raw_hits": self._raw_transition_cache_hits,
            "raw_misses": self._raw_transition_cache_misses,
            "raw_writes": self._raw_transition_cache_writes,
            "raw_evictions": self._raw_transition_cache_evictions,
            "resolved_hits": self._resolved_transition_cache_hits,
            "resolved_misses": self._resolved_transition_cache_misses,
            "resolved_writes": self._resolved_transition_cache_writes,
            "resolved_evictions": self._resolved_transition_cache_evictions,
        }

    def clear_transition_cache(self) -> None:
        """
        Clear memoized transition distributions.

        This is normally unnecessary because one GridMMDP object represents an
        immutable environment configuration. It is provided for diagnostics
        and memory-control experiments.
        """
        self._raw_transition_cache.clear()
        self._resolved_transition_cache.clear()

    @staticmethod
    def _validate_probability_sum(
        transitions: tuple[Transition, ...],
        *,
        label: str,
    ) -> None:
        probability_sum = sum(
            probability
            for _, probability in transitions
        )

        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise RuntimeError(
                f"{label} transition probabilities sum to "
                f"{probability_sum}, not 1"
            )

    @staticmethod
    def sample_from_transitions(
        transitions: tuple[Transition, ...],
        rng: random.Random,
    ) -> State:
        """Sample from a transition tuple that was already computed."""
        if not transitions:
            raise ValueError("transitions cannot be empty")

        roll = rng.random()
        cumulative_probability = 0.0

        for next_state, probability in transitions:
            cumulative_probability += probability
            if roll <= cumulative_probability:
                return next_state

        return transitions[-1][0]

    def sample_next(
        self,
        state: State,
        joint_action: JointAction,
        rng: random.Random,
    ) -> State:
        """Sample one actual successor after an action has been selected."""
        transitions = self.joint_transitions(state, joint_action)
        return self.sample_from_transitions(transitions, rng)

    def all_joint_actions(
        self,
    ) -> Iterator[JointAction]:
        """Generate all complete joint actions lazily."""
        return product(
            ACTIONS,
            repeat=self.n_agents,
        )

    def summary(self) -> str:
        return (
            f"MMDP map: {self.map_name}\n"
            f"Size: {self.width} x {self.height}\n"
            f"Agents: {self.n_agents}\n"
            f"Scenario: {self.instance.scenario_file.name}\n"
            f"Slip-to-stay probability: "
            f"{self.config.slip_to_stay_probability:.2f}\n"
            f"Freeze agents at goal: "
            f"{self.config.freeze_agents_at_goal}\n"
            f"Reject conflicting transitions: "
            f"{self.config.reject_conflicting_transitions}\n"
            f"Transition cache max entries per cache: "
            f"{self.config.transition_cache_max_entries}\n"
            "Transition cost: number of unfinished agents"
        )
