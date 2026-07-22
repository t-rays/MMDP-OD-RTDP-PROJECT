from __future__ import annotations

"""
Slip-aware shortest-path heuristics and structured tie guidance.

The numeric heuristic remains an admissible lower bound for the cooperative
Stochastic Shortest Path objective.

For one isolated agent at shortest-path distance d, with movement success
probability q = 1 - slip, the expected remaining number of charged steps is:

    d / q

because every successful unit of shortest-path progress requires a geometric
number of attempts with mean 1 / q.

Baseline RTDP therefore uses:

    h(state) = sum_i d_i(state) / q

while ignoring collisions and inter-agent waiting. Ignoring those interactions
can only make the estimate smaller than the real cooperative cost.

For an OD state (state, prefix), actions already fixed in the prefix are
accounted for using a collision-safe one-agent lower bound.  A fixed action's
isolated expected future value is capped by the agent's current value:

    min(H_i(current), E[H_i(after isolated action)])

This cap is necessary because collision rejection can replace a harmful raw
move by the current state.  Without the cap, a prefix that fixes a move away
from the goal can be overestimated and cease to be admissible.  Agents that
have not yet received an action are assumed to choose their individually best
safe one-step action.  The empty-prefix OD heuristic remains exactly equal to
the Baseline heuristic.

Shortest-path distance alone deliberately treats every equally short route as
identical. To avoid large arbitrary tie sets, this module also exposes
structured secondary guidance keys. They are used only after Bellman/Q values
are tied, so they do not replace or perturb the primary optimization objective.
The guidance prefers, in order:

- lower probability of remaining in exactly the same joint state;
- lower collision risk;
- fewer deterministic/partial conflicts;
- fewer moves into another agent's goal or current cell;
- genuine shortest-path progress over waiting or blocked moves;
- branches with more remaining shortest paths;
- positions with more future shortest-path choices and local mobility.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Sequence

from mmdp.domain.grid_mmdp import (
    ACTIONS,
    Action,
    GridMMDP,
    JointAction,
    State,
)
from mmdp.domain.map_creator import Position


DistanceTable = dict[Position, int]
LogPathCountTable = dict[Position, float]
GuidanceKey = tuple[float | int, ...]


def build_distance_table(
    mdp: GridMMDP,
    goal: Position,
) -> DistanceTable:
    """Compute exact four-directional shortest-path distances to one goal."""
    if goal not in mdp.free_cells:
        raise ValueError(
            f"Cannot build a distance table for blocked goal {goal}"
        )

    distances: DistanceTable = {goal: 0}
    queue: deque[Position] = deque([goal])

    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1

        for neighbor in mdp.instance.grid_map.neighbors4(current):
            if neighbor in distances:
                continue

            distances[neighbor] = next_distance
            queue.append(neighbor)

    return distances


def _logsumexp(values: list[float]) -> float:
    """Stable logarithm of a sum of exponentials."""
    if not values:
        return -math.inf

    maximum = max(values)
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )


def build_shortest_path_log_count_table(
    mdp: GridMMDP,
    distances: DistanceTable,
    goal: Position,
) -> LogPathCountTable:
    """
    Store log(number of shortest paths) from every reachable cell to the goal.

    Using logarithms avoids enormous integers on large open maps. The value is
    used only as secondary guidance: a branch that preserves more shortest
    routes is preferred when the Bellman values are tied.
    """
    log_counts: LogPathCountTable = {goal: 0.0}  # log(1)

    for position, distance in sorted(
        distances.items(),
        key=lambda item: item[1],
    ):
        if distance == 0:
            continue

        predecessor_logs = [
            log_counts[neighbor]
            for neighbor in mdp.instance.grid_map.neighbors4(position)
            if distances.get(neighbor) == distance - 1
        ]

        if not predecessor_logs:
            raise RuntimeError(
                f"Reachable cell {position} has no shortest-path predecessor"
            )

        log_counts[position] = _logsumexp(predecessor_logs)

    return log_counts


@dataclass
class ShortestPathHeuristic:
    """Obstacle-aware, slip-aware shortest-path lower bound and tie guidance."""

    mdp: GridMMDP

    distance_tables: tuple[DistanceTable, ...] = field(
        init=False,
        repr=False,
    )
    shortest_path_log_tables: tuple[LogPathCountTable, ...] = field(
        init=False,
        repr=False,
    )
    movement_success_probability: float = field(
        init=False,
    )
    _state_value_cache: dict[State, float] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _distance_summary_cache: dict[State, tuple[float, ...]] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        self.movement_success_probability = (
            1.0 - self.mdp.config.slip_to_stay_probability
        )

        if self.movement_success_probability <= 0.0:
            raise ValueError(
                "ShortestPathHeuristic requires a positive movement success "
                "probability"
            )

        self.distance_tables = tuple(
            build_distance_table(self.mdp, goal)
            for goal in self.mdp.goals
        )

        self.shortest_path_log_tables = tuple(
            build_shortest_path_log_count_table(
                self.mdp,
                distances,
                goal,
            )
            for distances, goal in zip(
                self.distance_tables,
                self.mdp.goals,
            )
        )

        self._validate_initial_positions()

    def _validate_initial_positions(self) -> None:
        for agent_index, start in enumerate(self.mdp.starts):
            if start not in self.distance_tables[agent_index]:
                raise ValueError(
                    f"Agent {agent_index} cannot reach its goal "
                    f"{self.mdp.goals[agent_index]} from start {start}"
                )

    def agent_distance(
        self,
        agent_index: int,
        position: Position,
    ) -> float:
        """Return one agent's deterministic shortest-path distance."""
        if not 0 <= agent_index < self.mdp.n_agents:
            raise IndexError(
                f"Agent index {agent_index} is outside the valid range"
            )

        if position not in self.mdp.free_cells:
            raise ValueError(
                f"Position {position} is not a free map cell"
            )

        return float(
            self.distance_tables[agent_index].get(
                position,
                math.inf,
            )
        )

    def stochastic_agent_distance(
        self,
        agent_index: int,
        position: Position,
    ) -> float:
        """Return the isolated-agent slip-aware lower bound d / q."""
        distance = self.agent_distance(agent_index, position)

        if math.isinf(distance):
            return math.inf

        return distance / self.movement_success_probability

    def __call__(
        self,
        state: State,
    ) -> float:
        """
        Return the Baseline lower bound:

            h(state) = sum_i shortest_distance_i / movement_success_probability

        The map, goals, and slip probability are immutable, so the heuristic is
        immutable as well.  Evaluation revisits the same successor states many
        thousands of times while comparing joint actions; memoizing the result
        avoids recomputing the same sum and repeated validation.
        """
        cached = self._state_value_cache.get(state)
        if cached is not None:
            return cached

        self.mdp.validate_state(state)
        total = 0.0

        for agent_index, position in enumerate(state):
            distance = float(
                self.distance_tables[agent_index].get(position, math.inf)
            )
            if math.isinf(distance):
                self._state_value_cache[state] = math.inf
                return math.inf
            # Preserve the original operation order exactly: each agent's
            # distance is divided before the contributions are accumulated.
            total += distance / self.movement_success_probability

        result = total
        self._state_value_cache[state] = result
        return result

    def deterministic_result(
        self,
        agent_index: int,
        position: Position,
        action: Action,
    ) -> Position:
        """Return the intended result before slip is sampled."""
        return self.mdp.move_one(
            agent_index,
            position,
            action,
        )

    def distance_after_action(
        self,
        agent_index: int,
        position: Position,
        action: Action,
    ) -> float:
        """Return deterministic shortest-path distance after one intended move."""
        next_position = self.deterministic_result(
            agent_index,
            position,
            action,
        )

        return self.agent_distance(
            agent_index,
            next_position,
        )

    def expected_stochastic_distance_after_action(
        self,
        agent_index: int,
        position: Position,
        action: Action,
    ) -> float:
        """
        Expected isolated-agent lower bound after executing one stochastic move.

        The immediate real-step cost is not included here; od_value adds it once
        for every unfinished agent before summing these future contributions.
        """
        expected = 0.0

        for next_position, probability in self.mdp.individual_outcomes(
            agent_index,
            position,
            action,
        ):
            future = self.stochastic_agent_distance(
                agent_index,
                next_position,
            )

            if math.isinf(future):
                return math.inf

            expected += probability * future

        return expected

    def safe_expected_stochastic_distance_after_action(
        self,
        agent_index: int,
        position: Position,
        action: Action,
    ) -> float:
        """Return a collision-safe future lower bound for one fixed action.

        Collision rejection maps a conflicting raw outcome back to the current
        joint state.  For a move that increases an agent's isolated distance,
        the raw one-agent expectation can therefore be larger than the true
        cooperative successor value.  Capping it by the current isolated value
        preserves admissibility for every possible collision pattern.
        """
        current = self.stochastic_agent_distance(
            agent_index,
            position,
        )
        expected = self.expected_stochastic_distance_after_action(
            agent_index,
            position,
            action,
        )
        return min(current, expected)

    def best_expected_stochastic_distance(
        self,
        agent_index: int,
        position: Position,
    ) -> float:
        """Best collision-safe isolated future lower bound after one action."""
        return min(
            self.safe_expected_stochastic_distance_after_action(
                agent_index,
                position,
                action,
            )
            for action in ACTIONS
        )

    def shortest_path_log_count(
        self,
        agent_index: int,
        position: Position,
    ) -> float:
        """Return log(number of shortest routes) from position to the goal."""
        return self.shortest_path_log_tables[agent_index].get(
            position,
            -math.inf,
        )

    def progress_action_count(
        self,
        agent_index: int,
        position: Position,
    ) -> int:
        """Count actions that reduce deterministic shortest-path distance by 1."""
        current_distance = self.agent_distance(
            agent_index,
            position,
        )

        if current_distance <= 0.0 or math.isinf(current_distance):
            return 0

        return sum(
            self.distance_after_action(
                agent_index,
                position,
                action,
            )
            == current_distance - 1.0
            for action in ACTIONS
        )

    def local_degree(
        self,
        position: Position,
    ) -> int:
        """Return the number of legal four-neighbor movements from a cell."""
        return sum(
            1
            for _ in self.mdp.instance.grid_map.neighbors4(position)
        )

    def individual_stay_probability(
        self,
        agent_index: int,
        position: Position,
        action: Action,
    ) -> float:
        """Return the chance that one agent remains on its current cell."""
        return sum(
            probability
            for next_position, probability in self.mdp.individual_outcomes(
                agent_index,
                position,
                action,
            )
            if next_position == position
        )

    def minimum_individual_stay_probability(
        self,
        agent_index: int,
        position: Position,
    ) -> float:
        """Best isolated-agent chance of leaving the current cell."""
        return min(
            self.individual_stay_probability(
                agent_index,
                position,
                action,
            )
            for action in ACTIONS
        )

    def optimistic_prefix_self_loop_probability(
        self,
        state: State,
        prefix: Sequence[Action],
    ) -> float:
        """Optimistic all-agents-stay probability for an incomplete prefix.

        Fixed prefix actions use their exact isolated stay probabilities.
        Every unselected agent is assumed to choose the action with the lowest
        isolated stay probability.  Collision rejection is deliberately not
        added here; partial collision indicators remain separate guidance
        fields.  The value is used only to order Bellman-tied OD operators.
        """
        self.mdp.validate_state(state)

        if len(prefix) > self.mdp.n_agents:
            raise ValueError(
                f"Prefix contains {len(prefix)} actions, "
                f"but the problem has only {self.mdp.n_agents} agents"
            )

        probability = 1.0

        for agent_index, position in enumerate(state):
            if agent_index < len(prefix):
                stay_probability = self.individual_stay_probability(
                    agent_index,
                    position,
                    prefix[agent_index],
                )
            else:
                stay_probability = self.minimum_individual_stay_probability(
                    agent_index,
                    position,
                )

            probability *= stay_probability

        return probability

    def od_value(
        self,
        state: State,
        prefix: Sequence[Action],
    ) -> float:
        """
        Return the slip-aware lower bound for an OD state.

        A future real transition charges one unit per unfinished agent. For a
        fixed prefix action we add a collision-safe future lower bound:

            min(current isolated value, isolated expected value after action)

        Unselected agents receive their best safe individual one-step action.
        This remains optimistic even when a harmful raw move would be rejected
        by a collision and converted back to the current state.
        """
        self.mdp.validate_state(state)

        if len(prefix) > self.mdp.n_agents:
            raise ValueError(
                f"Prefix contains {len(prefix)} actions, "
                f"but the problem has only {self.mdp.n_agents} agents"
            )

        invalid_actions = [
            action
            for action in prefix
            if action not in ACTIONS
        ]

        if invalid_actions:
            raise ValueError(
                f"Prefix contains unknown actions: {invalid_actions}"
            )

        if self.mdp.is_terminal(state):
            return 0.0

        estimate = float(
            self.mdp.unfinished_agent_count(state)
        )

        for agent_index, position in enumerate(state):
            if position == self.mdp.goals[agent_index]:
                continue

            if agent_index < len(prefix):
                future = self.safe_expected_stochastic_distance_after_action(
                    agent_index,
                    position,
                    prefix[agent_index],
                )
            else:
                future = self.best_expected_stochastic_distance(
                    agent_index,
                    position,
                )

            if math.isinf(future):
                return math.inf

            estimate += future

        return estimate

    def _intended_joint_state(
        self,
        state: State,
        joint_action: JointAction,
    ) -> State:
        return tuple(
            self.deterministic_result(
                agent_index,
                position,
                action,
            )
            for agent_index, (position, action) in enumerate(
                zip(state, joint_action)
            )
        )

    def _foreign_goal_occupancy_count(
        self,
        intended_state: State,
    ) -> int:
        count = 0

        for agent_index, position in enumerate(intended_state):
            for other_index, other_goal in enumerate(self.mdp.goals):
                if other_index != agent_index and position == other_goal:
                    count += 1

        return count

    def joint_action_guidance_key(
        self,
        state: State,
        joint_action: JointAction,
    ) -> GuidanceKey:
        """
        Lexicographic secondary key for Q-tied complete joint actions.

        Smaller tuples are preferred. This method never changes the primary
        Q-value and therefore only resolves genuine numerical ties.
        """
        self.mdp.validate_state(state)
        self.mdp.validate_joint_action(joint_action)

        intended_state = self._intended_joint_state(
            state,
            joint_action,
        )

        self_loop_probability = self.mdp.self_loop_probability(
            state,
            joint_action,
        )

        conflict_probability = self.mdp.conflict_probability(
            state,
            joint_action,
        )

        deterministic_conflicts = (
            self.mdp.vertex_conflict_count(intended_state)
            + self.mdp.edge_swap_conflict_count(state, intended_state)
        )

        foreign_goal_occupancy = self._foreign_goal_occupancy_count(
            intended_state
        )

        nonprogress_count = 0
        stationary_count = 0
        total_log_path_count = 0.0
        total_progress_options = 0
        total_degree = 0

        for agent_index, (position, next_position) in enumerate(
            zip(state, intended_state)
        ):
            if position == self.mdp.goals[agent_index]:
                continue

            current_distance = self.agent_distance(
                agent_index,
                position,
            )
            next_distance = self.agent_distance(
                agent_index,
                next_position,
            )

            if next_distance >= current_distance:
                nonprogress_count += 1

            if next_position == position:
                stationary_count += 1

            total_log_path_count += self.shortest_path_log_count(
                agent_index,
                next_position,
            )
            total_progress_options += self.progress_action_count(
                agent_index,
                next_position,
            )
            total_degree += self.local_degree(next_position)

        return (
            round(self_loop_probability, 12),
            round(conflict_probability, 12),
            deterministic_conflicts,
            foreign_goal_occupancy,
            nonprogress_count,
            stationary_count,
            round(-total_log_path_count, 12),
            -total_progress_options,
            -total_degree,
        )

    def od_operator_guidance_key(
        self,
        state: State,
        prefix: Sequence[Action],
        action: Action,
    ) -> GuidanceKey:
        """
        Secondary key for tied OD operators.

        A complete prefix receives the same joint-action guidance as Baseline.
        For an incomplete prefix, the key evaluates the next agent's progress,
        shortest-path diversity, mobility, and conflicts with already fixed
        actions or still-occupied cells.
        """
        self.mdp.validate_state(state)

        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action!r}")

        extended_prefix = tuple(prefix) + (action,)

        if len(extended_prefix) > self.mdp.n_agents:
            raise ValueError("OD prefix is longer than the number of agents")

        if len(extended_prefix) == self.mdp.n_agents:
            return self.joint_action_guidance_key(
                state,
                extended_prefix,
            )

        optimistic_self_loop_probability = (
            self.optimistic_prefix_self_loop_probability(
                state,
                extended_prefix,
            )
        )

        current_agent = len(prefix)
        current_position = state[current_agent]
        intended_position = self.deterministic_result(
            current_agent,
            current_position,
            action,
        )

        fixed_intended_positions = tuple(
            self.deterministic_result(
                agent_index,
                state[agent_index],
                fixed_action,
            )
            for agent_index, fixed_action in enumerate(extended_prefix)
        )

        partial_vertex_conflicts = self.mdp.vertex_conflict_count(
            fixed_intended_positions
        )
        partial_edge_swaps = self.mdp.edge_swap_conflict_count(
            state[: len(extended_prefix)],
            fixed_intended_positions,
        )

        unselected_current_positions = state[len(extended_prefix) :]
        occupancy_risk = sum(
            intended_position == other_position
            for other_position in unselected_current_positions
        )

        foreign_goal_occupancy = sum(
            intended_position == other_goal
            for other_index, other_goal in enumerate(self.mdp.goals)
            if other_index != current_agent
        )

        current_distance = self.agent_distance(
            current_agent,
            current_position,
        )
        next_distance = self.agent_distance(
            current_agent,
            intended_position,
        )

        nonprogress = int(
            current_position != self.mdp.goals[current_agent]
            and next_distance >= current_distance
        )
        stationary = int(
            current_position != self.mdp.goals[current_agent]
            and intended_position == current_position
        )

        return (
            round(optimistic_self_loop_probability, 12),
            partial_vertex_conflicts + partial_edge_swaps,
            occupancy_risk,
            foreign_goal_occupancy,
            nonprogress,
            stationary,
            round(
                -self.shortest_path_log_count(
                    current_agent,
                    intended_position,
                ),
                12,
            ),
            -self.progress_action_count(
                current_agent,
                intended_position,
            ),
            -self.local_degree(intended_position),
        )

    def distance_summary(
        self,
        state: State,
    ) -> tuple[float, ...]:
        """Return deterministic shortest-path distances for limits and logs."""
        cached = self._distance_summary_cache.get(state)
        if cached is not None:
            return cached

        self.mdp.validate_state(state)
        result = tuple(
            float(self.distance_tables[agent_index].get(position, math.inf))
            for agent_index, position in enumerate(state)
        )
        self._distance_summary_cache[state] = result
        return result
