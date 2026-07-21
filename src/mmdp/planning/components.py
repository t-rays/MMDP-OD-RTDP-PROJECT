from __future__ import annotations

"""Injectable components used by the generic RTDP planner.

The planner (``planner.RTDPPlanner``) is written against the abstract
interfaces defined here.  Concrete algorithms (Baseline RTDP, OD-RTDP)
implement ``PlanningDomain`` -- usually through ``domain_base.RTDPDomainBase``
-- and receive a ``ValueStore``, ``SolvedTracker`` and ``TieBreaker`` by
dependency injection.
"""

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Iterable, Iterator, Sequence, TypeVar

if TYPE_CHECKING:
    from mmdp.resource_monitor import ResourceMonitor
    from mmdp.planning.results import TrialResult

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")


class ValueStore(ABC, Generic[StateType]):
    """A component that stores and retrieves expected costs (V-values)."""

    @abstractmethod
    def get(self, state: StateType) -> float | None:
        ...

    @abstractmethod
    def set(self, state: StateType, value: float) -> None:
        ...

    @abstractmethod
    def states(self) -> Iterator[StateType]:
        """Iterate over every state with a stored value."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class DictValueStore(ValueStore[StateType]):
    """Default implementation of a ValueStore using a dictionary."""

    def __init__(self) -> None:
        self._values: dict[StateType, float] = {}

    def get(self, state: StateType) -> float | None:
        return self._values.get(state)

    def set(self, state: StateType, value: float) -> None:
        self._values[state] = value

    def states(self) -> Iterator[StateType]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def clear(self) -> None:
        self._values.clear()


class SolvedTracker(ABC, Generic[StateType]):
    """A component that tracks which states are fully solved (LRTDP)."""

    @abstractmethod
    def is_solved(self, state: StateType) -> bool:
        ...

    @abstractmethod
    def mark_solved(self, states: Iterable[StateType]) -> None:
        ...

    @abstractmethod
    def states(self) -> Iterator[StateType]:
        """Iterate over every state marked as solved."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class SetSolvedTracker(SolvedTracker[StateType]):
    """Default implementation of a SolvedTracker using a set."""

    def __init__(self) -> None:
        self._solved: set[StateType] = set()

    def is_solved(self, state: StateType) -> bool:
        return state in self._solved

    def mark_solved(self, states: Iterable[StateType]) -> None:
        self._solved.update(states)

    def states(self) -> Iterator[StateType]:
        return iter(self._solved)

    def __len__(self) -> int:
        return len(self._solved)

    def clear(self) -> None:
        self._solved.clear()


class TieBreaker(ABC, Generic[ActionType]):
    """A generic strategy for breaking ties among equally good actions."""

    @abstractmethod
    def choose(self, candidates: Sequence[ActionType], state_context: Any) -> ActionType:
        ...


class DeterministicTieBreaker(TieBreaker[ActionType]):
    """Breaks ties deterministically using a hash of the seed, context, and candidates."""

    def __init__(self, seed: int, tie_type: str = "tie") -> None:
        self.seed = seed
        self.tie_type = tie_type

    def choose(self, candidates: Sequence[ActionType], state_context: Any) -> ActionType:
        if not candidates:
            raise ValueError("candidates cannot be empty")
        payload = repr(
            (self.seed, state_context, tuple(candidates), self.tie_type)
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        index = int.from_bytes(digest[:8], byteorder="big") % len(candidates)
        return candidates[index]


class PlanningDomain(ABC, Generic[StateType, ActionType]):
    """
    The domain interface defining the state and action spaces for the planner,
    as well as transition rules and backup logic.

    Beyond the abstract search interface below, concrete domains also expose
    greedy policy extraction (``policy_action`` / ``policy_action_with_info``)
    over *real* environment states; the planner forwards those calls for
    evaluation.
    """

    # Set by the planner for the duration of a solve so that deadline checks
    # inside trials can also enforce the memory limit.
    _resource_monitor: "ResourceMonitor | None" = None

    def attach_resource_monitor(self, monitor: "ResourceMonitor") -> None:
        self._resource_monitor = monitor

    def detach_resource_monitor(self) -> None:
        self._resource_monitor = None

    @abstractmethod
    def initial_state(self) -> StateType:
        """Return the starting state for trials."""
        ...

    @abstractmethod
    def is_terminal(self, state: StateType) -> bool:
        """Return True if the state is an absorbing goal state."""
        ...

    @abstractmethod
    def get_value(self, state: StateType) -> float:
        """
        Return the value of a state. Should query the injected ValueStore
        and fall back to an injected Heuristic if unseen.
        """
        ...

    @abstractmethod
    def is_solved(self, state: StateType) -> bool:
        """
        Check if a state is solved. Typically consults the SolvedTracker and
        handles terminal state implicitly.
        """
        ...

    @abstractmethod
    def is_initial_state_solved(self) -> bool:
        """Return True if the root initial state is fully solved."""
        ...

    @abstractmethod
    def mark_solved(self, states: set[StateType]) -> None:
        """Mark a set of states as solved in the underlying SolvedTracker."""
        ...

    @abstractmethod
    def backup(
        self, state: StateType, *, deadline: float | None = None
    ) -> tuple[ActionType, float, float]:
        """
        Evaluate all available actions, select the best (using TieBreaker),
        update the ValueStore, and return (best_action, residual, scaled_residual).
        """
        ...

    @abstractmethod
    def sample_next(self, state: StateType, action: ActionType) -> StateType:
        """Sample a successor state for the forward simulation phase."""
        ...

    @abstractmethod
    def get_best_action_and_value_for_solved_check(
        self, state: StateType, *, deadline: float | None = None
    ) -> tuple[ActionType, float]:
        """Return the best action and its expected cost for the LRTDP check."""
        ...

    @abstractmethod
    def get_successors_for_solved_check(
        self, state: StateType, action: ActionType
    ) -> tuple[StateType, ...]:
        """Return all possible stochastic successors for a given state-action pair."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the domain components (value store, trackers, caches)."""
        ...

    @abstractmethod
    def reset_caches(self) -> None:
        """Reset only transient policy caches."""
        ...

    @abstractmethod
    def run_trial(self, *, deadline: float | None = None) -> "TrialResult[StateType]":
        """Run a single forward simulation trial and return the result."""
        ...

    @abstractmethod
    def build_result_kwargs(self) -> dict[str, Any]:
        """Return the domain-specific planning metrics for the result dataclass."""
        ...
