from __future__ import annotations

"""Small reusable components used by the two RTDP planning domains."""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Generic, TypeVar

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")


class ValueStore(ABC, Generic[StateType]):
    @abstractmethod
    def get(self, state: StateType) -> float | None: ...

    @abstractmethod
    def set(self, state: StateType, value: float) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class DictValueStore(ValueStore[StateType]):
    def __init__(self) -> None:
        self._values: dict[StateType, float] = {}

    def get(self, state: StateType) -> float | None:
        return self._values.get(state)

    def set(self, state: StateType, value: float) -> None:
        self._values[state] = value

    def clear(self) -> None:
        self._values.clear()


class SolvedTracker(ABC, Generic[StateType]):
    @abstractmethod
    def is_solved(self, state: StateType) -> bool: ...

    @abstractmethod
    def mark_solved(self, states: Iterable[StateType]) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class SetSolvedTracker(SolvedTracker[StateType]):
    def __init__(self) -> None:
        self._solved: set[StateType] = set()

    def is_solved(self, state: StateType) -> bool:
        return state in self._solved

    def mark_solved(self, states: Iterable[StateType]) -> None:
        self._solved.update(states)

    def clear(self) -> None:
        self._solved.clear()


class TieBreaker(ABC, Generic[ActionType]):
    @abstractmethod
    def choose(self, candidates: Sequence[ActionType], state_context: Any) -> ActionType: ...


class DeterministicTieBreaker(TieBreaker[ActionType]):
    """Choose reproducibly among numerically tied actions."""

    def __init__(self, seed: int, tie_type: str = "tie") -> None:
        self.seed = seed
        self.tie_type = tie_type

    def choose(self, candidates: Sequence[ActionType], state_context: Any) -> ActionType:
        if not candidates:
            raise ValueError("candidates cannot be empty")
        payload = repr((self.seed, state_context, tuple(candidates), self.tie_type)).encode()
        digest = hashlib.sha256(payload).digest()
        return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


class PlanningDomain(ABC, Generic[StateType, ActionType]):
    @abstractmethod
    def initial_state(self) -> StateType: ...

    @abstractmethod
    def is_terminal(self, state: StateType) -> bool: ...

    @abstractmethod
    def get_value(self, state: StateType) -> float: ...

    @abstractmethod
    def is_solved(self, state: StateType) -> bool: ...

    @abstractmethod
    def is_initial_state_solved(self) -> bool: ...

    @abstractmethod
    def mark_solved(self, states: set[StateType]) -> None: ...

    @abstractmethod
    def backup(self, state: StateType, *, deadline: float | None = None) -> ActionType: ...

    @abstractmethod
    def sample_next(self, state: StateType, action: ActionType) -> StateType: ...

    @abstractmethod
    def get_best_action_and_value_for_solved_check(
        self, state: StateType, *, deadline: float | None = None
    ) -> tuple[ActionType, float]: ...

    @abstractmethod
    def get_successors_for_solved_check(
        self, state: StateType, action: ActionType
    ) -> tuple[StateType, ...]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def reset_caches(self) -> None: ...

    @abstractmethod
    def run_trial(self, *, deadline: float | None = None) -> tuple[StateType, ...]: ...

    @abstractmethod
    def policy_action(self, state: Any) -> Any: ...
