from __future__ import annotations

import pytest

from mmdp.planning.components import DeterministicTieBreaker, DictValueStore, SetSolvedTracker


def test_deterministic_tie_breaker_is_repeatable() -> None:
    breaker = DeterministicTieBreaker(seed=42)
    candidates = ["north", "south", "east", "west"]
    assert breaker.choose(candidates, (1, 2)) == breaker.choose(candidates, (1, 2))


def test_tie_breaker_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError):
        DeterministicTieBreaker(seed=0).choose([], "ctx")


def test_value_store_roundtrip() -> None:
    store: DictValueStore[str] = DictValueStore()
    assert store.get("a") is None
    store.set("a", 1.5)
    assert store.get("a") == 1.5
    store.clear()
    assert store.get("a") is None


def test_solved_tracker_roundtrip() -> None:
    tracker: SetSolvedTracker[str] = SetSolvedTracker()
    tracker.mark_solved(["a", "b"])
    assert tracker.is_solved("a")
    tracker.clear()
    assert not tracker.is_solved("a")
