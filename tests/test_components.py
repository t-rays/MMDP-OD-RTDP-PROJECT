from __future__ import annotations

import pytest

from mmdp import DeterministicTieBreaker, DictValueStore, SetSolvedTracker


def test_deterministic_tie_breaker_is_repeatable() -> None:
    breaker = DeterministicTieBreaker(seed=42)
    candidates = ["north", "south", "east", "west"]
    first = breaker.choose(candidates, state_context=(1, 2))
    second = breaker.choose(candidates, state_context=(1, 2))
    assert first == second
    assert first in candidates


def test_deterministic_tie_breaker_depends_on_seed_and_context() -> None:
    candidates = list(range(100))
    by_seed = {
        seed: DeterministicTieBreaker(seed=seed).choose(candidates, "ctx")
        for seed in range(20)
    }
    assert len(set(by_seed.values())) > 1

    breaker = DeterministicTieBreaker(seed=0)
    by_context = {breaker.choose(candidates, context) for context in range(20)}
    assert len(by_context) > 1


def test_tie_breaker_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError):
        DeterministicTieBreaker(seed=0).choose([], "ctx")


def test_dict_value_store_roundtrip() -> None:
    store: DictValueStore[str] = DictValueStore()
    assert store.get("a") is None
    store.set("a", 1.5)
    assert store.get("a") == 1.5
    assert len(store) == 1
    assert list(store.states()) == ["a"]
    store.clear()
    assert len(store) == 0


def test_set_solved_tracker_roundtrip() -> None:
    tracker: SetSolvedTracker[str] = SetSolvedTracker()
    assert not tracker.is_solved("a")
    tracker.mark_solved(["a", "b"])
    assert tracker.is_solved("a")
    assert len(tracker) == 2
    assert set(tracker.states()) == {"a", "b"}
    tracker.clear()
    assert len(tracker) == 0
