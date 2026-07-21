from __future__ import annotations

"""Deterministic paired-seed generation shared by all experiment drivers."""

import random


def seed_pairs(master_seed: int, count: int) -> list[tuple[int, int]]:
    """Derive ``count`` unique (planning_seed, evaluation_seed) pairs."""
    rng = random.Random(master_seed)
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    while len(pairs) < count:
        planning_seed = rng.randrange(0, 2**31)
        evaluation_seed = rng.randrange(0, 2**63)
        if planning_seed in used:
            continue
        used.add(planning_seed)
        pairs.append((planning_seed, evaluation_seed))
    return pairs
