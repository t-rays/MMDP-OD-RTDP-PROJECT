from __future__ import annotations

from pathlib import Path

import pytest

from mmdp import GridMMDP, MMDPConfig, ShortestPathHeuristic, create_map_instance

REPO_ROOT = Path(__file__).resolve().parents[1]
EASY_MAP = REPO_ROOT / "maps" / "empty-8-8"


@pytest.fixture()
def easy_mdp() -> GridMMDP:
    instance = create_map_instance(EASY_MAP, 2, scenario_number=1, task_offset=0)
    return GridMMDP(
        instance,
        MMDPConfig(
            slip_to_stay_probability=0.2,
            transition_cache_max_entries=100_000,
        ),
    )


@pytest.fixture()
def easy_heuristic(easy_mdp: GridMMDP) -> ShortestPathHeuristic:
    return ShortestPathHeuristic(easy_mdp)
