from __future__ import annotations

"""Fixed configuration for the experiment."""

from dataclasses import dataclass

FIXED_SEED = 20260708
ALGORITHMS = ("baseline", "od")
AGENT_COUNTS = (1, 2, 3, 4, 5, 6)
SLIP_PROBABILITY = 0.20
PLANNING_TIME_LIMIT_SECONDS = 60.0
EVALUATION_EPISODES = 5
EVALUATION_TIME_LIMIT_SECONDS = 8.0
CONDITION_WATCHDOG_SECONDS = 75.0
TRANSITION_CACHE_MAX_ENTRIES = 100_000
STEP_CAP_FAMILYWISE_ERROR = 0.01
EPSILON = 1e-8
RELATIVE_EPSILON = 1e-6
TIE_ULPS = 8


@dataclass(frozen=True)
class FinalMapConfig:
    folder: str
    scenario_number: int
    task_offset: int
    evaluation_max_steps: int


FINAL_MAPS = {
    "easy": FinalMapConfig(
        folder="maps/empty-8-8",
        scenario_number=1,
        task_offset=0,
        evaluation_max_steps=80,
    ),
    "medium": FinalMapConfig(
        folder="maps/warehouse-10-20-10-2-1",
        scenario_number=1,
        task_offset=0,
        evaluation_max_steps=160,
    ),
    "hard": FinalMapConfig(
        folder="maps/room-64-64-16",
        scenario_number=1,
        task_offset=0,
        evaluation_max_steps=260,
    ),
}
