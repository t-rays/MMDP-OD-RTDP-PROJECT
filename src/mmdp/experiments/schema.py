from __future__ import annotations

"""Compact CSV schema for experiment results."""

IMPLEMENTATION_VERSION = "mmdp-rtdp-v1"

CSV_FIELDS = (
    "run_id",
    "map_group",
    "map_name",
    "n_agents",
    "algorithm",
    "seed",
    "status",
    "planning_stop_reason",
    "planning_time_seconds",
    "planning_peak_memory_delta_mb",
    "evaluation_successful_episodes",
    "evaluation_failed_episodes",
    "evaluation_episodes_completed",
    "evaluation_uncompleted_episodes",
    "evaluation_scheduled_episodes",
    "evaluation_success_rate",
    "evaluation_time_seconds",
    "condition_time_seconds",
)


def make_run_id(map_group: str, map_name: str, n_agents: int, algorithm: str) -> str:
    return (
        f"{IMPLEMENTATION_VERSION}|{map_group}|{map_name}|"
        f"a{n_agents}|seed20260708|{algorithm}"
    )


def normalized_row(row: dict) -> dict:
    return {field: row.get(field, "") for field in CSV_FIELDS}
