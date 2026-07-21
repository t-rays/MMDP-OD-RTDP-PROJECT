from __future__ import annotations

"""External resource profiles for time/memory-constrained experiments."""

import json
from pathlib import Path
from typing import Any


RESOURCE_MODES = (
    "custom",
    "unconstrained",
    "time",
    "time_or_solved",
    "memory",
    "time_memory",
    "time_memory_or_solved",
)


def load_profile(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    profile_path = Path(path)
    return json.loads(profile_path.read_text(encoding="utf-8"))


def resolve_resource_limits(
    profile: dict[str, Any],
    *,
    map_name: str,
    n_agents: int,
) -> tuple[float | None, float | None]:
    """Resolve exact map/agent limits, then map defaults, then global defaults."""
    map_section = profile.get("maps", {}).get(map_name, {})
    agent_section = map_section.get("agents", {}).get(str(n_agents), {})
    defaults = profile.get("defaults", {})
    time_limit = agent_section.get(
        "time_limit_seconds",
        map_section.get("time_limit_seconds", defaults.get("time_limit_seconds")),
    )
    memory_limit = agent_section.get(
        "memory_limit_mb",
        map_section.get("memory_limit_mb", defaults.get("memory_limit_mb")),
    )
    return (
        None if time_limit is None else float(time_limit),
        None if memory_limit is None else float(memory_limit),
    )
