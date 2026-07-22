from __future__ import annotations

"""Assemble configured planners and configuration dataclasses for one run."""

import argparse
import json
from pathlib import Path
from typing import Any

from mmdp.planning.baseline_rtdp import BaselineDomain
from mmdp.planning.components import (
    DeterministicTieBreaker,
    DictValueStore,
    SetSolvedTracker,
)
from mmdp.planning.config import RTDPConfig
from mmdp.evaluation import EvaluationConfig
from mmdp.domain.grid_mmdp import GridMMDP, MMDPConfig
from mmdp.domain.heuristic import ShortestPathHeuristic
from mmdp.planning.od_rtdp import OperatorDecompositionDomain
from mmdp.planning.planner import RTDPPlanner
from mmdp.planning.results import ODRTDPPlanningResult, RTDPPlanningResult


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
    """Load an optional JSON file containing time and memory limits."""
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
    """Resolve exact map/agent limits, then map and global defaults."""
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


ALGORITHMS = ("baseline", "od")


def create_planner(
    *,
    algorithm: str,
    mdp: GridMMDP,
    heuristic: ShortestPathHeuristic,
    config: RTDPConfig,
) -> RTDPPlanner:
    if algorithm == "baseline":
        domain = BaselineDomain(
            mdp=mdp,
            heuristic=heuristic,
            config=config,
            value_store=DictValueStore(),
            solved_tracker=SetSolvedTracker(),
            tie_breaker=DeterministicTieBreaker(seed=config.seed),
        )
        return RTDPPlanner(
            domain=domain,
            config=config,
            result_builder=RTDPPlanningResult,
        )

    if algorithm == "od":
        domain = OperatorDecompositionDomain(
            mdp=mdp,
            heuristic=heuristic,
            config=config,
            value_store=DictValueStore(),
            solved_tracker=SetSolvedTracker(),
            tie_breaker=DeterministicTieBreaker(seed=config.seed),
            joint_tie_breaker=DeterministicTieBreaker(
                seed=config.seed, tie_type="complete-od-policy"
            ),
        )
        return RTDPPlanner(
            domain=domain,
            config=config,
            result_builder=ODRTDPPlanningResult,
        )

    raise ValueError(f"Unknown algorithm: {algorithm!r}")


def planning_config_from_args(
    args: argparse.Namespace,
    *,
    map_name: str,
    n_agents: int,
    resource_profile: dict[str, Any],
) -> RTDPConfig:
    profile_time, profile_memory = resolve_resource_limits(
        resource_profile, map_name=map_name, n_agents=n_agents
    )

    mode = args.resource_mode
    if mode == "unconstrained":
        time_limit = None
        memory_limit = None
        stop_when_stable = False
        stop_when_solved = True
    elif mode == "time":
        time_limit = (
            args.time_limit_seconds
            if args.time_limit_seconds is not None
            else profile_time
        )
        memory_limit = None
        stop_when_stable = False
        stop_when_solved = False
    elif mode == "time_or_solved":
        time_limit = (
            args.time_limit_seconds
            if args.time_limit_seconds is not None
            else profile_time
        )
        memory_limit = None
        stop_when_stable = False
        stop_when_solved = True
    elif mode == "memory":
        time_limit = None
        memory_limit = (
            args.memory_limit_mb
            if args.memory_limit_mb is not None
            else profile_memory
        )
        stop_when_stable = False
        stop_when_solved = True
    elif mode == "time_memory":
        time_limit = (
            args.time_limit_seconds
            if args.time_limit_seconds is not None
            else profile_time
        )
        memory_limit = (
            args.memory_limit_mb
            if args.memory_limit_mb is not None
            else profile_memory
        )
        stop_when_stable = False
        stop_when_solved = False
    elif mode == "time_memory_or_solved":
        time_limit = (
            args.time_limit_seconds
            if args.time_limit_seconds is not None
            else profile_time
        )
        memory_limit = (
            args.memory_limit_mb
            if args.memory_limit_mb is not None
            else profile_memory
        )
        stop_when_stable = False
        stop_when_solved = True
    else:
        time_limit = args.time_limit_seconds
        memory_limit = args.memory_limit_mb
        stop_when_stable = args.stop_when_stable
        stop_when_solved = args.stop_when_solved

    if mode in {"time", "time_or_solved", "time_memory", "time_memory_or_solved"} and time_limit is None:
        raise ValueError(
            f"No time limit configured for {map_name}, agents={n_agents}"
        )
    if mode in {"memory", "time_memory", "time_memory_or_solved"} and memory_limit is None:
        raise ValueError(
            f"No memory limit configured for {map_name}, agents={n_agents}"
        )

    step_tail_probability = (
        args.step_tail_probability
        if args.step_tail_probability is not None
        else args.step_cap_familywise_error
        / (args.evaluation_episodes * n_agents)
    )

    return RTDPConfig(
        max_trials=args.max_trials,
        max_steps_per_trial=args.max_steps_per_trial,
        step_limit_multiplier=args.step_limit_multiplier,
        step_tail_probability=step_tail_probability,
        time_limit_seconds=time_limit,
        memory_limit_mb=memory_limit,
        epsilon=args.epsilon,
        relative_epsilon=args.relative_epsilon,
        stable_trials_required=args.stable_trials_required,
        stop_when_stable=stop_when_stable,
        stop_when_solved=stop_when_solved,
        require_goal_for_stability=True,
        tie_tolerance=args.tie_tolerance,
        tie_ulps=args.tie_ulps,
        seed=args.seed,
    )


def evaluation_config_from_args(
    args: argparse.Namespace,
) -> EvaluationConfig:
    return EvaluationConfig(
        episodes=args.evaluation_episodes,
        seed=args.seed,
        max_steps_per_episode=args.evaluation_max_steps,
        measure_conflict_risk=not args.disable_conflict_risk,
        randomize_greedy_ties=args.randomize_evaluation_ties,
        cache_only_executed_actions=not args.cache_all_evaluation_transitions,
        collect_diagnostics=not args.disable_evaluation_diagnostics,
        time_limit_seconds=args.evaluation_time_limit_seconds,
    )


def mdp_config_from_args(args: argparse.Namespace) -> MMDPConfig:
    return MMDPConfig(
        slip_to_stay_probability=args.slip,
        freeze_agents_at_goal=True,
        reject_conflicting_transitions=True,
        transition_cache_max_entries=args.transition_cache_max_entries,
    )
