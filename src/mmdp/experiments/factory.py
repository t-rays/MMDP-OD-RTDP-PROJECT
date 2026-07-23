from __future__ import annotations

"""Construct the environment, planners, and evaluation configuration."""

from mmdp.domain.grid_mmdp import GridMMDP, MMDPConfig
from mmdp.domain.heuristic import ShortestPathHeuristic
from mmdp.evaluation import EvaluationConfig
from mmdp.experiments.final_config import (
    EPSILON,
    EVALUATION_EPISODES,
    EVALUATION_TIME_LIMIT_SECONDS,
    FIXED_SEED,
    PLANNING_TIME_LIMIT_SECONDS,
    RELATIVE_EPSILON,
    SLIP_PROBABILITY,
    STEP_CAP_FAMILYWISE_ERROR,
    TIE_ULPS,
    TRANSITION_CACHE_MAX_ENTRIES,
)
from mmdp.planning.baseline_rtdp import BaselineDomain
from mmdp.planning.components import DeterministicTieBreaker, DictValueStore, SetSolvedTracker
from mmdp.planning.config import RTDPConfig
from mmdp.planning.od_rtdp import OperatorDecompositionDomain
from mmdp.planning.planner import RTDPPlanner


def create_mdp(instance) -> GridMMDP:
    return GridMMDP(
        instance,
        MMDPConfig(
            slip_to_stay_probability=SLIP_PROBABILITY,
            transition_cache_max_entries=TRANSITION_CACHE_MAX_ENTRIES,
        ),
    )


def create_planning_config(n_agents: int) -> RTDPConfig:
    return RTDPConfig(
        time_limit_seconds=PLANNING_TIME_LIMIT_SECONDS,
        step_tail_probability=(
            STEP_CAP_FAMILYWISE_ERROR / (EVALUATION_EPISODES * n_agents)
        ),
        epsilon=EPSILON,
        relative_epsilon=RELATIVE_EPSILON,
        tie_ulps=TIE_ULPS,
        seed=FIXED_SEED,
    )


def create_planner(
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
            tie_breaker=DeterministicTieBreaker(FIXED_SEED),
        )
    elif algorithm == "od":
        domain = OperatorDecompositionDomain(
            mdp=mdp,
            heuristic=heuristic,
            config=config,
            value_store=DictValueStore(),
            solved_tracker=SetSolvedTracker(),
            tie_breaker=DeterministicTieBreaker(FIXED_SEED),
            joint_tie_breaker=DeterministicTieBreaker(
                FIXED_SEED,
                tie_type="complete-od-policy",
            ),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
    return RTDPPlanner(domain=domain, config=config)


def create_evaluation_config(max_steps: int) -> EvaluationConfig:
    return EvaluationConfig(
        episodes=EVALUATION_EPISODES,
        seed=FIXED_SEED,
        max_steps_per_episode=max_steps,
        time_limit_seconds=EVALUATION_TIME_LIMIT_SECONDS,
    )
