from __future__ import annotations

"""Budgeted evaluation of the fixed policy produced by a planner."""

from dataclasses import dataclass
import random
import time
from typing import Protocol

from mmdp.domain.grid_mmdp import GridMMDP, JointAction, State


class PolicyPlanner(Protocol):
    mdp: GridMMDP
    def policy_action(self, state: State) -> JointAction: ...


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int
    seed: int
    max_steps_per_episode: int
    time_limit_seconds: float

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive")
        if self.time_limit_seconds <= 0.0:
            raise ValueError("time_limit_seconds must be positive")


@dataclass(frozen=True)
class EvaluationSummary:
    scheduled_episodes: int
    completed_episodes: int
    successful_episodes: int
    failed_episodes: int
    uncompleted_episodes: int
    success_rate: float
    elapsed_seconds: float


def _episode_succeeds(
    mdp: GridMMDP,
    planner: PolicyPlanner,
    *,
    episode_seed: int,
    max_steps: int,
) -> bool:
    rng = random.Random(episode_seed)
    state = mdp.initial_state()

    for _ in range(max_steps):
        if mdp.is_terminal(state):
            return True

        # Policy extraction may reuse transitions computed during planning, but
        # only the selected action is added to the cache during evaluation.
        with mdp.transition_cache_writes(False):
            action = planner.policy_action(state)
        transitions = mdp.joint_transitions(state, action)

        # A fixed deterministic policy cannot escape a guaranteed self-loop.
        if len(transitions) == 1 and transitions[0][0] == state:
            return False

        state = mdp.sample_from_transitions(transitions, rng)

    return mdp.is_terminal(state)


def evaluate_policy(
    mdp: GridMMDP,
    planner: PolicyPlanner,
    config: EvaluationConfig,
) -> EvaluationSummary:
    if planner.mdp is not mdp:
        raise ValueError("The planner belongs to a different GridMMDP")

    seed_rng = random.Random(config.seed)
    episode_seeds = [seed_rng.randrange(0, 2**63) for _ in range(config.episodes)]
    started_at = time.perf_counter()
    deadline = started_at + config.time_limit_seconds
    completed = 0
    successful = 0

    for episode_seed in episode_seeds:
        if completed > 0 and time.perf_counter() >= deadline:
            break
        successful += int(
            _episode_succeeds(
                mdp,
                planner,
                episode_seed=episode_seed,
                max_steps=config.max_steps_per_episode,
            )
        )
        completed += 1

    failed = completed - successful
    uncompleted = config.episodes - completed
    return EvaluationSummary(
        scheduled_episodes=config.episodes,
        completed_episodes=completed,
        successful_episodes=successful,
        failed_episodes=failed,
        uncompleted_episodes=uncompleted,
        success_rate=successful / config.episodes,
        elapsed_seconds=time.perf_counter() - started_at,
    )
