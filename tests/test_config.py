from __future__ import annotations

import pytest

from mmdp import RTDPConfig


def test_valid_config_accepts_time_limit() -> None:
    config = RTDPConfig(time_limit_seconds=5.0)
    assert config.time_limit_seconds == 5.0


def test_requires_at_least_one_stopping_mechanism() -> None:
    with pytest.raises(ValueError, match="stopping mechanism"):
        RTDPConfig(time_limit_seconds=None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_trials": 0},
        {"max_steps_per_trial": -1},
        {"step_limit_multiplier": 0.0},
        {"step_tail_probability": 0.0},
        {"time_limit_seconds": 0.0},
        {"memory_limit_mb": -5.0},
        {"epsilon": -1.0},
        {"stable_trials_required": 0},
        {"tie_ulps": 0},
    ],
)
def test_invalid_values_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RTDPConfig(**kwargs)
