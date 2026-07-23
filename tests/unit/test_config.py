from __future__ import annotations

import pytest

from mmdp import RTDPConfig


def test_valid_config() -> None:
    config = RTDPConfig(time_limit_seconds=5.0, seed=7)
    assert config.time_limit_seconds == 5.0
    assert config.seed == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_limit_seconds": 0.0},
        {"step_tail_probability": 0.0},
        {"epsilon": -1.0},
        {"relative_epsilon": -1.0},
        {"tie_ulps": 0},
    ],
)
def test_invalid_values_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RTDPConfig(**kwargs)
