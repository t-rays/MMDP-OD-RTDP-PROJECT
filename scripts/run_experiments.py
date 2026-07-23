from __future__ import annotations

"""Run one condition from the fixed final experiment and write one JSON row."""

import argparse
import json
from pathlib import Path

from mmdp.experiments.final_config import ALGORITHMS, FINAL_MAPS
from mmdp.experiments.runner import run_condition

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=tuple(FINAL_MAPS))
    parser.add_argument("--agents", required=True, type=int, choices=range(1, 7))
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    map_config = FINAL_MAPS[args.group]
    row = run_condition(
        map_group=args.group,
        map_folder=(ROOT / map_config.folder).resolve(),
        scenario_number=map_config.scenario_number,
        task_offset=map_config.task_offset,
        evaluation_max_steps=map_config.evaluation_max_steps,
        n_agents=args.agents,
        algorithm=args.algorithm,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row), encoding="utf-8")


if __name__ == "__main__":
    main()
