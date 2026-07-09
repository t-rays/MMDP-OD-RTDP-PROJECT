from __future__ import annotations

"""Run every experimental condition in a fresh Python subprocess.

Fresh processes are required for meaningful RSS comparisons: Python may retain
memory between sequential runs even after objects are released.  This wrapper
reads a JSON manifest, expands map/agent/scenario/seed/resource combinations,
and invokes ``experiments.py`` once per row.

The optional watchdog is only a technical fail-safe.  A watchdog termination is
written to a separate JSONL log and is never labelled as convergence.
"""

import argparse
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


def _seed_pairs(master_seed: int, count: int) -> list[tuple[int, int]]:
    rng = random.Random(master_seed)
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    while len(pairs) < count:
        planning = rng.randrange(0, 2**31)
        evaluation = rng.randrange(0, 2**63)
        if planning in used:
            continue
        used.add(planning)
        pairs.append((planning, evaluation))
    return pairs


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _append_options(command: list[str], options: dict[str, Any]) -> None:
    for name, value in options.items():
        if value is False or value is None:
            continue
        option = _flag(name)
        if value is True:
            command.append(option)
        elif isinstance(value, list):
            command.append(option)
            command.extend(str(item) for item in value)
        else:
            command.extend((option, str(value)))


def _resolve(project_root: Path, text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--watchdog-seconds",
        type=float,
        default=None,
        help="Technical subprocess timeout; omitted means no external timeout.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    project_root = Path(__file__).resolve().parents[1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiments_path = project_root / "src" / "experiments.py"
    if not experiments_path.is_file():
        raise FileNotFoundError(experiments_path)

    output = (
        args.output.resolve()
        if args.output is not None
        else _resolve(
            project_root,
            manifest.get("output", "results/final_matrix_v7.csv"),
        ).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    watchdog_log = output.with_suffix(output.suffix + ".watchdog.jsonl")

    seed_count = int(manifest.get("seed_count", 10))
    master_seed = int(manifest.get("master_seed", 20260708))
    pairs = _seed_pairs(master_seed, seed_count)
    algorithms = manifest.get("algorithms", ["baseline", "od"])
    modes = manifest.get(
        "resource_modes", ["unconstrained", "time", "memory", "time_memory"]
    )
    common_options = dict(manifest.get("experiments_args", {}))
    profile = manifest.get("resource_profile")
    if profile is not None:
        common_options["resource_profile"] = str(
            _resolve(project_root, profile).resolve()
        )

    conditions: list[tuple[Any, ...]] = []
    for map_spec in manifest["maps"]:
        map_path = _resolve(project_root, map_spec["folder"]).resolve()
        agent_counts = map_spec.get("agent_counts", [2, 3])
        scenarios = map_spec.get("scenario_numbers", [1])
        offsets = map_spec.get("task_offsets", [0])
        map_modes = map_spec.get("resource_modes", modes)
        for condition in itertools.product(
            agent_counts,
            scenarios,
            offsets,
            pairs,
            algorithms,
            map_modes,
        ):
            conditions.append((map_path, *condition))

    print(f"Expanded conditions: {len(conditions)}")
    wrote_first = output.exists() and output.stat().st_size > 0 and not args.overwrite
    failures = 0
    for index, condition in enumerate(conditions, start=1):
        (
            map_path,
            n_agents,
            scenario,
            offset,
            seed_pair,
            algorithm,
            mode,
        ) = condition
        planning_seed, evaluation_seed = seed_pair
        command = [
            args.python,
            str(experiments_path),
            str(map_path),
            "--agent-counts",
            str(n_agents),
            "--algorithms",
            str(algorithm),
            "--planning-seeds",
            str(planning_seed),
            "--evaluation-seeds",
            str(evaluation_seed),
            "--scenario-numbers",
            str(scenario),
            "--task-offsets",
            str(offset),
            "--resource-mode",
            str(mode),
            "--master-seed",
            str(master_seed),
            "--output",
            str(output),
        ]
        _append_options(command, common_options)
        if not wrote_first and args.overwrite:
            command.append("--overwrite")

        label = (
            f"[{index}/{len(conditions)}] {map_path.name} agents={n_agents} "
            f"scenario={scenario} offset={offset} pseed={planning_seed} "
            f"eseed={evaluation_seed} {algorithm} mode={mode}"
        )
        print(label, flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            continue

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                check=False,
                timeout=args.watchdog_seconds,
            )
            return_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            return_code = None
            timed_out = True

        wrote_first = output.exists() and output.stat().st_size > 0
        if timed_out or return_code != 0:
            failures += 1
            record = {
                "condition_index": index,
                "label": label,
                "command": command,
                "timed_out": timed_out,
                "watchdog_seconds": args.watchdog_seconds,
                "return_code": return_code,
                "elapsed_seconds": time.perf_counter() - started,
            }
            with watchdog_log.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"FAILED: {record}", flush=True)
            if args.fail_fast:
                raise SystemExit(1)

    print(f"Finished. Results: {output}")
    print(f"Subprocess/watchdog failures: {failures}")
    if failures:
        print(f"Failure log: {watchdog_log}")


if __name__ == "__main__":
    main()
