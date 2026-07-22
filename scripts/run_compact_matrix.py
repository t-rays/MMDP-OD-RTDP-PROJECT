from __future__ import annotations

"""Run one final experiment group and write a compact resumable CSV.

The final experiment is defined in ``mmdp.experiments.final_config``.  No
external manifest is loaded, and every condition uses the same fixed seed.
Each condition still runs in a fresh process so peak-memory measurements remain
comparable and a watchdog can stop an unresponsive condition safely.
"""

import argparse
import csv
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from mmdp.experiments.final_config import (
    AGENT_COUNTS,
    ALGORITHMS,
    CONDITION_WATCHDOG_SECONDS,
    EVALUATION_EPISODES,
    EVALUATION_TIME_LIMIT_SECONDS,
    FINAL_MAPS,
    FIXED_SEED,
    PLANNING_TIME_LIMIT_SECONDS,
    TRANSITION_CACHE_MAX_ENTRIES,
)

ROOT = Path(__file__).resolve().parents[1]

COMPACT_FIELDS = [
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
    "states_examined",
    "success_rate",
    "evaluation_episodes_completed",
    "evaluation_time_seconds",
    "condition_time_seconds",
]


def _existing_ids(output: Path) -> set[str]:
    if not output.exists() or output.stat().st_size == 0:
        return set()
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COMPACT_FIELDS:
            raise ValueError(
                "The existing CSV uses an older schema. Move or delete it, or "
                "choose a new --output path before running this version."
            )
        return {
            row.get("run_id", "")
            for row in reader
            if row.get("status") == "ok"
        }


def _append_row(output: Path, row: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPACT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in COMPACT_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def _float(row: dict[str, str], key: str) -> float | str:
    value = row.get(key, "")
    if value in (None, ""):
        return ""
    try:
        return float(value)
    except ValueError:
        return ""


def _int(row: dict[str, str], key: str) -> int | str:
    value = row.get(key, "")
    if value in (None, ""):
        return ""
    try:
        return int(str(value))
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return ""


def _compact_from_full(
    full: dict[str, str],
    *,
    run_id: str,
    map_group: str,
    condition_elapsed: float,
) -> dict[str, Any]:
    states = _int(full, "planning_visited_real_states")
    if states == "":
        states = _int(full, "planning_visited_states")

    successful = _int(full, "evaluation_successful_episodes")
    failed = _int(full, "evaluation_failed_episodes")
    if isinstance(successful, int) and isinstance(failed, int):
        completed_episodes: int | str = successful + failed
    else:
        completed_episodes = ""

    return {
        "run_id": run_id,
        "map_group": map_group,
        "map_name": full.get("map_name", ""),
        "n_agents": _int(full, "n_agents"),
        "algorithm": full.get("algorithm", ""),
        "seed": FIXED_SEED,
        "status": full.get("status", "error"),
        "planning_stop_reason": full.get("planning_stop_reason", ""),
        "planning_time_seconds": _float(full, "planning_elapsed_seconds"),
        "planning_peak_memory_delta_mb": _float(
            full, "planning_peak_rss_delta_mb"
        ),
        "states_examined": states,
        "success_rate": _float(full, "evaluation_success_rate"),
        "evaluation_episodes_completed": completed_episodes,
        "evaluation_time_seconds": _float(full, "evaluation_elapsed_seconds"),
        "condition_time_seconds": condition_elapsed,
    }


def _condition_command(
    *,
    map_path: Path,
    n_agents: int,
    algorithm: str,
    scenario_number: int,
    task_offset: int,
    evaluation_max_steps: int,
    full_csv: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts/run_experiments.py"),
        str(map_path),
        "--agent-counts",
        str(n_agents),
        "--algorithms",
        algorithm,
        "--scenario-numbers",
        str(scenario_number),
        "--task-offsets",
        str(task_offset),
        "--resource-mode",
        "time_or_solved",
        "--time-limit-seconds",
        str(PLANNING_TIME_LIMIT_SECONDS),
        "--evaluation-episodes",
        str(EVALUATION_EPISODES),
        "--evaluation-time-limit-seconds",
        str(EVALUATION_TIME_LIMIT_SECONDS),
        "--evaluation-max-steps",
        str(evaluation_max_steps),
        "--transition-cache-max-entries",
        str(TRANSITION_CACHE_MAX_ENTRIES),
        "--disable-conflict-risk",
        "--disable-evaluation-diagnostics",
        "--output",
        str(full_csv),
        "--overwrite",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=tuple(FINAL_MAPS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    map_config = FINAL_MAPS[args.group]
    map_path = (ROOT / map_config.folder).resolve()
    output = args.output.resolve()
    existing = _existing_ids(output)
    conditions = list(itertools.product(AGENT_COUNTS, ALGORITHMS))

    print(
        f"Group {args.group}: {len(conditions)} conditions "
        "(1 map × 6 agent counts × 2 algorithms)",
        flush=True,
    )
    print(f"Fixed seed: {FIXED_SEED}", flush=True)
    completed = skipped = failed = 0

    for index, (n_agents, algorithm) in enumerate(conditions, 1):
        run_id = (
            f"v16|{args.group}|{map_path.name}|a{n_agents}|"
            f"seed{FIXED_SEED}|{algorithm}"
        )
        label = (
            f"[{index}/{len(conditions)}] {map_path.name} | "
            f"agents={n_agents} | {algorithm}"
        )
        if run_id in existing:
            skipped += 1
            print(f"SKIP  {label}", flush=True)
            continue
        print(f"START {label}", flush=True)

        if args.dry_run:
            continue

        with tempfile.TemporaryDirectory(prefix="mmdp_") as temp_dir:
            full_csv = Path(temp_dir) / "full.csv"
            command = _condition_command(
                map_path=map_path,
                n_agents=n_agents,
                algorithm=algorithm,
                scenario_number=map_config.scenario_number,
                task_offset=map_config.task_offset,
                evaluation_max_steps=map_config.evaluation_max_steps,
                full_csv=full_csv,
            )
            started = time.perf_counter()
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(command, cwd=ROOT, env=env)
            timed_out = False
            try:
                return_code = process.wait(timeout=CONDITION_WATCHDOG_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return_code = None
            elapsed = time.perf_counter() - started

            common_failure = {
                "run_id": run_id,
                "map_group": args.group,
                "map_name": map_path.name,
                "n_agents": n_agents,
                "algorithm": algorithm,
                "seed": FIXED_SEED,
                "condition_time_seconds": elapsed,
            }

            if timed_out:
                failed += 1
                _append_row(
                    output,
                    {**common_failure, "status": "condition_timeout"},
                )
                existing.add(run_id)
                print(f"TIMEOUT {label} | {elapsed:.1f}s", flush=True)
                continue

            if return_code != 0 or not full_csv.exists():
                failed += 1
                _append_row(
                    output,
                    {
                        **common_failure,
                        "status": f"process_error_{return_code}",
                    },
                )
                existing.add(run_id)
                print(f"FAILED {label} | return={return_code}", flush=True)
                continue

            with full_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 1:
                raise RuntimeError(f"Expected one internal row, found {len(rows)}")

            compact = _compact_from_full(
                rows[0],
                run_id=run_id,
                map_group=args.group,
                condition_elapsed=elapsed,
            )
            _append_row(output, compact)
            existing.add(run_id)
            completed += 1
            print(
                f"DONE  {label} | plan={compact['planning_time_seconds']}s | "
                f"mem={compact['planning_peak_memory_delta_mb']}MB | "
                f"states={compact['states_examined']} | "
                f"success={compact['success_rate']} | "
                f"episodes={compact['evaluation_episodes_completed']} | "
                f"total={elapsed:.1f}s",
                flush=True,
            )

    print(f"\nFinished group {args.group}", flush=True)
    print(
        f"Completed now: {completed} | skipped: {skipped} | "
        f"failed/timeouts: {failed}",
        flush=True,
    )
    print("Compact results:", output, flush=True)


if __name__ == "__main__":
    main()
