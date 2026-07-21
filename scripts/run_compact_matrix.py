from __future__ import annotations

"""Run the fixed focused experiment and write one compact CSV.

Each condition runs in a fresh process for fair memory measurement. The large
internal CSV produced by the core runner is temporary and deleted immediately;
only the small final CSV is retained.
"""

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from mmdp.experiments.seeds import seed_pairs

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "easy": ROOT / "manifests/final/easy.json",
    "medium": ROOT / "manifests/final/medium.json",
    "hard": ROOT / "manifests/final/hard.json",
}

COMPACT_FIELDS = [
    "run_id",
    "map_group",
    "map_name",
    "n_agents",
    "algorithm",
    "seed_index",
    "planning_seed",
    "evaluation_seed",
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


def _append_options(command: list[str], options: dict[str, Any]) -> None:
    for name, value in options.items():
        if value is None or value is False or name == "resource_mode":
            continue
        flag = "--" + name.replace("_", "-")
        if value is True:
            command.append(flag)
        else:
            command.extend([flag, str(value)])


def _existing_ids(output: Path) -> set[str]:
    if not output.exists() or output.stat().st_size == 0:
        return set()
    with output.open(newline="", encoding="utf-8") as handle:
        return {
            row.get("run_id", "")
            for row in csv.DictReader(handle)
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
    seed_index: int,
    condition_elapsed: float,
) -> dict[str, Any]:
    states = _int(full, "planning_visited_real_states")
    if states == "":
        states = _int(full, "planning_visited_states")
    completed_episodes = _int(full, "evaluation_successful_episodes")
    failed_episodes = _int(full, "evaluation_failed_episodes")
    if isinstance(completed_episodes, int) and isinstance(failed_episodes, int):
        completed_episodes += failed_episodes
    else:
        completed_episodes = ""
    return {
        "run_id": run_id,
        "map_group": map_group,
        "map_name": full.get("map_name", ""),
        "n_agents": _int(full, "n_agents"),
        "algorithm": full.get("algorithm", ""),
        "seed_index": seed_index,
        "planning_seed": _int(full, "planning_seed"),
        "evaluation_seed": _int(full, "evaluation_seed"),
        "status": full.get("status", "error"),
        "planning_stop_reason": full.get("planning_stop_reason", ""),
        "planning_time_seconds": _float(full, "planning_elapsed_seconds"),
        "planning_peak_memory_delta_mb": _float(full, "planning_peak_rss_delta_mb"),
        "states_examined": states,
        "success_rate": _float(full, "evaluation_success_rate"),
        "evaluation_episodes_completed": completed_episodes,
        "evaluation_time_seconds": _float(full, "evaluation_elapsed_seconds"),
        "condition_time_seconds": condition_elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=tuple(MANIFESTS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFESTS[args.group].read_text(encoding="utf-8"))
    output = args.output.resolve()
    master_seed = int(manifest["master_seed"])
    resolved_seed_pairs = seed_pairs(master_seed, int(manifest["seed_count"]))
    algorithms = list(manifest["algorithms"])
    common = dict(manifest["experiments_args"])
    watchdog = float(manifest["condition_watchdog_seconds"])
    existing = _existing_ids(output)

    conditions: list[tuple[dict[str, Any], int, tuple[int, int], str]] = []
    for map_spec in manifest["maps"]:
        for n_agents, indexed_seed, algorithm in itertools.product(
            map_spec["agent_counts"], enumerate(resolved_seed_pairs), algorithms
        ):
            conditions.append((map_spec, n_agents, indexed_seed, algorithm))

    print(f"Group {args.group}: {len(conditions)} conditions (1 map × 6 agent counts × 2 seeds × 2 algorithms)", flush=True)
    completed = skipped = failed = 0

    for index, (map_spec, n_agents, indexed_seed, algorithm) in enumerate(conditions, 1):
        seed_index, (planning_seed, evaluation_seed) = indexed_seed
        map_path = (ROOT / map_spec["folder"]).resolve()
        # The "v15|" prefix is kept intentionally so existing compact CSVs
        # keep resuming correctly; it is a schema tag, not a code version.
        run_id = (
            f"v15|{args.group}|{map_path.name}|a{n_agents}|"
            f"s{seed_index}|p{planning_seed}|e{evaluation_seed}|{algorithm}"
        )
        label = f"[{index}/{len(conditions)}] {map_path.name} | agents={n_agents} | seed={seed_index+1}/2 | {algorithm}"
        if run_id in existing:
            skipped += 1
            print(f"SKIP  {label}", flush=True)
            continue
        print(f"START {label}", flush=True)

        if args.dry_run:
            continue

        with tempfile.TemporaryDirectory(prefix="mmdp_") as temp_dir:
            full_csv = Path(temp_dir) / "full.csv"
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/run_experiments.py"),
                str(map_path),
                "--agent-counts", str(n_agents),
                "--algorithms", algorithm,
                "--planning-seeds", str(planning_seed),
                "--evaluation-seeds", str(evaluation_seed),
                "--scenario-numbers", str(map_spec.get("scenario_number", 1)),
                "--task-offsets", str(map_spec.get("task_offset", 0)),
                "--resource-mode", str(common.get("resource_mode", "time_or_solved")),
                "--master-seed", str(master_seed),
                "--output", str(full_csv),
                "--overwrite",
            ]
            _append_options(command, common)
            started = time.perf_counter()
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(command, cwd=ROOT, env=env)
            timed_out = False
            try:
                return_code = process.wait(timeout=watchdog)
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

            if timed_out:
                failed += 1
                _append_row(output, {
                    "run_id": run_id,
                    "map_group": args.group,
                    "map_name": map_path.name,
                    "n_agents": n_agents,
                    "algorithm": algorithm,
                    "seed_index": seed_index,
                    "planning_seed": planning_seed,
                    "evaluation_seed": evaluation_seed,
                    "status": "condition_timeout",
                    "condition_time_seconds": elapsed,
                })
                existing.add(run_id)
                print(f"TIMEOUT {label} | {elapsed:.1f}s", flush=True)
                continue

            if return_code != 0 or not full_csv.exists():
                failed += 1
                _append_row(output, {
                    "run_id": run_id,
                    "map_group": args.group,
                    "map_name": map_path.name,
                    "n_agents": n_agents,
                    "algorithm": algorithm,
                    "seed_index": seed_index,
                    "planning_seed": planning_seed,
                    "evaluation_seed": evaluation_seed,
                    "status": f"process_error_{return_code}",
                    "condition_time_seconds": elapsed,
                })
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
                seed_index=seed_index,
                condition_elapsed=elapsed,
            )
            _append_row(output, compact)
            existing.add(run_id)
            completed += 1
            print(
                f"DONE  {label} | plan={compact['planning_time_seconds']}s | "
                f"mem={compact['planning_peak_memory_delta_mb']}MB | "
                f"states={compact['states_examined']} | success={compact['success_rate']} | "
                f"episodes={compact['evaluation_episodes_completed']} | total={elapsed:.1f}s",
                flush=True,
            )

    print("\nFinished group", args.group, flush=True)
    print(f"Completed now: {completed} | skipped: {skipped} | failed/timeouts: {failed}", flush=True)
    print("Compact results:", output, flush=True)


if __name__ == "__main__":
    main()
