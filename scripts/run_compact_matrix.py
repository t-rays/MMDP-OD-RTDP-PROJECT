from __future__ import annotations

"""Run the 12 serial conditions for one final-experiment map group."""

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

from mmdp.experiments.final_config import (
    AGENT_COUNTS,
    ALGORITHMS,
    CONDITION_WATCHDOG_SECONDS,
    FINAL_MAPS,
    FIXED_SEED,
)
from mmdp.experiments.schema import CSV_FIELDS, make_run_id, normalized_row

ROOT = Path(__file__).resolve().parents[1]


def existing_run_ids(output: Path) -> set[str]:
    if not output.exists() or output.stat().st_size == 0:
        return set()
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(
                "The existing CSV uses a different schema. Use a new output path."
            )
        return {row["run_id"] for row in reader if row.get("run_id")}


def append_row(output: Path, row: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized_row(row))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=tuple(FINAL_MAPS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    map_config = FINAL_MAPS[args.group]
    map_name = Path(map_config.folder).name
    output = args.output.resolve()
    recorded = existing_run_ids(output)
    conditions = list(itertools.product(AGENT_COUNTS, ALGORITHMS))

    print(f"Group {args.group}: {len(conditions)} conditions (serial execution)")
    print(f"Fixed seed: {FIXED_SEED}")
    completed = skipped = failed = 0

    for index, (n_agents, algorithm) in enumerate(conditions, 1):
        run_id = make_run_id(args.group, map_name, n_agents, algorithm)
        label = f"[{index}/{len(conditions)}] {map_name} | agents={n_agents} | {algorithm}"
        if run_id in recorded:
            skipped += 1
            print(f"SKIP    {label}")
            continue
        print(f"START   {label}", flush=True)
        if args.dry_run:
            continue

        with tempfile.TemporaryDirectory(prefix="mmdp_") as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/run_experiments.py"),
                "--group",
                args.group,
                "--agents",
                str(n_agents),
                "--algorithm",
                algorithm,
                "--output",
                str(result_path),
            ]
            started = time.perf_counter()
            process = subprocess.Popen(command, cwd=ROOT)
            try:
                return_code = process.wait(timeout=CONDITION_WATCHDOG_SECONDS)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return_code = None
            condition_time = time.perf_counter() - started

            base = {
                "run_id": run_id,
                "map_group": args.group,
                "map_name": map_name,
                "n_agents": n_agents,
                "algorithm": algorithm,
                "seed": FIXED_SEED,
                "condition_time_seconds": condition_time,
            }

            if timed_out:
                row = {**base, "status": "condition_timeout"}
                append_row(output, row)
                recorded.add(run_id)
                failed += 1
                print(f"TIMEOUT {label} | {condition_time:.1f}s")
                continue

            if return_code != 0 or not result_path.exists():
                row = {**base, "status": f"process_error_{return_code}"}
                append_row(output, row)
                recorded.add(run_id)
                failed += 1
                print(f"FAILED  {label} | return={return_code}")
                continue

            row = json.loads(result_path.read_text(encoding="utf-8"))
            row["condition_time_seconds"] = condition_time
            append_row(output, row)
            recorded.add(run_id)
            completed += 1
            print(
                f"DONE    {label} | stop={row['planning_stop_reason']} | "
                f"plan={float(row['planning_time_seconds']):.3f}s | "
                f"memory={float(row['planning_peak_memory_delta_mb']):.2f}MB | "
                f"success={row['evaluation_successful_episodes']}/"
                f"{row['evaluation_scheduled_episodes']} | total={condition_time:.1f}s"
            )

    print(f"\nCompleted: {completed} | skipped: {skipped} | failed/timeouts: {failed}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
