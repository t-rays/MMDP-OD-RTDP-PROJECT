from __future__ import annotations

"""Create map/agent resource limits from unconstrained pilot CSV results.

The calibration is external to the planners.  Limits are derived separately
for every map and agent count, so larger instances naturally receive larger
budgets.

By default, rows are paired by scenario, task offset, and seed index.  For each
pair we take the larger resource requirement of Baseline and OD, then use the
median across paired pilot conditions.  This produces one common budget that
is not calibrated in favour of either algorithm: at least half of the paired
pilot conditions needed no more than that budget for *both* planners.

Alternative empirical quantiles and pooled calibration remain available for
sensitivity analysis.  Rounding is optional and disabled by default, so no
hidden 5-second or 64-MiB constants are introduced.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable


def parse_optional_positive_float(text: str) -> float | None:
    if text.strip().lower() in {"none", "off", "null"}:
        return None
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive or none")
    return value


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def round_up(value: float, unit: float | None) -> float:
    return value if unit is None else unit * math.ceil(value / unit)


def summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def _value(row: dict[str, str], field: str) -> float | None:
    text = row.get(field, "").strip()
    return float(text) if text else None


def _paired_observations(
    rows: Iterable[dict[str, str]],
    *,
    required_algorithms: tuple[str, ...],
) -> tuple[dict[tuple[str, int], list[float]], dict[tuple[str, int], list[float]], list[dict[str, object]]]:
    paired: dict[tuple[str, int, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["map_name"],
            int(row["n_agents"]),
            row.get("scenario_number", ""),
            row.get("task_offset", ""),
            row.get("seed_index", row.get("planning_seed", "")),
        )
        paired[key][row["algorithm"]] = row

    times: dict[tuple[str, int], list[float]] = defaultdict(list)
    memories: dict[tuple[str, int], list[float]] = defaultdict(list)
    omitted: list[dict[str, object]] = []
    for pair_key, by_algorithm in sorted(paired.items()):
        missing = [name for name in required_algorithms if name not in by_algorithm]
        if missing:
            omitted.append({"pair": pair_key, "reason": "missing_algorithms", "missing": missing})
            continue
        selected = [by_algorithm[name] for name in required_algorithms]
        pair_times = [_value(row, "planning_first_stability_elapsed_seconds") for row in selected]
        pair_memories = [_value(row, "planning_peak_rss_delta_mb") for row in selected]
        if any(value is None for value in pair_times):
            omitted.append({"pair": pair_key, "reason": "missing_stability_time"})
            continue
        if any(value is None for value in pair_memories):
            omitted.append({"pair": pair_key, "reason": "missing_memory_measurement"})
            continue
        group_key = (pair_key[0], pair_key[1])
        times[group_key].append(max(value for value in pair_times if value is not None))
        memories[group_key].append(max(value for value in pair_memories if value is not None))
    return times, memories, omitted


def _pooled_observations(
    rows: Iterable[dict[str, str]],
) -> tuple[dict[tuple[str, int], list[float]], dict[tuple[str, int], list[float]], list[dict[str, object]]]:
    times: dict[tuple[str, int], list[float]] = defaultdict(list)
    memories: dict[tuple[str, int], list[float]] = defaultdict(list)
    omitted: list[dict[str, object]] = []
    for row in rows:
        key = (row["map_name"], int(row["n_agents"]))
        time_value = _value(row, "planning_first_stability_elapsed_seconds")
        memory_value = _value(row, "planning_peak_rss_delta_mb")
        if time_value is None or memory_value is None:
            omitted.append(
                {
                    "map_name": key[0],
                    "n_agents": key[1],
                    "algorithm": row.get("algorithm"),
                    "seed_index": row.get("seed_index"),
                    "reason": "missing_measurement",
                }
            )
            continue
        times[key].append(time_value)
        memories[key].append(memory_value)
    return times, memories, omitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.50,
        help="Empirical quantile used as the common constrained budget (default: median).",
    )
    parser.add_argument(
        "--calibration-policy",
        choices=("paired_max", "pooled"),
        default="paired_max",
        help="paired_max is the fair default; pooled is retained for sensitivity analysis.",
    )
    parser.add_argument("--algorithms", nargs="+", default=["baseline", "od"])
    parser.add_argument("--source-resource-mode", default="unconstrained")
    parser.add_argument("--time-fraction", type=float, default=1.0)
    parser.add_argument("--memory-fraction", type=float, default=1.0)
    parser.add_argument(
        "--time-rounding-seconds",
        type=parse_optional_positive_float,
        default=None,
    )
    parser.add_argument(
        "--memory-rounding-mb",
        type=parse_optional_positive_float,
        default=None,
    )
    parser.add_argument(
        "--profile-name", default="paired-median-from-unconstrained-pilots"
    )
    args = parser.parse_args()

    if not 0.0 <= args.quantile <= 1.0:
        parser.error("--quantile must be in [0, 1]")
    if args.time_fraction <= 0.0 or args.memory_fraction <= 0.0:
        parser.error("resource fractions must be positive")
    required_algorithms = tuple(dict.fromkeys(args.algorithms))
    if not required_algorithms:
        parser.error("--algorithms cannot be empty")

    rows: list[dict[str, str]] = []
    with args.input_csv.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("status") != "ok":
                continue
            if row.get("resource_mode") != args.source_resource_mode:
                continue
            rows.append(row)

    if args.calibration_policy == "paired_max":
        time_groups, memory_groups, omitted = _paired_observations(
            rows, required_algorithms=required_algorithms
        )
        budget_definition = (
            "For each scenario/offset/seed pair, take the maximum requirement "
            "across the required algorithms; then take the requested empirical quantile."
        )
    else:
        time_groups, memory_groups, omitted = _pooled_observations(rows)
        budget_definition = (
            "Pool algorithm runs within each map/agent group and take the requested empirical quantile."
        )

    maps: dict[str, dict] = {}
    group_keys = sorted(set(time_groups) | set(memory_groups))
    for map_name, n_agents in group_keys:
        stability_times = time_groups.get((map_name, n_agents), [])
        peak_deltas = memory_groups.get((map_name, n_agents), [])
        if not stability_times or not peak_deltas:
            omitted.append(
                {
                    "map_name": map_name,
                    "n_agents": n_agents,
                    "reason": "no_complete_observations",
                }
            )
            continue

        time_reference = quantile(stability_times, args.quantile)
        memory_reference = quantile(peak_deltas, args.quantile)
        entry = maps.setdefault(map_name, {"agents": {}})
        entry["agents"][str(n_agents)] = {
            "time_limit_seconds": round_up(
                time_reference * args.time_fraction,
                args.time_rounding_seconds,
            ),
            "memory_limit_mb": round_up(
                memory_reference * args.memory_fraction,
                args.memory_rounding_mb,
            ),
            "source_observations": min(len(stability_times), len(peak_deltas)),
            "time_reference_seconds": time_reference,
            "memory_reference_mb": memory_reference,
            "time_observed": summary(stability_times),
            "memory_observed": summary(peak_deltas),
        }

    output = {
        "profile_name": args.profile_name,
        "calibration": {
            "source_csv": str(args.input_csv.resolve()),
            "source_resource_mode": args.source_resource_mode,
            "calibration_policy": args.calibration_policy,
            "required_algorithms": list(required_algorithms),
            "budget_definition": budget_definition,
            "quantile": args.quantile,
            "time_fraction": args.time_fraction,
            "memory_fraction": args.memory_fraction,
            "time_rounding_seconds": args.time_rounding_seconds,
            "memory_rounding_mb": args.memory_rounding_mb,
            "memory_definition": "additional planning RSS above fresh-process baseline",
        },
        "maps": maps,
        "omitted_observations": omitted,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {args.output_json.resolve()}")
    if omitted:
        print(f"Omitted {len(omitted)} incomplete pilot observations")


if __name__ == "__main__":
    main()
