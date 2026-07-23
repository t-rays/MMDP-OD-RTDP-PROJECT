from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

FIELDS = [
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
    "evaluation_failed_episodes",
    "evaluation_uncompleted_episodes",
    "evaluation_success_rate",
    "evaluation_successful_episodes",
    "evaluation_episodes_completed",
    "evaluation_scheduled_episodes",
    "evaluation_time_seconds",
    "condition_time_seconds",
]


def _write_sample_results(path: Path) -> None:
    rows = []
    map_names = {
        "easy": "empty-8-8",
        "medium": "warehouse-10-20-10-2-1",
        "hard": "room-64-64-16",
    }
    for group, map_name in map_names.items():
        for agents in range(1, 7):
            for algorithm in ("baseline", "od"):
                timeout = group == "hard" and agents == 6 and algorithm == "baseline"
                solved = group == "easy" and (algorithm == "od" or agents <= 4)
                successful = 5
                completed = 5
                if group == "hard" and agents >= 5:
                    successful = 0
                if group == "hard" and agents == 4:
                    successful = 2
                    completed = 2 if algorithm == "baseline" else 4
                if timeout:
                    rows.append(
                        {
                            "run_id": f"{group}-{agents}-{algorithm}",
                            "map_group": group,
                            "map_name": map_name,
                            "n_agents": agents,
                            "algorithm": algorithm,
                            "seed": 20260708,
                            "status": "condition_timeout",
                            "planning_stop_reason": "",
                            "planning_time_seconds": "",
                            "planning_peak_memory_delta_mb": "",
                            "evaluation_failed_episodes": "",
                            "evaluation_uncompleted_episodes": "",
                            "evaluation_success_rate": "",
                            "evaluation_successful_episodes": "",
                            "evaluation_episodes_completed": "",
                            "evaluation_scheduled_episodes": 5,
                            "evaluation_time_seconds": "",
                            "condition_time_seconds": 75.0,
                        }
                    )
                    continue

                rows.append(
                    {
                        "run_id": f"{group}-{agents}-{algorithm}",
                        "map_group": group,
                        "map_name": map_name,
                        "n_agents": agents,
                        "algorithm": algorithm,
                        "seed": 20260708,
                        "status": "ok",
                        "planning_stop_reason": (
                            "initial_state_solved" if solved else "time_limit"
                        ),
                        "planning_time_seconds": (
                            agents * (0.02 if algorithm == "od" else 0.03)
                            if solved
                            else 60.0
                        ),
                        "planning_peak_memory_delta_mb": agents * (
                            2.0 if algorithm == "od" else 5.0
                        ),
                        "evaluation_failed_episodes": completed - successful,
                        "evaluation_uncompleted_episodes": 5 - completed,
                        "evaluation_success_rate": successful / 5,
                        "evaluation_successful_episodes": successful,
                        "evaluation_episodes_completed": completed,
                        "evaluation_scheduled_episodes": 5,
                        "evaluation_time_seconds": 1.0,
                        "condition_time_seconds": 2.0,
                    }
                )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_analysis_matches_final_report_output_set(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    output_dir = tmp_path / "outputs"
    _write_sample_results(csv_path)

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "analyze_compact_results.py"),
        str(csv_path),
        "--group",
        "hard",
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr

    png_names = sorted(path.name for path in output_dir.glob("*.png"))
    assert png_names == ["figure_1c_room_peak_planning_memory.png"]
    assert not any("time" in name for name in png_names)
    assert not any("state" in name for name in png_names)

    planning = pd.read_csv(output_dir / "table_1_room_planning_outcomes.csv")
    assert list(planning.columns) == [
        "Agents",
        "Baseline RTDP",
        "OD-RTDP",
        "Comparison",
    ]
    assert len(planning) == 6

    evaluation = pd.read_csv(output_dir / "table_2_budgeted_policy_evaluation.csv")
    row_4 = evaluation[evaluation["Agents"] == 4].iloc[0]
    assert row_4["Room B"] == "2/5"
    assert row_4["Room OD"] == "2/5"
    row_6 = evaluation[evaluation["Agents"] == 6].iloc[0]
    assert row_6["Room B"] == "0/5 (timeout)"


def test_medium_analysis_creates_warehouse_planning_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    output_dir = tmp_path / "outputs"
    _write_sample_results(csv_path)

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "analyze_compact_results.py"),
        str(csv_path),
        "--group",
        "medium",
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr

    planning_path = output_dir / "table_1_warehouse_planning_outcomes.csv"
    assert planning_path.exists()
    planning = pd.read_csv(planning_path)
    assert len(planning) == 6
    assert list(planning.columns) == [
        "Agents",
        "Baseline RTDP",
        "OD-RTDP",
        "Comparison",
    ]
