from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_experiments_cli_writes_ok_rows(tmp_path: Path) -> None:
    output_csv = tmp_path / "results.csv"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_experiments.py"),
        str(REPO_ROOT / "maps" / "empty-8-8"),
        "--agent-counts", "2",
        "--planning-seeds", "7",
        "--evaluation-seeds", "11",
        "--resource-mode", "time_or_solved",
        "--time-limit-seconds", "15",
        "--evaluation-episodes", "3",
        "--disable-evaluation-diagnostics",
        "--disable-conflict-risk",
        "--output", str(output_csv),
        "--overwrite",
    ]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180
    )
    assert completed.returncode == 0, completed.stderr

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    by_algorithm = {row["algorithm"]: row for row in rows}
    assert set(by_algorithm) == {"baseline", "od"}
    for row in rows:
        assert row["status"] == "ok"
        assert row["planning_stop_reason"] == "initial_state_solved"
        assert int(row["planning_bellman_backups"]) > 0
        assert int(row["planning_total_real_steps"]) > 0
        assert float(row["evaluation_success_rate"]) == 1.0
    assert int(by_algorithm["baseline"]["planning_visited_states"]) > 0
    assert int(by_algorithm["od"]["planning_visited_od_states"]) > 0
