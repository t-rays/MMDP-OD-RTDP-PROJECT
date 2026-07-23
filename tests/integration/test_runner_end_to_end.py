from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_single_condition_worker_writes_result(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_experiments.py"),
        "--group", "easy",
        "--agents", "1",
        "--algorithm", "baseline",
        "--output", str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    row = json.loads(output.read_text())
    assert row["status"] == "ok"
    assert row["planning_stop_reason"] == "initial_state_solved"
    assert row["evaluation_successful_episodes"] == 5
