from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mmdp.experiments.final_config import (
    AGENT_COUNTS,
    ALGORITHMS,
    FINAL_MAPS,
    FIXED_SEED,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_final_configuration_uses_one_fixed_seed_and_no_manifests() -> None:
    assert FIXED_SEED == 20260708
    assert AGENT_COUNTS == (1, 2, 3, 4, 5, 6)
    assert ALGORITHMS == ("baseline", "od")
    assert set(FINAL_MAPS) == {"easy", "medium", "hard"}
    assert not (REPO_ROOT / "manifests").exists()


def test_compact_dry_run_contains_twelve_conditions(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_compact_matrix.py"),
        "--group",
        "easy",
        "--output",
        str(tmp_path / "results.csv"),
        "--dry-run",
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
    assert "12 conditions" in completed.stdout
    assert f"Fixed seed: {FIXED_SEED}" in completed.stdout
    assert completed.stdout.count("START") == 12
