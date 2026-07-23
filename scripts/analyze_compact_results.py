from __future__ import annotations

"""Create the figures and tables used to inspect the final experiment.

The analysis produces one peak-memory figure and one planning-outcomes table
for each map, plus one combined budgeted policy-evaluation table.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LABELS = {"baseline": "Baseline RTDP", "od": "OD-RTDP"}
GROUP_LABELS = {
    "easy": "Open grid",
    "medium": "Warehouse",
    "hard": "Room map",
}
FIGURE_SUFFIX = {
    "easy": "1a_open_grid",
    "medium": "1b_warehouse",
    "hard": "1c_room",
}
PLANNING_TABLE_FILES = {
    "easy": "table_1_open_grid_planning_outcomes.csv",
    "medium": "table_1_warehouse_planning_outcomes.csv",
    "hard": "table_1_room_planning_outcomes.csv",
}
REQUIRED_COLUMNS = {
    "run_id",
    "map_group",
    "n_agents",
    "algorithm",
    "status",
    "planning_stop_reason",
    "planning_time_seconds",
    "planning_peak_memory_delta_mb",
    "evaluation_successful_episodes",
    "evaluation_episodes_completed",
    "evaluation_scheduled_episodes",
}
NUMERIC_COLUMNS = {
    "n_agents",
    "planning_time_seconds",
    "planning_peak_memory_delta_mb",
    "evaluation_successful_episodes",
    "evaluation_episodes_completed",
    "evaluation_scheduled_episodes",
}


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load every outcome, including time limits and condition timeouts."""
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    duplicated = df.duplicated(
        ["map_group", "n_agents", "algorithm"], keep=False
    )
    if duplicated.any():
        duplicate_rows = df.loc[
            duplicated,
            ["run_id", "map_group", "n_agents", "algorithm", "status"],
        ]
        raise ValueError(
            "Expected one fixed-seed row per condition. Use a clean CSV path. Duplicates: "
            f"{duplicate_rows.to_dict('records')}"
        )
    return df.sort_values(["map_group", "n_agents", "algorithm"])


def group_results(df: pd.DataFrame, group: str) -> pd.DataFrame:
    part = df[df["map_group"] == group].copy()
    if part.empty:
        raise ValueError(f"No rows found for group {group}")
    return part.sort_values(["n_agents", "algorithm"])


def _condition_row(
    group_df: pd.DataFrame,
    n_agents: int,
    algorithm: str,
) -> pd.Series | None:
    rows = group_df[
        (group_df["n_agents"] == n_agents)
        & (group_df["algorithm"] == algorithm)
    ]
    if rows.empty:
        return None
    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for agents={n_agents}, algorithm={algorithm}"
        )
    return rows.iloc[0]


def plot_memory(group_df: pd.DataFrame, group: str, output: Path) -> None:
    """Create the report-style peak-memory figure for one map."""
    agents = np.arange(1, 7)
    x = np.arange(len(agents))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.55))
    plotted = []

    for offset, algorithm in [(-width / 2, "baseline"), (width / 2, "od")]:
        values: list[float] = []
        rows: list[pd.Series | None] = []
        for agent in agents:
            row = _condition_row(group_df, int(agent), algorithm)
            rows.append(row)
            if (
                row is not None
                and row["status"] == "ok"
                and pd.notna(row["planning_peak_memory_delta_mb"])
            ):
                values.append(max(float(row["planning_peak_memory_delta_mb"]), 0.01))
            else:
                values.append(np.nan)

        bars = ax.bar(x + offset, values, width, label=LABELS[algorithm])
        plotted.append(bars)

        for index, (bar, row) in enumerate(zip(bars, rows)):
            if row is None:
                continue
            if (
                row["status"] != "ok"
                or pd.isna(row["planning_peak_memory_delta_mb"])
            ):
                marker_y = 0.13 if group == "hard" else 0.055
                ax.plot(
                    x[index] + offset,
                    marker_y,
                    marker="x",
                    markersize=9,
                    markeredgewidth=2,
                )
                ax.annotate(
                    "timeout\n(no result)",
                    (x[index] + offset, marker_y),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            elif row["planning_stop_reason"] == "time_limit":
                bar.set_hatch("//")

    ax.set_yscale("log")
    ax.set_xticks(x, agents)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Peak additional planning memory (MB, log scale)")
    ax.set_title(f"Peak planning memory - {GROUP_LABELS[group]}")
    ax.grid(True, axis="y", alpha=0.3)

    if group == "easy":
        ax.set_ylim(0.05, 800)
    elif group == "medium":
        ax.set_ylim(0.05, 900)
    else:
        ax.set_ylim(0.1, 900)

    handles = [
        plotted[0],
        plotted[1],
        Patch(facecolor="none", hatch="//", label="60 s planning time limit reached"),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            label="condition timeout",
        ),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _planning_cell(row: pd.Series | None) -> str:
    if row is None:
        return "NA"
    if row["status"] != "ok" or pd.isna(row["planning_time_seconds"]):
        return "NA - condition timeout"
    seconds = float(row["planning_time_seconds"])
    if row["planning_stop_reason"] == "initial_state_solved":
        return f"{seconds:.4f} s - solved"
    if row["planning_stop_reason"] == "time_limit":
        return f"{seconds:.4f} s - time limit"
    reason = str(row["planning_stop_reason"] or "stopped")
    return f"{seconds:.4f} s - {reason}"


def _planning_comparison(
    baseline: pd.Series | None,
    od: pd.Series | None,
) -> str:
    if baseline is None or od is None:
        return "Incomplete condition pair"

    baseline_solved = (
        baseline["status"] == "ok"
        and baseline["planning_stop_reason"] == "initial_state_solved"
        and pd.notna(baseline["planning_time_seconds"])
    )
    od_solved = (
        od["status"] == "ok"
        and od["planning_stop_reason"] == "initial_state_solved"
        and pd.notna(od["planning_time_seconds"])
    )

    if baseline_solved and od_solved:
        baseline_time = float(baseline["planning_time_seconds"])
        od_time = float(od["planning_time_seconds"])
        if np.isclose(baseline_time, od_time):
            return "Similar time"
        if baseline_time < od_time:
            return f"Baseline {od_time / baseline_time:.1f}x faster"
        return f"OD {baseline_time / od_time:.1f}x faster"
    if baseline_solved and not od_solved:
        return "Only Baseline converged"
    if od_solved and not baseline_solved:
        return "Only OD converged"
    if baseline["status"] != "ok" or od["status"] != "ok":
        return "At least one condition timed out"
    return "Neither planner converged"


def planning_outcomes_table(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Build a planning-outcomes table for one map group."""
    group_df = df[df["map_group"] == group]
    if group_df.empty:
        raise ValueError(f"No rows found for group {group}")

    rows: list[dict[str, str | int]] = []
    for agent in range(1, 7):
        baseline = _condition_row(group_df, agent, "baseline")
        od = _condition_row(group_df, agent, "od")
        rows.append(
            {
                "Agents": agent,
                "Baseline RTDP": _planning_cell(baseline),
                "OD-RTDP": _planning_cell(od),
                "Comparison": _planning_comparison(baseline, od),
            }
        )
    return pd.DataFrame(rows)


def _budgeted_success_cell(
    df: pd.DataFrame,
    group: str,
    n_agents: int,
    algorithm: str,
) -> str:
    group_df = df[df["map_group"] == group]
    row = _condition_row(group_df, n_agents, algorithm)
    if row is None:
        return "NA"

    scheduled = row["evaluation_scheduled_episodes"]
    denominator = int(scheduled) if pd.notna(scheduled) else 5
    if row["status"] != "ok":
        return f"0/{denominator} (timeout)"

    successful = row["evaluation_successful_episodes"]
    numerator = int(successful) if pd.notna(successful) else 0
    return f"{numerator}/{denominator}"


def budgeted_evaluation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the report's fixed-denominator policy-evaluation table."""
    rows: list[dict[str, str | int]] = []
    for agent in range(1, 7):
        rows.append(
            {
                "Agents": agent,
                "Open B": _budgeted_success_cell(df, "easy", agent, "baseline"),
                "Open OD": _budgeted_success_cell(df, "easy", agent, "od"),
                "Warehouse B": _budgeted_success_cell(
                    df, "medium", agent, "baseline"
                ),
                "Warehouse OD": _budgeted_success_cell(df, "medium", agent, "od"),
                "Room B": _budgeted_success_cell(df, "hard", agent, "baseline"),
                "Room OD": _budgeted_success_cell(df, "hard", agent, "od"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--group", required=True, choices=tuple(GROUP_LABELS))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.csv_file.resolve())
    group_df = group_results(df, args.group)

    figure_path = output_dir / (
        f"figure_{FIGURE_SUFFIX[args.group]}_peak_planning_memory.png"
    )
    plot_memory(group_df, args.group, figure_path)

    planning_table_path = output_dir / PLANNING_TABLE_FILES[args.group]
    table_2_path = output_dir / "table_2_budgeted_policy_evaluation.csv"
    planning_outcomes_table(df, args.group).to_csv(
        planning_table_path, index=False
    )
    budgeted_evaluation_table(df).to_csv(table_2_path, index=False)

    print(f"Created memory figure: {figure_path}")
    print(f"Created planning table: {planning_table_path}")
    print(f"Created evaluation table: {table_2_path}")


if __name__ == "__main__":
    main()
