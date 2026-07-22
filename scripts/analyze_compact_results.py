from __future__ import annotations

"""Create exactly three figures from the final compact experiment CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LABELS = {"baseline": "Baseline RTDP", "od": "OD-RTDP"}
METRICS = [
    "planning_time_seconds",
    "planning_peak_memory_delta_mb",
    "states_examined",
    "success_rate",
]


def _load(csv_path: Path, group: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"map_group", "n_agents", "algorithm", "status", *METRICS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df[(df["map_group"] == group) & (df["status"] == "ok")].copy()
    for column in ["n_agents", *METRICS]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df.empty:
        raise ValueError(f"No completed rows for group {group}")
    duplicated = df.duplicated(["n_agents", "algorithm"], keep=False)
    if duplicated.any():
        duplicates = df.loc[duplicated, ["n_agents", "algorithm"]]
        raise ValueError(
            "The final experiment expects one fixed-seed row per condition; "
            f"duplicates found: {duplicates.to_dict('records')}"
        )
    return df.sort_values(["n_agents", "algorithm"])


def _plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
    *,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algorithm in ("baseline", "od"):
        part = summary[summary["algorithm"] == algorithm].dropna(subset=[metric])
        ax.plot(part["n_agents"], part[metric], marker="o", label=LABELS[algorithm])
    ax.set_xlabel("Number of agents")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(summary["n_agents"].dropna().astype(int).unique()))
    ax.grid(True, alpha=0.3)
    if log_y and not summary[metric].dropna().empty and (summary[metric].dropna() > 0).all():
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_states_and_success(summary: pd.DataFrame, output: Path) -> None:
    """Plot grouped bars so equal Baseline/OD values never hide each other."""
    agents = sorted(summary["n_agents"].dropna().astype(int).unique())
    x = list(range(len(agents)))
    width = 0.36

    def values_for(algorithm: str, metric: str) -> list[float]:
        part = summary[summary["algorithm"] == algorithm].set_index("n_agents")[metric]
        return [float(part.get(agent, float("nan"))) for agent in agents]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].bar(
        [value - width / 2 for value in x],
        values_for("baseline", "states_examined"),
        width,
        label=LABELS["baseline"],
    )
    axes[0].bar(
        [value + width / 2 for value in x],
        values_for("od", "states_examined"),
        width,
        label=LABELS["od"],
    )
    axes[0].set_ylabel("Real states examined")
    axes[0].set_title("Search size and policy success")
    axes[0].grid(True, axis="y", alpha=0.3)
    positive = summary["states_examined"].dropna()
    if not positive.empty and (positive > 0).all():
        axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].bar(
        [value - width / 2 for value in x],
        values_for("baseline", "success_rate"),
        width,
        label=LABELS["baseline"],
    )
    axes[1].bar(
        [value + width / 2 for value in x],
        values_for("od", "success_rate"),
        width,
        label=LABELS["od"],
    )
    axes[1].set_xlabel("Number of agents")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x, agents)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    agents = sorted(df["n_agents"].dropna().astype(int).unique())
    result = pd.DataFrame({"n_agents": agents}).set_index("n_agents")
    for metric in METRICS:
        for algorithm in ("baseline", "od"):
            values = (
                df[df["algorithm"] == algorithm]
                .set_index("n_agents")[metric]
                .reindex(agents)
            )
            result[f"{metric}_{algorithm}"] = values
        result[f"{metric}_od_minus_baseline"] = (
            result[f"{metric}_od"] - result[f"{metric}_baseline"]
        )
    return result.reset_index().round(4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--group", required=True, choices=("easy", "medium", "hard"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for file in out.glob("*.png"):
        file.unlink()

    summary = _load(args.csv_file.resolve(), args.group)
    _plot_metric(
        summary,
        "planning_time_seconds",
        "Planning time (seconds)",
        "Planning time by number of agents",
        out / "01_time.png",
    )
    _plot_metric(
        summary,
        "planning_peak_memory_delta_mb",
        "Peak memory increase (MB)",
        "Planning memory by number of agents",
        out / "02_memory.png",
    )
    _plot_states_and_success(summary, out / "03_states_and_success.png")

    print(comparison_table(summary).to_string(index=False))
    print("Created exactly three graphs in", out)


if __name__ == "__main__":
    main()
