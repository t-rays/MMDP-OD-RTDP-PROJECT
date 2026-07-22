# MMDP with Operator Decomposition

This project compares Baseline RTDP and OD-RTDP on three MovingAI maps with
1–6 agents.

## Project layout

```text
src/mmdp/                 Installable Python package
    domain/               Grid MMDP, heuristic, MovingAI map loading
    planning/             RTDP/LRTDP engine, domains, configuration, and shared helpers
    evaluation.py         Fixed-policy evaluation
    resource_monitor.py   CPU/RSS sampling thread
    experiments/          Experiment factory, runner, schema, and final config
    analysis/             Notebook visualization
scripts/                  Command-line entry points
    run_compact_matrix.py Run one final difficulty group
    run_experiments.py    Low-level experiment runner
    analyze_compact_results.py
                          Create the three final graphs
maps/                     Bundled MovingAI benchmark maps
tests/                    Dedicated test suite
    unit/                 Focused component/configuration tests
    integration/          Planner smoke and end-to-end tests
```

The final experiment does **not** use manifest files. Its complete configuration
is stored in `src/mmdp/experiments/final_config.py`.

Small helper-only modules were consolidated into their closest owning modules: planning helpers are in `planning/config.py`, optional resource-profile helpers are in `experiments/factory.py`, and CLI-only statistical defaults are in `scripts/run_experiments.py`.

## Setup

```bash
pip install -e .[notebooks,dev]
```

## Run tests

```bash
pytest
```

## Run in Google Colab

Open `MMDP-OD-RTDP-MAIN.ipynb` and choose one project source in the first code
cell:

- `github` clones the repository automatically.
- `manual_folder` uses a complete project folder already uploaded or copied into
  Colab; set `MANUAL_PROJECT_PATH` to that folder.
- `manual_zip` opens a Colab upload dialog, extracts one uploaded project ZIP,
  and locates the project root automatically.

A valid project folder must contain `pyproject.toml`, `src/mmdp`, `scripts`, and
`maps`. After preparation, run one of the three difficulty cells. Each cell
appends results to:

`/content/MMDP_OUTPUT/MMDP_results.csv`

Re-running a cell resumes from the CSV and skips completed run IDs.

## Run locally

```bash
python scripts/run_compact_matrix.py --group easy --output local_results/MMDP_results.csv
python scripts/analyze_compact_results.py local_results/MMDP_results.csv --group easy --output-dir local_results/graphs/easy
```

Replace `easy` with `medium` or `hard` for the other maps.

## Selected maps

- Easy: `empty-8-8`
- Medium: `warehouse-10-20-10-2-1`
- Hard: `room-64-64-16`

## Fixed final experiment

- one fixed seed: `20260708`
- Baseline RTDP and OD-RTDP
- 1–6 agents
- one map per difficulty level
- 12 conditions per difficulty: 6 agent counts × 2 algorithms
- planning: up to 60 seconds or until the initial state is labelled solved
- evaluation: up to 5 episodes, with an 8-second stage budget
- episode step cap by difficulty: 80 / 160 / 260
- condition watchdog: 75 seconds
- slip-to-stay probability: 0.20
- transition-cache bound: 100,000 entries
- evaluation diagnostics and evaluation conflict-risk calculations disabled

The same fixed seed initializes separate planning and evaluation RNG objects.
This preserves reproducibility without treating seeds as a separate experimental
dimension.

## Retained compact metrics

- planning time
- peak planning-memory increase
- real states examined
- evaluation success rate

The compact CSV also keeps the fixed seed, status, stopping reason, number of
completed evaluation episodes, and total condition time so failures and
timeouts remain visible.

The analysis creates exactly three figures per selected map:

1. planning time by number of agents
2. peak planning-memory increase by number of agents
3. real states examined and evaluation success rate
