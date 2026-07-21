# MMDP with Operator Decomposition

This project compares Baseline RTDP and OD-RTDP on three MovingAI maps with
1--6 agents.

## Project layout

```
src/mmdp/                 Installable Python package
    domain/               Grid MMDP, heuristic, MovingAI map loading
        grid_mmdp.py      Stochastic cooperative grid MMDP
        heuristic.py      Admissible shortest-path heuristic
        map_creator.py    MovingAI .map/.scen loading
    planning/             RTDP/LRTDP engine and both planning domains
        planner.py        Generic RTDP/LRTDP engine
        domain_base.py    Shared RTDP domain plumbing
        baseline_rtdp.py  Baseline RTDP planning domain
        od_rtdp.py        Operator-decomposition planning domain
        config.py         RTDPConfig shared by both algorithms
        components.py     Injectable ValueStore / SolvedTracker / TieBreaker
        results.py        Trial and planning result dataclasses
        exceptions.py     Deadline / memory-limit control-flow signals
        limits.py         Step-bound resolution
        numerics.py       Residual and tie-comparison helpers
    evaluation.py         Fixed-policy evaluation
    resource_monitor.py   CPU/RSS sampling thread
    experiments/          Experiment orchestration (schema, factory, runner)
    analysis/             Statistics helpers and notebook visualizations
scripts/                  Thin command-line entry points
tests/                    Pytest smoke and unit tests
maps/                     Bundled MovingAI benchmark maps
manifests/final/          Fixed experiment definitions per difficulty
```

## Setup

```bash
pip install -e .[notebooks,dev]
```

## Run tests

```bash
pytest
```

## Run in Google Colab

Open `MMDP_Colab_One_Map_Per_Level.ipynb`, run the preparation cells, then
run one of the three difficulty cells. Each difficulty cell runs one map and
appends results to:

`/content/MMDP_OUTPUT/MMDP_results.csv`

Re-running a cell resumes from the CSV and skips completed run IDs.

## Run locally

```bash
python scripts/run_compact_matrix.py --group easy --output local_results/MMDP_results.csv
python scripts/analyze_compact_results.py local_results/MMDP_results.csv --group easy --output-dir local_results/graphs
```

## Selected maps

- Easy: `empty-8-8`
- Medium: `warehouse-10-20-10-2-1`
- Hard: `room-64-64-16`

## Fixed experiment budget

- planning: up to 60 seconds or solved
- evaluation: up to 5 episodes, with an 8-second stage budget
- episode step cap by difficulty: 80 / 160 / 260
- condition watchdog: 75 seconds
- 2 paired seeds
- 1--6 agents (each difficulty cell = 24 conditions: 6 agent counts x 2 seeds x 2 algorithms)
- cache bound: 100,000 transition entries
- no evaluation diagnostics or conflict-risk calculations

## Retained metrics

- planning time
- peak planning-memory increase
- real states examined
- evaluation success rate

The CSV also keeps minimal run identity, status, seeds, completed evaluation
episodes, and condition time so failures and timeouts remain transparent.

The analysis creates exactly three figures per selected map:

1. planning time by number of agents
2. peak planning-memory increase by number of agents
3. real states examined and evaluation success rate
