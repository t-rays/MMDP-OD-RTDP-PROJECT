# MMDP with Operator Decomposition

This project compares Baseline RTDP and OD-RTDP on three MovingAI maps with
1--6 agents.

## Project layout

```
src/mmdp/                 Installable Python package
    grid_mmdp.py          Stochastic cooperative grid MMDP
    heuristic.py          Admissible shortest-path heuristic
    map_creator.py        MovingAI .map/.scen loading
    config.py             RTDPConfig shared by both algorithms
    components.py         Injectable ValueStore / SolvedTracker / TieBreaker
    domain_base.py        Shared RTDP domain plumbing
    baseline_rtdp.py      Baseline RTDP planning domain
    od_rtdp.py            Operator-decomposition planning domain
    planner.py            Generic RTDP/LRTDP engine
    results.py            Trial and planning result dataclasses
    evaluation.py         Fixed-policy evaluation
    experiments/          Experiment orchestration (schema, factory, runner)
scripts/                  Thin command-line entry points
tests/                    Pytest smoke and unit tests
maps/                     Bundled MovingAI benchmark maps
manifests/final/          Fixed experiment definitions per difficulty
docs/                     Experiment design and output documentation
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

`local_testing.ipynb` runs the same pipeline against `local_results/`, or use
the scripts directly:

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
- condition watchdog: 75 seconds
- 2 paired seeds
- 1--6 agents
- cache bound: 100,000 transition entries
- no evaluation diagnostics or conflict-risk calculations

## Retained metrics

- planning time
- peak planning-memory increase
- real states examined
- evaluation success rate

The analysis creates only three figures per selected map.
