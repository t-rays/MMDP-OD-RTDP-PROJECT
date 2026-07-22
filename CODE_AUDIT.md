# Code audit

## Configuration cleanup

- Removed `manifests/final/` and all JSON-manifest loading.
- Added `src/mmdp/experiments/final_config.py` as the single source of truth.
- Replaced paired-seed generation with the single constant `FIXED_SEED = 20260708`.
- Removed `src/mmdp/experiments/seeds.py`.
- Removed the seed lists/count from the CLI and compact CSV.
- Updated the notebook, README, tests, run IDs, and plots for one row per condition.

## Removed unused implementations

The following definitions had no call sites in the package, scripts, tests, or
notebook and were removed:

- `GridMMDP.goal_state`
- `GridMMDP.clear_transition_cache`
- `ShortestPathHeuristic.best_optimistic_distance`
- `ShortestPathHeuristic.value_for_od_state`
- `EvaluationResult.summary_dict`
- `RTDPDomainBase._trial_step_numbers`

Two unused imports were also removed:

- `time` from `resource_monitor.py`
- `RTDPPlanner` from `test_planning_smoke.py`

## Definitions retained intentionally

Abstract interface methods, protocol methods, package re-exports, context
manager methods, command-line entry points, and optional diagnostic paths were
retained when they have dynamic or configuration-dependent call sites. They are
not dead implementations even when a direct text search shows few references.

## Verification

Completed successfully after the cleanup:

- Python compilation of `src/`, `scripts/`, and `tests/`
- compilation of every code cell in `MMDP-OD-RTDP-MAIN.ipynb`
- `26` pytest tests
- fixed-seed Baseline and OD end-to-end smoke run
- compact CSV conversion and generation of all three analysis figures
- static definition scan: no remaining non-test function/class appears only at
  its own definition


## Consolidation pass

To reduce file fragmentation without merging the core algorithms into oversized files:

- `planning/limits.py`, `planning/numerics.py`, and `planning/exceptions.py` were merged into `planning/config.py`.
- `experiments/profiles.py` was merged into `experiments/factory.py`.
- `analysis/statistics_utils.py` was merged into the only consumer, `scripts/run_experiments.py`.
- Tests were divided under `tests/unit/` and `tests/integration/`, with shared fixtures in `tests/conftest.py`.

The larger domain, planner, evaluation, schema, and resource-monitor modules remain separate because each owns a distinct subsystem and combining them would make the code harder to review rather than simpler.

## Notebook source selection update

The Colab notebook now supports three project-loading modes without changing the experiment itself:

- `github`: clone the configured GitHub repository.
- `manual_folder`: validate and use a complete project folder already present in Colab.
- `manual_zip`: upload one ZIP, extract it safely, and detect the project root.

All three modes validate the same required structure (`pyproject.toml`, `src/mmdp`, `scripts`, and `maps`) before installation. The notebook's nine code cells compile successfully, manual-folder/nested-ZIP root detection was checked, and all 26 project tests still pass.
