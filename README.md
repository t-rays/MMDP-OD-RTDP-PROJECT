# Baseline RTDP vs. Operator-Decomposition RTDP — v7.1 bundled maps

Implementation version written to result CSV files:

```text
resource-modes-diagnostics-v7
```

Version 7.1 is the self-contained packaging revision of the v7 pre-final experimental framework. It keeps the corrected
sum-of-arrival-times heuristic from v6, adds failure diagnostics, independent
paired seeds, official and purpose-built small maps, real process-memory
measurement, isolated subprocess execution, and four explicit resource modes.

## Installation

Python 3.12 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
```

`psutil` is used to sample process RSS. Final memory comparisons should always
be run through `run_isolated_matrix.py`, which launches every condition in a
fresh Python process.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Current status:

```text
Ran 32 tests
OK
```

## Maps

All maps and scenario files required by the supplied manifests are already
bundled in the project. No network access or download step is required. Every
map keeps the same folder format used in the earlier project versions:

```text
maps/<map-name>/
├── <map-name>.map
└── scen/
    ├── ...-1.scen
    ├── ...-2.scen
    └── ...-3.scen
```

The three previously used large benchmark maps remain in the project:

- `room-64-64-16`
- `warehouse-20-40-10-2-1`
- `maze-128-128-10`

They were not replaced. Smaller maps were added to create a controlled
difficulty ladder and to permit full run-to-stability correctness tests:

- `empty-8-8`: very small open grid;
- `maze-32-32-2`: medium narrow-connectivity maze;
- `random-32-32-10`: medium sparse-obstacle map;
- `warehouse-10-20-10-2-1`: medium structured warehouse.

For these four bundled benchmark maps, three deterministic scenario files are
included. Scenario 1 uses shorter start-goal distances, scenario 2 medium
distances, and scenario 3 longer distances. Each file contains 12 pairwise
disjoint tasks, so task offsets `0`, `4`, and `8` are safe for experiments with
up to four agents. The distance bands were derived from sampled four-way
shortest-path quantiles on each map, rather than chosen as fixed path lengths.

The targeted diagnostic maps also remain bundled:

- `crossing-9-9`: agents share one intersection;
- `corridor-passing-11-7`: narrow corridor with a passing bay;
- `bottleneck-13-9`: two rooms connected by one doorway.

These maps test specific coordination mechanisms and complement the benchmark
maps. A complete inventory and source note is stored in `maps/MAPS_INCLUDED.md`.
The diagnostic maps can still be regenerated locally with:

```powershell
python src/generate_diagnostic_maps.py
```

## Random seeds

When explicit seeds are omitted, the runner derives independent planning and
evaluation seeds from a master seed:

```text
--master-seed 20260708
--seed-count 5
```

For every seed index, Baseline and OD receive the same planning seed and the
same evaluation seed. The evaluation seed is different from the planning seed.
This gives paired comparisons while preventing the planning trajectory from
reusing evaluation randomness. All concrete seeds are written to the CSV and
shown in subprocess progress messages.

Explicit paired seeds remain supported:

```text
--planning-seeds 11 22 33
--evaluation-seeds 101 202 303
```

## Four resource modes

### 1. `unconstrained`

No planning time or RSS limit is imposed. Planning stops when the initial
state is labelled solved by an LRTDP-style greedy-envelope check. A state is
labelled solved only when its scale-aware Bellman residual is within tolerance
and every positive-probability successor of the fixed deterministic greedy
action is terminal, already solved, or part of the same locally consistent
envelope. This mode measures time and peak RSS to a policy-relevant solved
criterion rather than to a streak of sampled trials.

### 2. `time`

A map/agent-specific time budget is applied. No memory budget is imposed and
stability does not stop the run early; both algorithms receive the full time
budget.

### 3. `memory`

A map/agent-specific additional-RSS budget is applied. There is no planning
time limit. The run stops when the initial state is solved or when the memory
limit is reached.

### 4. `time_memory`

Both map/agent-specific budgets are applied. Planning stops when the first
resource limit is reached. Stability is recorded but does not stop the fixed
budget run early.

Memory limits are cooperative RSS limits sampled by the process monitor. A run
may exceed the requested value slightly between samples. The CSV records the
actual peak and whether the memory condition fired.

## Deriving map-specific limits instead of inventing them

### Step 1: unconstrained pilot

Run the supplied pilot manifest; all referenced maps are already bundled:

```powershell
python src/run_isolated_matrix.py experiment_manifest_unconstrained_pilot.json --overwrite
```

Every condition runs in a fresh process. The pilot uses several maps,
scenarios, agent counts, and independent seed pairs.

### Step 2: calibrate a common profile

```powershell
python src/calibrate_resource_profiles.py `
    results/unconstrained_pilot_v7.csv `
    resource_profiles_calibrated.json
```

The default calibration is `paired_max` at the median:

1. pair Baseline and OD by map, agent count, scenario, task offset, and seed;
2. take the larger time-to-stability and larger peak RSS within each pair;
3. take the median of those paired requirements for that map/agent group.

Thus both algorithms receive the same externally derived budget and the
profile is not calibrated in favour of either one. No rounding is applied by
default.

Sensitivity profiles can be generated without code changes:

```powershell
# stricter lower-quartile limits
python src/calibrate_resource_profiles.py PILOT.csv profile_q25.json --quantile 0.25

# more generous upper-quartile limits
python src/calibrate_resource_profiles.py PILOT.csv profile_q75.json --quantile 0.75
```

### Step 3: run the four-mode matrix

```powershell
python src/run_isolated_matrix.py experiment_manifest_final_template.json --overwrite
```

The final template is intentionally large. Inspect it and use `--dry-run`
before starting:

```powershell
python src/run_isolated_matrix.py experiment_manifest_final_template.json --dry-run
```

An external watchdog can be supplied only as a technical safety mechanism:

```text
--watchdog-seconds 21600
```

A watchdog termination is logged separately and must not be reported as
algorithmic convergence.

## Diagnostics

Use:

```text
--evaluate-od-global-diagnostic
--diagnostics-output-dir results/diagnostics
```

For OD, the runner then evaluates two policies from the same value table:

1. normal sequential prefix extraction;
2. a diagnostic global real-state policy that enumerates complete joint
   actions using OD's learned real-state values.

This separates a problem in prefix policy extraction from a problem in the
learned real-state value function.

Per failed episode, the JSON diagnostics include:

- explicit failure reason (`step_limit` or deterministic self-loop policy);
- repeated state/action pairs and their counts;
- expected self-loop probability;
- vertex-conflict and edge-swap probability;
- non-collision no-motion probability;
- unfinished `stay` and blocked actions;
- arrival times and final state.

The CSV also contains policy-cache statistics, transition-cache statistics,
planning/evaluation RSS, time to first stability, and the overall process peak.

## Replacing hidden constants with interpretable parameters

### Evaluation episodes

If `--evaluation-episodes` is omitted, the count is derived from:

```text
--evaluation-confidence 0.95
--evaluation-half-width 0.10
```

The worst-case normal approximation gives 97 episodes. These are explicit
precision choices, not universal truths; tighter precision requires more
episodes.

### Episode/trial step cap

The old fixed `5 * expected distance` multiplier is no longer the default.
For each isolated agent, v7 derives a conservative binomial lower-tail bound
from the shortest-path distance and slip probability. The per-agent tail target
is obtained from:

```text
--step-cap-familywise-error 0.01
```

using a union bound over evaluation episodes and agents. Individual bounds are
summed, allowing a conservative sequential completion schedule. An explicit
legacy multiplier remains available only for controlled sensitivity tests.

### Solved-state stopping and legacy stability streak

`unconstrained` and `memory` modes now use LRTDP-style solved-state stopping.
The former consecutive-trial rule remains available only for sensitivity and
backward-compatible experiments through `--stop-when-stable`.

If `--stable-trials-required` is omitted, the legacy streak length is still
derived from:

```text
--stability-confidence 0.99
--minimum-unstable-trial-rate 0.10
```

This produces 44 consecutive stable trials, but it is no longer the default
run-to-convergence rule. The CSV reports both solved-state fields and the old
stability diagnostics.

Solved-state checking can legitimately be expensive for stochastic multi-agent
problems because it must inspect every positive-probability successor in the
current greedy policy envelope. A medium map with three or four agents may
therefore take minutes even when the sampled policy already succeeds. That time
is part of the run-to-solved measurement, not an accidental fixed delay.

### Residuals and ties

Residuals use an absolute plus relative tolerance:

```text
absolute epsilon = 1e-8
relative epsilon = 1e-6
```

This avoids treating the same absolute difference identically at values near 1
and near 10,000. Greedy ties use ULP-based comparison by default rather than a
problem-scale absolute tolerance. All settings are recorded in every row and
should receive a small-map sensitivity check before the final report.

## Direct example

A short paired diagnostic run can be launched directly:

```powershell
python src/experiments.py `
    maps/diagnostic/crossing-9-9 `
    --agent-counts 2 3 `
    --seed-count 3 `
    --scenario-numbers 1 2 3 `
    --resource-mode time `
    --time-limit-seconds 10 `
    --evaluation-episodes 30 `
    --evaluate-od-global-diagnostic `
    --diagnostics-output-dir results/crossing_diagnostics `
    --output results/crossing_direct.csv `
    --overwrite
```

For final memory comparisons, prefer the isolated manifest runner instead of a
direct multi-condition process.
