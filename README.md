# Stochastic Multi-Agent Path Finding with RTDP and Operator Decomposition

This project compares **Baseline Real-Time Dynamic Programming (RTDP)** with
**Operator-Decomposition RTDP (OD-RTDP)** in a cooperative stochastic
multi-agent path-finding problem.

The experiment measures how the two planning representations behave as the
number of coordinated agents increases. The reported outcomes are planning
convergence, planning time, peak additional planning memory, completion of the
full experimental condition, and policy-evaluation success under a fixed
budget.

## Problem model

The team is modeled as a centralized, fully observable cooperative
Multi-Agent Markov Decision Process (MMDP) formulated as an undiscounted
Stochastic Shortest Path problem.

- A state is the ordered tuple of all agent positions.
- A policy maps each joint state to one joint action.
- Each agent chooses `Up`, `Down`, `Left`, `Right`, or `Stay`.
- An intended movement succeeds with probability `0.80` and becomes `Stay`
  with probability `0.20`.
- Agents remain at their goals after arrival.
- A vertex conflict places two agents in the same cell.
- An edge-swap conflict makes two agents exchange positions in one step.
- A sampled joint outcome containing either conflict is rejected, so the team
  remains in the current state.
- The transition cost is the number of agents that have not reached their
  goals. Total accumulated cost is therefore the sum of individual arrival
  times.

With five local actions per agent, a team of `N` agents has `5^N` complete
joint actions in every real state. The joint state space also grows rapidly
with the number of possible position combinations.

## Planning algorithms

### Baseline RTDP

Baseline RTDP operates on real joint states and evaluates complete joint
actions directly during each Bellman backup. A Bellman backup replaces the
stored value of a state with the minimum expected immediate and future cost
over the available actions.

### OD-RTDP

OD-RTDP constructs the same simultaneous joint action one local choice at a
time. An intermediate OD state contains:

```text
(real joint state, selected action prefix)
```

The physical transition is applied only after the prefix contains one action
for every agent. This is a computational serialization of action selection,
not turn-taking by the agents. Operator decomposition reduces local action
branching while adding prefix states and additional decision depth.

### Shared heuristic

Both planners use the same obstacle-aware and slip-aware admissible heuristic.
For each agent, reverse breadth-first search from the goal computes the shortest
obstacle-aware distance `d_i` from every reachable cell. With movement success
probability `q = 0.8`, the real-state heuristic is:

```text
h(s) = sum_i d_i(s) / q
```

The estimate ignores waiting, blocking, and collision delays, so it does not
overestimate the cooperative expected cost. OD prefix states use a compatible
estimate for committed and uncommitted agents. At an empty prefix, the OD and
Baseline heuristics are equal.

### LRTDP-style termination

Planning uses repeated RTDP trials and an LRTDP-style solved-state test.

A **goal state** is a physical state in which all agents are at their goals. A
**solved state** is a planning label: its Bellman residual is within tolerance,
and its relevant positive-probability successors under the greedy policy are
terminal, already solved, or part of the same locally consistent region.

Planning stops when the initial state is labelled solved or when the planning
time limit is reached.

## Experiment configuration

The fixed configuration is defined in:

```text
src/mmdp/experiments/final_config.py
```

| Setting | Value |
|---|---:|
| Seed | `20260708` |
| Algorithms | Baseline RTDP, OD-RTDP |
| Agent counts | 1–6 |
| Maps | 3 |
| Conditions per map | 12 |
| Total conditions | 36 |
| Execution | Serial |
| Process isolation | One process per condition |
| Planning limit | 60 seconds |
| Condition watchdog | 75 seconds |
| Evaluation episodes | 5 |
| Evaluation time limit | 8 seconds |
| Slip probability | 0.20 |
| Transition-cache capacity | 100,000 entries |

The benchmark groups are:

| Group | Map | Evaluation step cap |
|---|---|---:|
| Easy | `empty-8-8` | 80 |
| Medium | `warehouse-10-20-10-2-1` | 160 |
| Hard | `room-64-64-16` | 260 |

Baseline RTDP and OD-RTDP receive the same map, task selection, parameters, and
seed in each paired condition.

## Project structure

```text
MMDP-OD-RTDP-MAIN.ipynb
README.md
pyproject.toml

maps/
    empty-8-8/
    warehouse-10-20-10-2-1/
    room-64-64-16/

src/mmdp/
    domain/
        grid_mmdp.py
        heuristic.py
        map_creator.py
    planning/
        baseline_rtdp.py
        od_rtdp.py
        planner.py
        domain_base.py
        components.py
        config.py
        results.py
    experiments/
        final_config.py
        factory.py
        runner.py
        schema.py
    analysis/
        notebook_visualizer.py
    evaluation.py
    resource_monitor.py

scripts/
    run_compact_matrix.py
    run_experiments.py
    analyze_compact_results.py

tests/
```

`run_compact_matrix.py` runs one map group in serial order. Each condition is
launched through `run_experiments.py` in a fresh process. The analysis script
creates the memory figures and result tables.

## Installation

Python 3.10 or newer is required.

Runtime installation:

```bash
python -m pip install -e .
```

Installation with notebook and test dependencies:

```bash
python -m pip install -e ".[notebooks,dev]"
```

## Tests

Run the complete test suite from the repository root:

```bash
python -m pytest -q
```

The tests cover configuration validation, shared planning components, Baseline
and OD planning, the single-condition runner, the serial experiment matrix,
resume behavior, and analysis outputs.

## Running the experiment locally

Create one CSV and append the three map groups:

```bash
RESULTS="local_results/MMDP_results.csv"

python scripts/run_compact_matrix.py --group easy   --output "$RESULTS"
python scripts/run_compact_matrix.py --group medium --output "$RESULTS"
python scripts/run_compact_matrix.py --group hard   --output "$RESULTS"
```

A dry run prints the 12 conditions in one group without executing them:

```bash
python scripts/run_compact_matrix.py \
  --group easy \
  --output local_results/MMDP_results.csv \
  --dry-run
```

The matrix runner resumes from an existing CSV by skipping recorded `run_id`
values. Use a new output path or delete the existing CSV to run every condition
again. The runner rejects a CSV whose header does not match the current schema.

## Running in Google Colab

Open:

```text
MMDP-OD-RTDP-MAIN.ipynb
```

The notebook can load the project from GitHub, an existing Colab folder, or an
uploaded ZIP. It visualizes each benchmark map, runs the three experiment
groups, and displays the generated tables and figures.

Results are written to:

```text
/content/MMDP_OUTPUT/MMDP_results_final_report.csv
```

Analysis outputs are written to:

```text
/content/MMDP_OUTPUT/final_report_outputs/
```

## Results CSV

The CSV contains one row per experimental condition with the following fields:

```text
run_id
map_group
map_name
n_agents
algorithm
seed
status
planning_stop_reason
planning_time_seconds
planning_peak_memory_delta_mb
evaluation_successful_episodes
evaluation_failed_episodes
evaluation_episodes_completed
evaluation_uncompleted_episodes
evaluation_scheduled_episodes
evaluation_success_rate
evaluation_time_seconds
condition_time_seconds
```

### Status and stopping values

- `status = ok` means the condition returned a result.
- `status = condition_timeout` means the complete condition exceeded 75
  seconds and was stopped by the supervising process.
- `planning_stop_reason = initial_state_solved` means the initial state passed
  the solved-state test. `planning_time_seconds` is then a convergence time.
- `planning_stop_reason = time_limit` means the planner used the 60-second
  planning budget without solving the initial state. The reported value is a
  consumed budget, not a time-to-solution measurement.

### Policy evaluation

Five episodes are scheduled for each returned condition. An episode succeeds
only when all agents reach their goals within the map-specific step cap.

The reported success rate uses the fixed scheduled denominator:

```text
evaluation_successful_episodes / evaluation_scheduled_episodes
```

Episodes not completed within the evaluation time limit remain recorded as
uncompleted and count as unsuccessful in this budgeted end-to-end measure.

## Analysis outputs

Run the analysis once for each group:

```bash
OUTPUTS="local_results/report_outputs"

python scripts/analyze_compact_results.py "$RESULTS" --group easy   --output-dir "$OUTPUTS"
python scripts/analyze_compact_results.py "$RESULTS" --group medium --output-dir "$OUTPUTS"
python scripts/analyze_compact_results.py "$RESULTS" --group hard   --output-dir "$OUTPUTS"
```

The output directory contains:

```text
figure_1a_open_grid_peak_planning_memory.png
figure_1b_warehouse_peak_planning_memory.png
figure_1c_room_peak_planning_memory.png
table_1_open_grid_planning_outcomes.csv
table_1_warehouse_planning_outcomes.csv
table_1_room_planning_outcomes.csv
table_2_budgeted_policy_evaluation.csv
```

Hatched memory bars identify conditions that reached the 60-second planning time limit; the hatching does not indicate a memory limit. A timeout marker identifies conditions that returned no memory measurement.

## Reproducibility

The random seed, maps, task selection, algorithm parameters, execution order,
and resource limits are fixed by the project configuration. Wall-clock time and
process-memory measurements can still vary with hardware, operating-system
scheduling, and concurrent system load.
