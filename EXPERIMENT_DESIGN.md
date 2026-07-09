# Proposed final experiment design

## Phase A — correctness and sensitivity

Use the three diagnostic maps plus the bundled `empty-8-8`, `maze-32-32-2`, and `random-32-32-10` maps.
Run 2–4 agents to empirical stability with no resource limits. Verify:

- Baseline and OD real-state initial values agree within the residual tolerance;
- both fixed policies succeed;
- OD sequential and OD global diagnostic policies agree or any disagreement is
  explained by the failure JSON;
- conclusions are unchanged under a small tolerance sensitivity grid.

## Phase B — unconstrained efficiency

For each selected map/agent/scenario condition, use paired independent seeds and
run each algorithm in a fresh process until its initial state is LRTDP-solved. Report:

- elapsed time and time to initial-state solved;
- peak RSS delta and total peak RSS;
- Bellman backups, complete joint actions, and transition outcomes;
- value-state and cache sizes;
- policy success, cost, and makespan.

This phase generates the calibration data for constrained phases.

## Phase C — constrained efficiency

Derive one common time and memory profile per map and agent count from Phase B.
The primary profile is the paired median requirement. Run:

1. time only;
2. memory only;
3. time and memory together.

Both algorithms receive exactly the same externally generated limit. Optional
q25 and q75 profiles provide strict/generous sensitivity analyses.

## Pairing and replication

- Pair Baseline and OD on map, agent count, scenario, task offset, planning
  seed, and evaluation seed.
- Planning and evaluation seeds are independent.
- Start with 3 pilot pairs; use 10 pairs in the final matrix if runtime permits.
- Use several scenarios rather than relying on scenario 1/task offset 0.

## Primary outcomes

- Unconstrained: time and peak memory until the initial state is solved.
- Constrained: policy success rate, sum of arrival times, makespan, and resource
  stop reason.
- Computational mechanisms: complete joint actions, transition outcomes,
  trials, backups, and policy decision time.

## Interpretation

OD's structural claim is lower branching and fewer complete joint-action
computations. Whether that produces a faster or smaller end-to-end solver is an
empirical question; the experiment must permit a crossover where Baseline is
better at small agent counts and OD becomes preferable as joint branching
increases.
