# Solved-state stopping update

- Added LRTDP-style solved labels for Baseline RTDP real states.
- Added LRTDP-style solved labels for the full OD state space, including action prefixes.
- `unconstrained` and `memory` modes now stop when the initial state is solved.
- Kept the former consecutive-stable-trials rule as an optional legacy diagnostic.
- Added CSV fields for solved-state counts, checks, and time/trial of first solution.
- Added `--stop-when-solved` for custom runs.
- Updated implementation identifier to `lrtdp-solved-stopping-v8`.
