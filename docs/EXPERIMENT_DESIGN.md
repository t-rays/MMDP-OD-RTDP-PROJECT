# Final focused experiment

The final experiment compares Baseline RTDP with Operator-Decomposition RTDP.

## Fixed conditions

- one representative map per difficulty group
- 1--6 agents
- 2 paired random seeds
- planning stops when the initial state is solved or after 60 seconds
- evaluation requests up to 5 episodes and stops starting new episodes after 8 seconds
- evaluation diagnostics and conflict-risk calculations are disabled
- each episode has a fixed step cap by difficulty: 80 / 160 / 260
- the transition cache is bounded to 100,000 entries for both algorithms
- each complete condition has a 75-second technical watchdog

Selected maps:

- easy: `empty-8-8`
- medium: `warehouse-10-20-10-2-1`
- hard: `room-64-64-16`

Each difficulty cell therefore contains 24 conditions: 1 map × 6 agent counts × 2 seeds × 2 algorithms.

## Reported metrics

The compact CSV retains only:

1. planning time
2. peak planning-memory increase
3. number of real planning states examined
4. evaluation success rate

The CSV also contains minimal run identity, status, seeds, completed evaluation episodes, and condition time so failures and timeouts remain transparent.
