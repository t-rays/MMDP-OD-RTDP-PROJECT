# Changes in v7

Implementation ID: `resource-modes-diagnostics-v7`

## Experimental design

- Added four resource regimes: unconstrained, time, memory, and time+memory.
- Added fresh-process matrix runner for valid peak-RSS comparisons.
- Added map/agent resource profiles and external pilot calibration.
- Calibration now defaults to a paired-algorithm median budget (`paired_max`),
  preventing either algorithm from defining its own limit.
- Fixed constrained modes so profile limits are used unless an explicit CLI
  override is supplied.

## Randomness and instances

- Added reproducible independent planning/evaluation seed pairs derived from a
  master seed.
- Both algorithms use the same pair within each condition.
- Added multiple scenario numbers and task offsets to the experiment matrix.

## Diagnostics

- Added per-failure JSON diagnostics.
- Split selected-action risk into vertex conflicts, edge swaps, non-collision
  no-motion, unfinished stay, and blocked movement.
- Added repeated state/action counts and deterministic self-loop detection.
- Added an OD global real-state diagnostic policy to distinguish learned-value
  errors from sequential prefix-extraction errors.
- Added policy-cache, transition-cache, time-to-stability, and RSS fields.

## Maps

- Added downloader for selected official MovingAI small/medium maps.
- Added generated crossing, passing-corridor, and bottleneck diagnostic maps,
  each with three scenarios.

## Reduced arbitrary constants

- Replaced the default `5 * distance` step cap with a stochastic tail bound.
- Derived default evaluation episodes from confidence and precision targets.
- Derived the stability streak from explicit detection assumptions.
- Replaced the absolute greedy tie threshold with ULP comparison.
- Replaced a single absolute residual with absolute+relative scaling.
- Removed the arbitrary OD global-prefix cap used in v5.
- Transition caches are unbounded by default; memory experiments use measured
  RSS rather than cache-entry count as the resource limit.

## Preserved correctness changes

- Retains the v6 admissible collision-safe OD-prefix heuristic.
- Retains optimized evaluation that caches only executed actions.
- Retains deterministic greedy tie resolution as the primary evaluation mode.
