# MMDP with Operator Decomposition — focused V16

This is the cleaned final experiment package. It compares Baseline RTDP and OD-RTDP on three MovingAI maps and 1--6 agents.

## Run in Google Colab

Open `MMDP_V16_Colab_Focused_Metrics.ipynb`, run the two preparation cells, then run one of the three difficulty cells. Each difficulty cell runs one map and appends results to:

`/content/MMDP_OUTPUT/MMDP_results.csv`

Re-running a cell resumes from the CSV and skips completed run IDs.

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

## V16 graph fix
The combined states/success figure now uses side-by-side bars. Equal Baseline and OD values no longer overlap and hide Baseline. Existing V15 compact CSV files remain compatible; experiments do not need to be rerun.
