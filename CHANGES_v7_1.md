# Packaging revision v7.1

No planning or evaluation algorithm was changed. The implementation version in
CSV files remains `resource-modes-diagnostics-v7`.

Changes:

- removed `src/download_official_maps.py`;
- bundled every map referenced by the pilot and final manifests;
- retained the original `room-64-64-16`, `warehouse-20-40-10-2-1`, and
  `maze-128-128-10` maps;
- added a difficulty ladder with `empty-8-8`, `maze-32-32-2`,
  `random-32-32-10`, and `warehouse-10-20-10-2-1`;
- generated three deterministic shortest-path-distance scenario tiers for each
  bundled small/medium benchmark map;
- updated both experiment manifests to use only bundled map folders;
- added `maps/MAPS_INCLUDED.md` and `maps/BUNDLED_MAPS_MANIFEST.json`.
