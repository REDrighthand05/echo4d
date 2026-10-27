# M2 first provider bridge: typed CPU TSDF execution (branch-only)

This new module binds exactly one M1 typed graph to the **existing** ECHO CPU TSDF reconstruction function; it does not duplicate the fusion kernel, change frozen TUM protocols or execute 42 routes. It is a runnable module, not a synthetic-only planner. The graph requires metric camera-Z depth, metric camera-to-world poses, right-handed coordinates and timestamps in seconds. Source class and evidence grades must be explicitly declared; the source JSON's recorded/synthetic label is checked but **not authenticated as external sensor truth**.

## Synthetic smoke (E2, not real-world acceptance)

```bash
python -m echo4d demo --output outputs/typed-fixture --frames 8
python -c "from pathlib import Path; from echo4d.model import file_sha256; print(file_sha256(Path('outputs/typed-fixture/recording.json')))"
python -m echo4d.composable_runtime --recording outputs/typed-fixture/recording.json --expected-manifest-sha256 <PASTE_SHA256> --output outputs/typed-fixture-mesh --receipt outputs/typed-fixture-receipt.json --source synthetic --pose-evidence generated --depth-evidence generated --scene-id fixture --coordinate-id fixture_world --time-id fixture_pts --bounds -1.8 -.2 -1.9 1.8 2.2 1.8 --voxel .06
python -m pytest -q tests/python/test_composable_runtime.py
```

Each receipt is a new, non-overwritten JSON file. On success it includes actual input manifest SHA, module graph fingerprint, source-file provider revision SHA, actual git checkout SHA when discoverable, Mesh/report SHA, frame count and elapsed time. On failure it records the exception without claiming success. The existing TSDF report preserves per-frame file SHA and output evidence. Neither source labels nor train/holdout isolation can be authenticated from the existing recording format alone.

## Real-input handoff (NOT executed in this change)

Use a previously validated public TUM imported recording, its actual SHA and explicit `--source recorded --pose-evidence measured --depth-evidence measured` **only** when those evidence classes are supported by the specific import protocol. Use scene-specific bounds; do not copy synthetic bounds. Real source license, original RGB resolution, fixed train/holdout, camera and scale provenance must be checked externally. A completed Mesh is not an independent surface-truth score and does not satisfy #74's three corresponding clear RGB views plus two free views and human review. Next task: run this bridge on one frozen real TUM recording and archive the receipt plus 3+2 image evidence, then add a genuinely different provider graph.
