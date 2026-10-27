# M2 experimental execution bridge (branch-only)

Depends on M1 PR #79 fixed SHA `2bd0e1ec877fbd21dfe2b651980b133bc50c7a29`; this child branch is not merged into main. It invokes **the existing known-pose RGB-D TSDF algorithm** through a typed route, atomically publishes a Mesh and file-hash execution receipt, and refuses missing or unpinned providers. It does not implement a second algorithm, RGB-only SLAM, new sensors or independent geometry quality scoring.

- `src/echo4d/graph_runner.py`: fail-closed graph adapter executor, actual Git checkout SHA and adapter source SHA-256 verification, manifest and every referenced RGB/depth/mask source-file SHA-256 (with during-run mutation detection), output SHA-256, adapter config fingerprints, typed output checks, atomic success publication and persistent `.failed.json` receipts.
- `tsdf_recording_adapter` binds the existing `echo4d reconstruct` CPU fusion. The input `echo_recording` must already contain calibrated metric right-handed camera-to-world RGB-D frames. A `recorded` source label is not independent proof of capture authenticity; manifest and individual image sources require separate audit.
- Tests execute the actual TSDF on the repository's **synthetic demo** and check a nonempty Mesh and SHA receipt, plus missing provider, wrong SHA, wrong checkout, output escape and unavailable-algorithm counterexamples. Synthetic smoke is **not** a new real-world reconstruction.

## Verification

```sh
python -m pytest -q tests/python/test_graph_runner.py
python tools/verify.py
```

## Real-data gate

Next: run a pinned public TUM recording through this graph runner, then a **different actual algorithm provider** on the same frozen input. Keep Desk V2 16.7779cm / 15cm FAIL and #74 visual FAIL unchanged. `executed_unscored` deliberately differs from geometry or human-visual PASS. This code is an API, not yet a CLI command or GUI. Rollback: abandon this dependent child branch; PR #79 and main are unchanged.

## Concurrent development note

A separate M1 branch update introduced `composable_runtime.py` while this child branch was being written. That existing CLI already runs one typed CPU TSDF route; this branch is an **experimental generic multi-provider graph executor**, not a second distinct reconstruction algorithm. Keep the implementations separate pending review and deduplicate the TSDF binding before integration. Do not count them as two real strategies.

## Two different algorithm providers (experimental, not accepted)

The projective provider is imported from independent branch `research/projective-depth-mesh-provider-m2-20261028` at `050e0a00e5a578f0cadd16d61887ff86fe1d38f8`. It triangulates per-frame observed depth rather than fusing TSDF; its stride>1 unsampled-hole rejection remains intact. `projective_recording_adapter` pins the actual algorithm source, uses the same typed metric input contract and emits a distinct provider report and Mesh.

The existing pinned TUM workflow now runs the projective provider **after** the original TSDF on the same imported 48 train/12 holdout split, using the same checkout and source. `tum_graph_projective_ab.py` refuses a mismatched baseline and scores both actual Mesh outputs with the existing frozen holdout scorer. This is a same-sensor *diagnostic*, not independent truth or a generalization win. A run counts as real A/B **only after** the corresponding Actions job succeeds and both Mesh SHA and diagnostic JSON are verified. On failure, inspect `projective-alternative/ab-failure.json`; do not call this an executed second real route.

```sh
python -m pytest -q tests/python/test_graph_runner.py tests/python/test_projective_meshing.py tests/python/test_tum_graph_projective_ab.py
```

Historical desk V2 16.7779cm > 15cm and #74 visual acceptance remain FAIL. Avoid reusing the same heldout data to select a winner without a new independent frozen scene.
