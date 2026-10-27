# Desk V2: frozen camera-local pose sensitivity (2026-10-27)

**Research-only; original geometry screening FAIL remains unchanged.** This executes the 12 single-axis perturbations pre-registered in [Issue #15](https://github.com/REDrighthand05/echo4d/issues/15#issuecomment-6434329855), after the 48-training-view support analysis was accepted in [PR #70](https://github.com/REDrighthand05/echo4d/pull/70). This is *not* pose calibration, an improved reconstruction, a new holdout, or independent scanned-surface truth.

## Frozen inputs and reproducibility

- Starting main: [c5f7141e6843ae685af4d2b06050539b7aa9d1fd](https://github.com/REDrighthand05/echo4d/commit/c5f7141e6843ae685af4d2b06050539b7aa9d1fd). No edits to frozen data, Mesh, split, masks, metric thresholds, or existing renderer.
- Original [real machine run 40404725134](https://github.com/REDrighthand05/echo4d/actions/runs/40404725134), [artifact 12689728255](https://github.com/REDrighthand05/echo4d/actions/runs/40404725134/artifacts/12689728255), original ZIP SHA256 df328812537ea4e4667f36ec7cdd9f901ff5c7fe25a1b37022daf81c20538104, ZIP CRC clean.
- Original Mesh SHA256 320da61fcd421c0e55aa6fc014750aeaa3e6e2d98b607e9feaae268f83d2a939; each of the 12 NPZ array SHA256 values was checked against the original residual report.
- The checked-in read-only reproducer is [tools/desk_pose_sensitivity_audit.py](../../tools/desk_pose_sensitivity_audit.py), Git blob 41a7530315e76b5229905a341eecfa0ef81e8890, SHA256 072e98421a17e8e73085c1bb8d74b7cb26e5d6f3581064e9ca0efada2ab1ae7d. It uses the existing echo4d.holdout.rasterize_mesh_depth, not a new scoring implementation.
- Local **executed** copy of this exact Git blob (same 6,481 bytes) produced full 13-case / 12-frame report SHA256 9c37861ae87954578a4e51b84c9928a7c9a110b4fa8611f2fa7f4220cf200f0b. The run used a separately verified Numba rasterizer shim for speed; a pure NumPy implementation matching the repository renderer was cross-checked *pixel-for-pixel* on baseline, +X 1cm and -X 0.5° on frame_00004 (0 differing float32 pixels). **The full report has not yet been generated with the repository renderer in CI**. This limitation matters: a passing general CI does not certify the research run.

Reproduce from this PR checkout with the exact downloaded original artifact (do not rerun SfM):

    PYTHONPATH=src python tools/desk_pose_sensitivity_audit.py --evidence-zip desk-residual-diagnostic-40404725134.zip --output desk-pose-sensitivity-audit.json
    sha256sum desk-residual-diagnostic-40404725134.zip tools/desk_pose_sensitivity_audit.py desk-pose-sensitivity-audit.json

Baseline asserts *all 12 original predicted float32 depth arrays are exactly identical* and verifies original overlap and SSE before permitting comparison. If reproduction produces a different report hash, compare full numeric results, Python/NumPy versions and renderer identity; **do not claim byte determinism without evidence**.

## Frozen intervention and comparisons

For every original estimated holdout camera-to-world pose, apply pose @ local_delta, **one axis at a time**: translations ±0.01m along X/Y/Z and rotations ±0.5° about X/Y/Z, plus baseline. No fitting, selection, exclusions or changes to the original 48/12 split. Rerender the same Mesh for all 12 heldout frames.

For each case record (a) all currently valid measured pixels covered by the perturbed prediction and their RMSE/SSE, (b) original-overlap pixels that disappear and newly covered pixels, (c) the SSE on the **same retained original pixels** for both original and perturbed predictions, and (d) ≥30cm tail counts. A lower RMSE on a *different overlap mask* is **not** a valid accuracy improvement claim.

Original: 158,851 overlapping pixels, 93.7357% coverage, SSE 4,471.6241m², RMSE 16.7779cm. The unchanged 15cm fixed-mask gate requires a reduction of **897.4766m²**.

| Intervention | New RMSE (cm) | Original pixels lost | New coverage | Perturbed SSE on retained original pixels (m²) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 16.7779 | 0 | 93.7357% | 4471.624 |
| X translation -1cm | 16.8051 | 675 | 93.6218% | 4370.254 |
| X translation +1cm | 16.8928 | 500 | 93.8259% | 4391.060 |
| Y translation -1cm | 17.1242 | 695 | 93.7038% | 4523.582 |
| Y translation +1cm | 16.6348 | 677 | 93.7581% | 4218.805 |
| Z translation -1cm | 16.9964 | 414 | 93.6235% | 4528.176 |
| Z translation +1cm | 16.6244 | 203 | 93.8702% | 4312.385 |
| X rotation -0.5° | 16.6723 | 1030 | 93.8537% | **4069.745** |
| X rotation +0.5° | 17.4319 | 1261 | 93.5822% | 4569.707 |
| Y rotation -0.5° | 16.9465 | 1179 | 93.5545% | 4338.427 |
| Y rotation +0.5° | 16.9387 | 913 | 93.8867% | 4260.616 |
| Z rotation -0.5° | 16.8453 | 515 | 93.7256% | 4376.408 |
| Z rotation +0.5° | 16.8179 | 489 | 93.7628% | 4376.119 |

**Falsification:** All 12 perturbations lose some original-overlap pixels (203–1261), so none preserves the complete fixed mask. Stronger conservative bound: even if *every* lost original pixel were assigned **zero error** and the remaining original pixels kept their perturbed errors, the lowest possible fixed-mask RMSE is **16.006cm** (X rotation -0.5°), still above 15cm. This is an optimistic *post hoc mathematical bound*, not an implemented completion method or a newly accepted score.

The largest reduction in **paired** SSE on identically retained original pixels is ~191.152m² (X rotation -0.5°), much smaller than the original 897.4766m² target; do not confuse the larger apparent full-original SSE reduction with improvements on identical pixels, because missing pixels were removed. This does **not** rule out spatially varying pose correction, multi-axis corrections, a different pose estimator, occlusion, projection or Mesh errors.

## Verification levels, limitations, rollback

Actual local original ZIP SHA/CRC, 12 array hashes and exact baseline float32 depth reproduction: **PASS**. 13 cases × 12 frames executed with a separately checked fast renderer: **PASS**, with full per-frame outputs and independent NumPy aggregation. Pure NumPy repository-algorithm cross-check for three frame/case combinations: **0 differing pixels**. Different-human review, real-device replication, Windows research execution, independent surface truth and validation on another frozen scene: **NOT DONE**. The existing heldout measured sensor depth remains an evaluation proxy, not an independent surface scan. Do not feed these 12 holdouts back into optimization.

This Draft adds only a read-only reproducer and evidence note. No main or scoring change; rollback is to close without merging. Next: obtain independent review of perturbation handedness and masking, run the exact checked-in script with the repository renderer and archive its raw JSON (with environment versions), then select a **separately pre-registered** surface/occlusion-versus-pose diagnostic on an independent frozen scene. Do not spawn another CI or repeat SfM merely to produce a second copy of these numbers.
