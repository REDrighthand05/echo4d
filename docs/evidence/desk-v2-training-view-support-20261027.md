# Desk V2: frozen 48-training-view support versus residual tail (2026-10-27)

**Research-only. No new reconstruction, no new geometry PASS.** This executes the question and 0/1–2/≥3 bins pre-registered in [Issue #15 comment 6431196534](https://github.com/REDrighthand05/echo4d/issues/15#issuecomment-6431196534). The 48/12 split, original Mesh, estimated poses, 0.05m support tolerance, 0.30m residual tail, and original 80% coverage / 15cm RMSE screening gate were not tuned. The original **93.7357% coverage / 16.7779cm RMSE remains FAIL**.

## Immutable inputs and commands

- Base main: [94645c4e8216e242139d15f6cbc5a75f75abd3ca](https://github.com/REDrighthand05/echo4d/commit/94645c4e8216e242139d15f6cbc5a75f75abd3ca).
- Original real [run 40404725134](https://github.com/REDrighthand05/echo4d/actions/runs/40404725134), [artifact 12689728255](https://github.com/REDrighthand05/echo4d/actions/runs/40404725134/artifacts/12689728255), ZIP SHA256 `df328812537ea4e4667f36ec7cdd9f901ff5c7fe25a1b37022daf81c20538104` (CRC clean).
- Frozen Mesh SHA256 `320da61fcd421c0e55aa6fc014750aeaa3e6e2d98b607e9feaae268f83d2a939`, 4,543 vertices, 7,578 triangles. Original 12 saved array hashes all matched the report.
- Local executed script SHA256 `47d3c33f42f05a98a7b4cc9c191b71d4c93efafc1c3d8afac456b434dc93a007`. Independent oracle SHA256 `e88503d75377bb7bd7d1faae404d88cef0e6c8fe40e707a25e5bc161c43675df`. Original generated JSON SHA256 `b219388ece3a09f89793e69ff8b91b6467cfa27ac1eca1d2e25481e4fde7dab0` (regenerate from the script; not a claim of a new machine CI artifact).

Run from this Draft checkout, after downloading the exact original ZIP:

```bash
python tools/desk_training_support_audit.py --evidence-zip desk-residual-diagnostic-40404725134.zip --output desk-training-support-audit.json
python tools/desk_support_independent_oracle.py --evidence-zip desk-residual-diagnostic-40404725134.zip
sha256sum desk-residual-diagnostic-40404725134.zip desk-training-support-audit.json
```

The execution used local Linux Python/NumPy and the original CPU rasterization algorithm; **no new GitHub Actions CI or Windows execution** is claimed. The first standalone local attempt failed before any support computation because holdout entries were strings rather than objects (`TypeError: string indices must be integers`). Parsing was corrected without changing hypotheses or thresholds, then completed. A second shorter script using repository rasterization reproduced the **same report bytes**. Separately, re-rendering all 12 holdout predictions produced **zero different float32 pixels**. A Möller–Trumbore ray/triangle oracle (independent from screen-space barycentric rasterization) checked 40 rays across five training cameras: maximum depth difference **1.18e-7m**. This is a different mathematical check, not different-human review.

## Frozen rule

For every original **overlap** pixel, backproject its predicted camera-Z using the original estimated holdout pose. For each of 48 estimated training cameras, project that world point; require positive camera-Z and a nearest integer in-frame pixel (`np.rint`, ties to even). Render the *same frozen Mesh* into that training view; count one support when its z-buffer is nonzero and within **0.05m** of projected camera-Z. **No training measured depth** enters support. Held-out measured depth only labels original residuals; no pixel is dropped or refitted.

## Results (all 158,851 original overlapping pixels)

| Training-view support | Pixels | Original abs residual ≥30cm | Tail rate | Original SSE (m²) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 53 | 6 | 11.32% | 3.590 |
| 1–2 | 316 | 66 | 20.89% | 52.800 |
| ≥3 | 158,482 | 5,391 | 3.40% | 4,415.235 |
| **All** | **158,851** | **5,463** | **3.44%** | **4,471.624** |

**99.768%** of overlapping pixels, **98.739%** of squared error, and **98.682%** of ≥30cm tail pixels have at least three training-view supports under this definition. This rejects the *strong, specific hypothesis* that most of the original squared error lies on low-support (0–2) Mesh points.

Low-support pixels have a higher conditional tail rate (72/369 = **19.51%**) than ≥3 (5,391/158,482 = **3.40%**), but are only **0.232%** of overlap. One of 12 holdout frames has no low-support pixels; among the other 11, low-support tail rate is greater in **7**, lower in **4**. No stable per-frame negative association is established. Counts, tails, and SSE conserve over the original 12 frames.


### Archived raw output and per-frame counterexample check

The exact **original** 12-frame output is now committed as [`desk-v2-training-view-support-20261027.json`](desk-v2-training-view-support-20261027.json), SHA256 `b219388ece3a09f89793e69ff8b91b6467cfa27ac1eca1d2e25481e4fde7dab0` (Git blob `f566526acb814837ba09aa91058a6779463f7da4`). It is not the independently replayed JSON, which had SHA256 `1470b925e7e57f9d34728dbbfcd911b69597d188ddd9cd11c71d3f893485c040` despite identical headline counts. The original local Python/NumPy versions were not recorded; the separately executed reviewer replay used **Python 3.12.14 / NumPy 2.2.6**. Do not conflate these environments or infer byte-for-byte determinism from equal counts.

Pre-specified low-support (0–2) minus ≥3-support **within-frame ≥30cm tail-rate difference**, in percentage points; `N/A` means zero low-support pixels:

| Holdout frame | Low-support n | Low-support ≥30cm tails | ≥3-support n | ≥3-support tails | Tail-rate difference (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| frame_00004.png | 6 | 3 | 13071 | 957 | 42.6784 |
| frame_00009.png | 15 | 9 | 13230 | 98 | 59.2593 |
| frame_00014.png | 4 | 0 | 15323 | 58 | -0.3785 |
| frame_00019.png | 2 | 0 | 13593 | 509 | -3.7446 |
| frame_00024.png | 1 | 0 | 12677 | 153 | -1.2069 |
| frame_00029.png | 72 | 9 | 12297 | 707 | 6.7506 |
| frame_00034.png | 0 | 0 | 15295 | 0 | N/A |
| frame_00039.png | 41 | 27 | 14765 | 176 | 64.6617 |
| frame_00044.png | 23 | 7 | 10604 | 715 | 23.6920 |
| frame_00049.png | 13 | 2 | 13359 | 411 | 12.3080 |
| frame_00054.png | 186 | 13 | 10514 | 904 | -1.6088 |
| frame_00059.png | 6 | 2 | 13754 | 703 | 28.2221 |

Among 11 comparable frames, 7 differences are positive and 4 negative; observed range **−3.7446 to +64.6617 pp**. The low-support group is extremely small (369 pixels overall). These frame-wise counterexamples limit any cross-frame association claim and do not alter the narrow error-mass conclusion or original geometry FAIL.

## Limits, review, rollback

This support proxy is **not independent surface truth**: the same Mesh and estimated camera poses define both the original prediction and the support test. A geometrically wrong surface may be visible from many training views. Nearest-pixel rounding, the 5cm threshold, self-occlusion and heavily overlapping cameras affect the result. Do **not** conclude that poses, occlusion, geometry or depth are correct; the narrow low-support error-*mass* hypothesis is all this falsifies.

Next exploration must be separately pre-registered and test pose/projection versus surface/occlusion with a new frozen validation scene if a fix is selected. Do not tune on these 12 holdouts and reuse them as independent validation. The AI/Hi3D branch, RGB-only predicted-depth C and external scanned-surface truth remain unverified. This Draft changes neither main nor frozen data. Rollback: close the Draft; retain the original artifact and failure receipt.
