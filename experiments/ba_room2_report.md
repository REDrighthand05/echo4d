# Pose-side global bundle adjustment on room2 — report

Date: 2026-10-31. Job: `jobs/room2_ba` (scratch copy of `jobs/room2`; room2 itself untouched).
Scripts: `experiments/ba_room2.py`, `experiments/measure_spread.py`, `experiments/render_top.py`.

## Question

Depth-side corrections were exhausted on room1_big (affine + deformation field;
cross-frame spread stuck at 0.0834). Hypothesis under test: the remaining wall
warp is pose/trajectory error, fixable by the principled pose-side tool —
global bundle adjustment over the COLMAP reconstruction.

## What was done

1. Loaded the main reconstruction `sfm/0` (230 registered images, 48,502
   sparse points; `sfm/1` is a 16-image fragment that both stage_poses and
   fuse ignore). Cameras are **per-image** SIMPLE_RADIAL, 1080×1920.
2. Ran `pycolmap.bundle_adjustment` (pycolmap 4.2.1, API as documented —
   no adaptation needed) with `refine_focal_length=True`,
   `refine_principal_point=False`, `refine_extra_params=True`.
3. Wrote the refined model back to `sfm/0` and regenerated `poses.json` in
   stage_poses' exact format (230 poses, same keys).
4. Re-ran the production fuse stage on the scratch job.
5. Measured both clouds with the identical yardstick: the cross-frame spread
   of `experiments/prep_refine.py` (stride-8 sampling, per-frame (a,b) fits,
   1–99% trim, voxels at extent/110 = 0.1564 units, RMS distance to voxel
   centroid over points in ≥2-frame voxels), plus camera-up-aligned top
   renders at shared framing/scale.

## Numbers

| metric | room2 (before) | room2_ba (after BA) | change |
|---|---|---|---|
| Mean reprojection error | 0.775359 px | 0.775338 px | −0.003 % |
| Camera-centre shift from BA | — | mean 0.0005, median 0.0001, max 0.024 units | ≈ 1/1500 of a voxel (median) |
| Rotation change from BA | — | mean 0.0028°, max 0.114° | negligible |
| Focal change from BA | — | mean 0.14 px, max 7.5 px | negligible |
| Fused points (fuse stage) | 295,511 | 295,547 | +36 (+0.012 %) |
| Cross-frame spread (RMS) | **0.0765910** | **0.0765936** | +0.003 % (noise) |
| Cross-frame voxels / sampled points | 46,376 / 6,945,091 | 46,371 / 6,945,084 | ≈ identical |
| Top-view renders | — | 96.9 % of pixels identical; footprint 208,371 vs 208,380 px | ≈ identical |

Renders: `experiments/ba_room2_top.png` (before) and
`experiments/ba_room2ba_top.png` (after). Side by side they are the same
picture: same room mass, same splayed/feathered edges — the warp signature
is unchanged everywhere.

## Verdict

**Pose-side global BA does not reduce the warp. The hypothesis, as posed,
is refuted.** The reconstruction was already at the BA optimum before I
ran anything: COLMAP's incremental mapping ends with global BA, and a
fresh global BA moved the mean reprojection error by 0.00002 px and the
median camera by 1/1500 of a measurement voxel. Every downstream yardstick
(point count, cross-frame spread, rendered shape) moved by ≤ 0.01 % —
pure noise. There is no pose error left *for the sparse-feature
reprojection objective to see*.

What this does and doesn't rule out:

- It rules out "the trajectory is a sloppy local optimum of standard BA."
  Poses, sparse points and per-image intrinsics are jointly self-consistent
  to sub-millipixel level.
- It does **not** rule out warp that is invisible to sparse reprojection:
  BA never sees the dense depth field. The fused cloud can still bend if
  the monocular depth maps disagree with the (self-consistent) sparse
  geometry in a direction BA has no term for — e.g. depth bias correlated
  with viewing direction. That is a depth-vs-sparse *consistency* problem,
  addressable only by a joint objective that includes the dense depth
  samples (the LSQ machinery from optimize_refine, extended so poses —
  not just (a,b) and fields — are the variables).
- One concrete suspect for a future run: the reconstruction carries **230
  independent cameras** (one per frame, free focal + radial distortion
  each; focal ranges ~1590–2270 px, one outlier at 2272 px / k=0.227) for
  what was physically a single phone camera. Those surplus degrees of
  freedom can absorb real trajectory/shape error into intrinsics wobble
  while keeping reprojection error spotless. Re-running mapping/BA with a
  **single shared camera** is the cheapest remaining pose-side test and is
  strictly more principled for a one-device capture.

Bottom line for the shape problem: both single-side families are now
measured and exhausted — depth-side corrections (room1_big) and pose-side
global BA (this run). The next lever is a **joint** refinement (poses +
depth consistency in one least-squares objective) or the shared-camera
remap; neither is a tuning pass on an existing stage.
