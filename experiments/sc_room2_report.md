# Shared-single-camera rebuild of room2 — report

Date: 2026-10-31. Job: `jobs/room2_sc` (scratch copy of `jobs/room2`;
room2 itself untouched). Script: `experiments/sc_room2.py`. Yardstick:
`experiments/measure_spread.py` (identical code/params as the BA run) and
`experiments/render_top.py`, framed jointly over room2 + room2_sc.

## Question

Follow-up to `experiments/ba_room2_report.md`: pose-side global BA moved
nothing, and its leftover suspect was the model itself — **230 independent
per-image cameras** for one physical phone (focal wandering ~1590–2270 px),
whose surplus degrees of freedom can absorb shape error into intrinsics
while reprojection error stays clean. This run removes those degrees of
freedom: rebuild from scratch with **one shared camera** for all frames.

## What was done

1. Fresh COLMAP database over all 320 frames with
   `pycolmap.extract_features(..., camera_mode=pycolmap.CameraMode.SINGLE)`
   (API verified in the venv, pycolmap 4.2.1). Verified directly in the
   database: exactly **1 camera**, all 320 images assigned to it.
2. Matching with the production pairing, unchanged: the pair list was
   reused verbatim from `jobs/room2/sfm/pairs.txt` (11,023 pairs —
   temporal band ±30 + first-40×last-40 loop pairs + every-12th-frame
   glue), matched via `ImportedPairingOptions` exactly as `stage_poses`.
3. `incremental_mapping`, best model picked the same way as production;
   `poses.json` regenerated with stage_poses' exact writer.
4. Production fuse stage on the scratch job (depth maps reused — they are
   pose-independent), then spread + top-view render with the BA run's
   yardsticks.

## Numbers

| metric | room2 (per-image cameras) | room2_sc (one shared camera) | change |
|---|---|---|---|
| Cameras in best model | 230 (one per image) | **1** | −229 intrinsics sets |
| Camera | SIMPLE_RADIAL ×230, f ≈ 1590–2270 px | SIMPLE_RADIAL ×1, **f = 1690.20 px**, cx = 540, cy = 960, k = 0.00927 | single f inside the old wander range |
| Registered images (best model) | 230 | **230** | 0 — no collapse |
| Other model fragments | 16 | 3 / 33 / 26 | mapper discarded 22 bad initial pairs before seeding (vs 0 drama in production), then converged to the same 230 |
| Sparse points | 48,502 | 48,463 | −0.08 % |
| Mean reprojection error | 0.7754 px | 0.8003 px | +3.2 % (expected: far fewer DOF) |
| Fused points (fuse stage) | 295,511 | 282,514 | −4.4 % |
| Cross-frame spread (RMS) | **0.0765910** | **0.0751171** | **−1.92 %** |
| Cross-frame voxels / sampled points | 46,376 / 6,945,091 | 45,896 / 6,875,368 | ≈ same support |
| Top-view footprint (joint framing) | 206,117 px | 195,691 px | −5.1 % (tighter, less splay) |
| Top-view renders | `experiments/sc_room2_baseline_top.png` | `experiments/sc_room2_top.png` | same room, same warp signature |

## Verdict

**No collapse, a small real gain, but no straightening.** Forcing one
shared camera:

- registered exactly as many frames (230) with a single, physically
  sensible camera (f = 1690 px — the per-image focals were wandering
  around a value like this all along);
- cost only +3.2 % reprojection error, so the 230 free cameras were
  *not* hiding a large sparse-geometry error — if they had been, the
  single-camera model would have paid for it visibly here;
- improved the fused cloud modestly: cross-frame spread −1.9 % (for
  scale: the BA run moved the same yardstick by +0.003 %, i.e. noise —
  this change is ~600× that), footprint −5 %, at −4.4 % point count;
- and **left the visible warp intact**: side by side, the top views show
  the same splayed diagonal sheets, the same feathered fringes, the same
  bent room mass. Nothing about the shape reads as straighter.

So the second pose-side suspect is also largely cleared. Two pose-side
tests in a row (global BA; shared-camera remap) leave the warp standing,
while both rule out their mechanism as its dominant cause. What remains
is what the BA report pointed at: the warp lives where sparse
reprojection cannot see it — in the **dense depth field's consistency
with the sparse geometry** (depth bias correlated with viewing
direction). The remaining lever is the joint refinement: poses *and*
depth alignment inside one least-squares objective (the optimize_refine
machinery, extended so poses are variables), not another remap.

Practical note: `CameraMode.SINGLE` is strictly more principled for a
one-phone capture and costs nothing in registration (same 230 frames,
+0.025 px reprojection). Adopting it in production `stage_poses` is a
one-argument change that buys the small spread gain and removes a
physically wrong model — worth doing on its own merits, but it is not
the shape fix.
