# Larger depth model (Depth Anything V2 Large) on room2_sc — report

Date: 2026-10-31. Job: `jobs/room2_large` (scratch copy of
`jobs/room2_sc`; room2 and room2_sc untouched — verified by mtime and
content). Driver: `experiments/large_depth.py` (scratch; production
code untouched). Yardstick: `experiments/measure_spread.py`; renders:
`experiments/render_top.py`, framed jointly over room2_sc +
room2_large (same convention as the sc and joint runs).

## Question

Every refinement family is refuted with evidence: poses (global BA),
shared camera, depth-side affine (a,b), per-frame deformation fields,
and the joint poses+depth optimization. The joint report localized the
residual warp inside the monocular depth maps themselves —
viewing-direction-correlated, intra-frame bias. The one input never
varied: the depth model itself. Production runs Depth Anything V2
**Base**; this run swaps in **Large**, everything else held fixed.

## Model acquisition (read this before re-downloading)

`models/depth_anything_v2_large.onnx` = `onnx/model.onnx` from HF
`onnx-community/depth-anything-v2-large` — the same transformers-export
family as the existing Base/Small files (input `pixel_values`, output
`predicted_depth`, verified in onnxruntime before the run).

⚠️ This VM's network path **silently corrupts large full-file
transfers**: two independent downloads (plain curl; huggingface_hub
over plain HTTPS) both arrived at exactly the right size
(1,336,922,232 B) with **wrong SHA-256** (`03e78521…`, `3b87340f…`)
and failed protobuf parsing. The file on disk now came through
`huggingface_hub` **with `hf-xet` installed** (chunk-verified
transfer); its SHA-256 is
`a93aa89b5e92e30e0afbe0f7c3ec692b35cfca791ae9004a190fb0ca2010e905`,
matching the repo's published LFS hash exactly. Do not trust a future
re-download that skips hash verification.

Pre-flight sanity (one frame, frame_00185): Large loads, output
(1920, 1080) all-finite, Spearman ρ vs the Base map = **0.9966** — same
scene ordering, as expected. Absolute value ranges differ wildly
(Large 1.2–640.3 vs Base 0.15–17.9); these are uncalibrated relative
inverse depths and the per-frame (a,b) fits absorb the scale, exactly
as designed. Full pass: 229 frames in 3,145 s ≈ **13.7 s/frame** on
this VM (4 vCPU, CPU-only).

## Setup choices (attribution hygiene)

- Frame set = poses ∩ existing depth files (**229 frames**) — the
  exact set the 0.0751171 baseline was fused/measured on. The
  room2_sc quirk is preserved on purpose so the model is the ONLY
  changed variable: frame_00163 is registered but has no baseline
  depth map; frame_00271 has a depth map but no pose. (A fresh
  production `stage_depth` would also emit frame_00163; noted, not
  done.)
- Baseline reproduced on the scratch copy before any change:
  spread **0.0751171** (45,896 cross-frame voxels, 6,875,368 sampled
  points) — identical to both prior reports.
- Fuse = the production stage, unmodified. NOTE: production fuse
  gained the K=2 corroboration trim while this experiment ran (the
  trim adoption landed the same morning), so the fused counts below
  are post-trim. The like-for-like Base number under the same fuse is
  the trim run's **208,666** (experiments/trim_report.md). The spread
  yardstick is unaffected by fuse changes — it recomputes from depth
  maps + poses directly.

## Numbers

| quantity | Base depth (room2_sc) | Large depth (room2_large) | change |
|---|---|---|---|
| Cross-frame spread (measure_spread) | **0.0751171** | **0.0736659** | **−1.93 %** |
| Cross-frame voxels | 45,896 | 43,210 | −5.9 % |
| Sampled points (spread) | 6,875,368 | 6,928,031 | +0.8 % |
| Points in cross-frame voxels | 6,212,699 | 6,371,111 | +2.6 % |
| Cloud extent (spread sampling) | 16.97 | 16.52 | −2.6 % |
| Fused points (production fuse, K=2 trim) | 208,666 | **208,720** | +0.03 % |
| Top-view footprint (joint framing) | 195,694 px | 142,271 px | see verdict — mostly the trim, not the model |

Renders: `experiments/large_sc_top.png` (before, Base/untrimmed) and
`experiments/large_room2_top.png` (after, Large/trimmed), same joint
framing.

## Verdict

**A small real gain — the same size as the single-camera gain — and
no straightening.** Swapping Base for Large:

- improves the independent spread yardstick by **−1.93 %** (for
  scale: global BA moved it +0.003 %, the joint refinement +0.47 % —
  this is a genuine effect, ~600× the BA noise floor, and it points
  the right way);
- leaves the fused output essentially unchanged in size (+54 points
  under the identical production fuse);
- and **does not visibly straighten the room**. The after-render is
  tighter than the before-render, but feature-for-feature that
  tightening is the K=2 trim's known signature (fringe skirt and
  fly-out sheets cut — the trim experiment's renders show the same
  change on Base depth); the corroborated core keeps the same splayed,
  diagonally-warped character, the same bent room mass. So the
  footprint delta above must not be read as shape improvement.

Interpretation, consistent with every prior run: model scale shrinks
the depth maps' intra-frame bias slightly but does not change its
character. The bias appears to be a property of monocular depth on
this footage (one frame at a time, no cross-view knowledge), not of
one checkpoint's capacity — Base → Large buys ~2 %, not a new regime.
With this, **every lever on the current input family is now measured**:
poses, cameras, alignment, deformation, joint optimization, evidence
trimming, and depth-model scale. What remains outside this family is
what the joint report already named: a depth source with cross-view
consistency built in (measured multi-view depth — blocked on this VM:
PatchMatch needs CUDA, SGBM is inapplicable to forward-motion
footage), or accepting the current ceiling.

## If Large were adopted into production (not done here)

1. `models/depth_anything_v2_large.onnx` is already in place
   (1.34 GB — a real packaging cost for the exe).
2. `stage_depth` in `echo_scene/pipeline.py` picks Base over Small
   (the `base_model`/`small_model` preference, ~lines 226–231); it
   would need to prefer the Large file. No other code change:
   `echo_scene/depth.py` is already model-agnostic.
3. Cost on this VM: ~13.7 s/frame CPU vs Base's visibly lower cost —
   a 230-frame job's depth stage grows to ~52 min here; on a user
   machine with more cores this shrinks, but it is the slowest
   configuration of the three.

## Artifacts

- Scratch job: `jobs/room2_large` (Large depth maps in `depth/`,
  production-fused `points.bin` = 208,720 points, poses = room2_sc's,
  untouched).
- Logs: `experiments/large_baseline_spread.log` (0.0751171
  reproduced), `experiments/large_depth.log` (229 frames, 3,145 s),
  `experiments/large_fuse.log`, `experiments/large_spread.log`
  (0.0736659).
- Renders: `experiments/large_sc_top.png`,
  `experiments/large_room2_top.png`.
