# Joint route+depth refinement on room2_sc — report

Date: 2026-10-31. Job: `jobs/room2_joint` (scratch copy of `jobs/room2_sc`;
room2 and room2_sc untouched — verified by mtime and content).
Scripts: `experiments/joint_prep.py`, `experiments/joint_optimize.py`,
`experiments/joint_check_jac.py` (Jacobian verification). Yardstick:
`experiments/measure_spread.py`; renders: `experiments/render_top.py`
(joint framing over room2_sc + room2_joint).

## Question

Both single-side families are measured and exhausted: depth-side
corrections (affine (a,b), then per-frame deformation fields — room1_big)
and pose-side tests (global BA: nothing moved; shared-camera remap:
spread −1.9 %, no visible straightening). The BA report named the
remaining lever: a **joint** objective where the dense depth field enters
the pose optimization — route and depth are continuous, they overlap,
and the overlap (the same surfaces seen from multiple route points) is
the constraint. This run builds exactly that, with poses as the only
variables.

## Design (as run)

- Base: room2_sc — one shared camera (f = 1690.20 px), 230 registered
  poses. (One data quirk found: `frame_00163` is registered but has no
  depth map — room2's depth set differs by one frame — so it takes no
  part in term 1; production fuse likewise uses 229 frames.)
- The fuse **(a,b) depth fits stay FIXED** (identical recipe to
  production: global a = 8.2701, b = 2.5448; 214 of 230 frames carry
  their own fit). Freeing (a,b) jointly with poses is the noted v2
  option; it was not taken (see Verdict).
- Variables: per-frame pose corrections — rotation increment
  (axis-angle, 3) + translation increment (3), applied in camera
  coordinates, Xc' = ΔR·(R·Xw + t) + Δt. 1,380 variables.
- Term 1 — co-visibility depth consistency (197,784 residuals): pairs
  are the COLMAP model's literal co-visibility (1,600 unordered pairs
  sharing ≥ 25 sparse points, both directions → 3,193 ordered pairs).
  Points sampled from frame i's depth map (stride 10, fuse validity +
  edge rules), moved into frame j with corrected poses, projected;
  j's depth map is bilinearly sampled at that pixel and both sides are
  compared in Z via the fixed (a,b). Targets re-sampled between outer
  rounds, fixed inside a solve (the optimize_refine alternating
  pattern). Loose agreement gates at prep/refresh are data hygiene
  only, in the prep_refine style.
- Term 2 — sparse-anchor reprojection guard (44,301 observations,
  deterministically subsampled from 265,803): COLMAP sparse points
  must keep projecting onto their observed pixels (SIMPLE_RADIAL
  distortion included), so the trajectory cannot drift to cheat
  term 1.
- Term 3 — Tikhonov pull of corrections toward 0 (weight 1.0).
- Solver: scipy `least_squares` (TRF, x_scale='jac'), analytic sparse
  Jacobian, first-order in the increments, re-linearized each outer
  round as increments are folded exactly into the base poses.
  **Jacobian verified against central finite differences**
  (joint_check_jac.py): worst relative error 1.7e-3 over
  source/target/anchor rotation+translation columns at |x| ~ 1e-3 —
  consistent with the first-order approximation; no wiring errors.
  Deterministic throughout: no RANSAC, no learned components, no
  voting/averaging merges. Total residuals ≈ 288k.

## Run 1 — weak guard (W2 = 0.05/px, no bounds): the cheat, measured

Term-1 RMS fell 0.36706 → 0.33366 (−9.1 %) in one round — but the
guard's own yardstick shows what paid for it: mean anchor reprojection
error **0.850 → 2.150 px**, median rotation correction 0.40°, max
**10.2°**, max translation 1.45 units. The optimizer bent the
trajectory (the trustworthy party — 0.78–0.85 px reprojection) to
follow the depth maps' biases (the suspect party). This configuration
was rejected on the spot; original poses restored (byte-content
verified). It exists in the log (`experiments/joint_optimize.log`) as
the measurement of how the objective behaves without a real guard.

## Run 2 — guarded (W2 = 0.5/px, per-round bounds ±0.86° / ±0.04 units)

The guard held end to end: mean anchor reprojection **0.8496 →
0.8477 px** (no growth). Corrections are small: median rotation
**0.031°**, median translation **0.0017 units**. What they bought:

| quantity | before | after | change |
|---|---|---|---|
| Term-1 RMS, round 0 (objective) | 0.36706 | 0.36568 | −0.4 % |
| Term-1 RMS, rounds 1–2 | 0.33716 / 0.35860 at refresh | 0.33706 / 0.35817 | ≈ 0 — refresh noise (±0.02–0.03) swamps solve gains |
| Cross-frame spread (measure_spread) | **0.0751171** | **0.0754739** | **+0.47 % (noise, wrong sign)** |
| Fused points (production fuse) | 282,514 | 281,663 | −0.3 % |
| Cross-frame voxels / sampled points | 45,896 / 6,875,368 | 45,930 / 6,874,933 | ≈ same support |
| Top-view footprint (joint framing) | 195,698 px | 194,598 px | −0.6 % |

The largest corrections are diagnostic, not progress: only 7 frames
exceed 0.5° of rotation, and they are exactly the sparsely-anchored
frames (median 7 subsampled anchors vs 148 for all frames;
max-correction frame_00102 has 3). Well-anchored frames — the bulk of
the route — essentially do not move (median 0.031°), because for them
the objective has nothing to offer inside the sparse trust region.

Renders: `experiments/joint_sc_top.png` (before) and
`experiments/joint_room2_top.png` (after), same joint framing, 87.5 %
of pixels identical. Side by side they are the same picture: same room
mass, same splayed diagonal sheets, same feathered fringes — the warp
signature is unchanged everywhere.

## Verdict

**The joint pose refinement does not straighten the room — the last
shape lever is refuted with evidence.** With poses held to the
sparse-geometry trust region (anchor reprojection flat at 0.85 px),
depth consistency is not purchasable: −0.4 % on the optimizer's own
objective, +0.5 % noise on the independent voxel-spread yardstick, no
visible change. The only way the joint objective improves materially
(run 1) is by deforming the trajectory *against* the sparse evidence —
fitting the poses to the depth maps, not the depth to the world.

Combined with everything before it — global BA (nothing to move),
shared camera (−1.9 %, cosmetic), depth-side affine + deformation
fields (families exhausted on room1_big) — the picture is now
consistent from every side: **the remaining warp lives inside the
monocular depth maps themselves** (bias correlated with viewing
direction, varying within each frame), in a place that no rigid
per-frame pose correction and no per-frame depth re-parametrization
tried so far can express. What is left is not another refinement of
this objective family: it would take either a depth source with
cross-view consistency built in (measured multi-view depth — already
shown blocked on this VM: PatchMatch needs CUDA, SGBM inapplicable to
forward-motion footage), or accepting the warp as the current
pipeline's ceiling.

Noted but not run: freeing (a,b) jointly with poses. Rationale for
skipping: the (a,b) family is already exhausted standalone (0.5 %
gain, converged), and run 1 shows what extra pose-side freedom gets
spent on. If it is ever run, it needs run 2's guard and bounds as the
starting configuration, plus the per-frame field result as a warning:
on room1_big the field improved anchor fit 27 % while cross-frame
spread got slightly *worse* — fitting harder and agreeing less is
this problem's signature failure mode.

## Artifacts

- Corrected poses: `jobs/room2_joint/poses.json` (original preserved
  at `jobs/room2_joint/joint/poses_original.json`; per-run stats at
  `jobs/room2_joint/joint/joint_result_w05.json`).
- Fused cloud from corrected poses: `jobs/room2_joint/points.bin`
  (281,663 points) — produced by the unmodified production fuse.
- Logs: `experiments/joint_prep.log`, `experiments/joint_optimize.log`
  (run 1), `experiments/joint_optimize_w05.log` (run 2),
  `experiments/joint_baseline_spread.log` (baseline 0.0751171
  reproduced on the scratch copy before any change),
  `experiments/joint_spread_after.log`.
