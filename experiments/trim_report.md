# Intersection-only trim (the user's "straight knife") — room2_sc

**Method.** Every previous attempt tried to *move* points (poses, cameras,
depth fits, joint refinement — all refuted). This one *deletes* them. The
cloud is one depth sheet per frame stitched together; a point is kept only
if the surface it sits on was independently seen by at least K distinct
frames. Implementation (`experiments/trim_fuse.py`): production
`stage_fuse` back-projection reproduced exactly (stride 4, per-frame (a,b)
anchor fits, validity + 0.18 depth-edge rejection, per-axis 1–99
percentile trim), with a per-point source-frame id; one new selection rule
before the production voxel_pick — a point survives only if its voxel at
the spread yardstick's size (extent/110 = 0.15427) contains points from
≥ K distinct frames; then production voxel_pick (extent/220,
closest-to-centroid, no averaging) as usual. Nothing is averaged or
moved; unproven points are cut.

**Replication proof.** A K=0 run of the trim script (filter disabled)
reproduced jobs/room2_sc/points.bin **byte-identically** (282,514 points).
The two variants therefore differ from production by the selection rule
alone, and the saved baseline frame ids align with the production cloud.

## Counts

| | final points | cut vs 282,514 |
|---|---|---|
| room2_sc (production) | 282,514 | — |
| K=2 (jobs/room2_trim2) | **208,666** | −73,848 (−26.1%) |
| K=3 (jobs/room2_trim3) | **164,522** | −117,992 (−41.8%) |

Dense stage (identical inputs for both variants): 29,088,660
back-projected points → 27,717,111 after the percentile trim → 92,503
yardstick voxels.

- K=2: 51,005 voxels (55.1%) have ≥2 frames; they hold 25,220,360 points
  (91.0% of the dense cloud — the single-frame voxels are mostly sparse
  fringe with few points each, but they sprawl over 44.9% of all cells
  and supply 26.1% of the delivered cloud's picks).
- K=3: 35,694 voxels (38.6%) have ≥3 frames; 23,304,926 points (84.1%).

## Spread

- Official yardstick (`experiments/measure_spread.py`): **0.0751171 for
  baseline, K=2 and K=3 alike.** Expected and stated plainly: that script
  re-derives its sample from the depth maps and poses, so point selection
  is invisible to it. It certifies the inputs didn't change, nothing more.
- The same statistic applied to the **delivered clouds** (frame ids kept
  through voxel_pick; script `experiments/trim_spread.py`): baseline
  0.070509 → K=2 **0.069385** (−1.6%) → K=3 0.070254 (−0.4%). The
  corroborated core's tightness barely moves — the knife cuts fringe,
  not core. Direct evidence: of the baseline's 165,439 delivered points
  sitting in multi-frame voxels, K=2 keeps 159,565 (**96.5%**) while
  cutting 26.1% of the cloud overall.

## Renders (shared framing — regenerated baseline top view is
pixel-identical to the delivered experiments/sc_room2_top.png)

- Top: `experiments/trim_sc_top.png` (baseline),
  `experiments/trim2_top.png`, `experiments/trim3_top.png`
- Perspective (v6-preview angle, shared center/radius from baseline):
  `experiments/trim_sc_view.png`, `experiments/trim2_view.png`,
  `experiments/trim3_view.png`

## Visual verdict (honest)

**The fringes do disappear.** In perspective, the baseline wears a wide
skirt: a fan of splayed sheets across the front-bottom, thin slabs
floating above the roof line, wispy curtains on both flanks. At K=2 the
fan and the floating slabs are essentially gone and the silhouette pulls
in to roughly one room; at K=3 the room block stands almost alone and the
left wall reads as one flat, straight plane. In top view, the long
diagonal streaks reaching into the top-left corner and the "grass" along
the top edge are heavily thinned at K=2 and further at K=3.

**What the knife cannot do:** it removes what sticks out *past* the
intersections; it cannot un-bend what the frames agree on. The surviving
core still carries the known warp — residual diagonal streak bundles in
the top view's upper-left and bottom-left are corroborated sheets that
two or more frames place in the same wrong position (the
viewing-direction-correlated intra-frame depth bias localized by the
joint-refinement run). Walls read *cleaner*, not *straighter*, in the
core. A fuzzy mass at the front-bottom also survives both K values: it
sits close to the camera route, many frames see it, so it is
corroborated clutter, not fringe.

**Cost / which K.** K=2 is the sweet spot: −26% density, 96.5% of the
corroborated core retained, fringe classes visibly eliminated. K=3
(−42%) buys a slightly tighter silhouette but starts eating real
surfaces — ragged top edges, thinning on surfaces seen from few route
positions. Recommendation: **K=2**.

## K=2 deliverables (point cloud + voxel model, ready to ship)

In `jobs/room2_trim2/`: `points.bin` / `scene_points.ply` (208,666),
production mesh (`mesh.ply`: 541,906 verts / 1,056,649 tris; web bins at
250,000 tris) and the frozen voxel model (`voxel_model.ply` / `.glb`:
208,666 cubes, 2,503,992 tris). Per-point frame ids:
`points_frame_ids.npy` in each trim job; baseline ids in
`experiments/trim_baseline_fids.npy`.

## Reproduce

```
.venv/bin/python experiments/measure_spread.py jobs/room2_sc   # baseline 0.0751171, voxel 0.15426816915966748
.venv/bin/python experiments/trim_fuse.py jobs/room2_trim2 2 0.15426816915966748 --save-baseline-fids experiments/trim_baseline_fids.npy
.venv/bin/python experiments/trim_fuse.py jobs/room2_trim3 3 0.15426816915966748
.venv/bin/python experiments/render_top.py jobs/room2 experiments/trim_ref_room2_top.png jobs/room2_sc experiments/trim_sc_top.png jobs/room2_trim2 experiments/trim2_top.png jobs/room2_trim3 experiments/trim3_top.png
.venv/bin/python experiments/trim_view.py jobs/room2_sc jobs/room2_sc experiments/trim_sc_view.png jobs/room2_trim2 experiments/trim2_view.png jobs/room2_trim3 experiments/trim3_view.png
.venv/bin/python -m echo_scene.pipeline mesh jobs/room2_trim2
.venv/bin/python -m echo_scene.pipeline voxel jobs/room2_trim2
```

jobs/room2, jobs/room2_sc, jobs/room2_joint and everything under
echo_scene/ were not modified.
