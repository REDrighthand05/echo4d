# Continuity cleanup — experiment report (2026-10-31)

**The user's idea:** a point-cloud file has no continuity concept — points are
just coordinates. Give every point an entity volume, enlarge it until ~80% of
the points touch each other, then delete the discontinuous remainder. The 80%
is an adjustable strictness parameter （严格度）.

## Input

The aligned K=2 trim cloud, 208,666 points. Note: by the time this ran, the
concurrent up-alignment work had already re-issued `room_scene_trim.html` with
a rigidly rotated (upright) version of the same cloud — verified as a pure
rotation (Kabsch max residual 7e-7, colours and point order identical). That
**aligned** cloud is what was used here, extracted from the current
`room_scene_trim.html`; it lives at `jobs/room2_clean_exp/points.bin`.

## Method (`experiments/continuity_clean.py`, strictness is a CLI parameter)

- Voxelise at cell size `s`; two points are "in contact" when their occupied
  voxels are 26-neighbours (face/edge/corner).
- Label connected components of occupied voxels (union via sparse graph),
  map back to points.
- Sweep `s`; for strictness `q`, keep the largest connected component at the
  smallest `s` whose coverage reaches `q`. Everything else is dropped.
- Deterministic selection only — points are kept or dropped, never moved or
  averaged (same keep/drop family as the corroboration trim).

## Sweep (main-component coverage vs cell size, key rows)

| s | coverage | components | 2nd largest |
|---|---|---|---|
| 0.041 | 0.05% | 97,672 | 74 |
| 0.048 | 3.2% | 30,761 | 2,541 |
| 0.050 | 42.6% | 18,303 | 4,895 |
| 0.052 | 74.6% | 10,279 | 2,371 |
| **0.0542** | **88.0%** | 5,878 | 484 |
| 0.0564 | 92.6% | 3,685 | 531 |
| 0.0587 | 94.5% | 2,453 | 579 |
| 0.066 | 97.0% | 1,017 | 585 |
| 0.095 | 99.0% | 177 | 394 |
| 0.454 | 99.91% | 3 | 175 |
| 0.936 | 100% | 1 | — |

**Honest degeneracy note:** the transition is a percolation jump, not a ramp —
coverage goes 3% → 43% → 75% → 88% across s = 0.048 → 0.054. There is no cell
size at which coverage sits at exactly 80%: an 80/20 split falls *inside* the
jump. Per the guard rule, q=0.80 takes the first crossing from above:
s = 0.0542, which keeps 88.0% and drops 12.0%. The looser bracket (s = 0.052)
keeps 74.6% / drops 25.4%. His nominal "drop the discontinuous 20%" lies
between those two settings. Also note the far tail: ~181 points (0.09%) are
so remote they only merge at s ≈ 0.94 — pure outliers by any setting.

## Variants

| variant | s | kept | dropped | dropped set structure |
|---|---|---|---|---|
| q=0.90 | 0.0564 | 193,328 (92.6%) | 15,338 (7.4%) | 3,684 comps, median size 2, largest chunk 531 |
| **q=0.80** | **0.0542** | **183,691 (88.0%)** | **24,975 (12.0%)** | 5,877 comps, median size 2, largest chunk 484; only 11.2% of dropped points sit in chunks ≥100 |
| q=0.70 | 0.0520 | 155,670 (74.6%) | 52,996 (25.4%) | 10,278 comps, **largest chunk 2,371**, 19.7% of dropped in chunks ≥100 |

## Visual verdict (renders inspected by me)

- **Chair area (his photo): YES, the scatter is gone.** A dropped-only render
  of the chair zoom (`jobs/room2_clean_exp/dropped80_chair.png`) shows exactly
  what was removed there: lone confetti floating in empty space around/above
  the brown chair mass — no surface in it. In the before/after chair zooms
  (`clean_before_chair.png` / `clean80_chair.png` in the goal files) the brown
  surface stays dense and intact while the haze above and around it thins out
  and its edges read cleaner.
- **Whole room (q=0.80): no real surface eaten.** Perspective and top views
  before/after (`clean_before_*` / `clean80_*`) show the identical room —
  walls, floor, blue object, chair — with a slightly crisper silhouette;
  the removed 12% is overwhelmingly speckle (median dropped component = 2
  points).
- **q=0.70 warning:** the 2,371-point largest dropped chunk and the jump in
  ≥100-point chunks say this setting starts cutting real, weakly-attached
  surface pieces (thin/separated furniture parts are the first to go).
  q=0.80 does not show this problem; q=0.90 is even safer but removes less
  noise (7.4%).

## Deliverables

- `room_scene_clean.html` (goal files) — q=0.80 points (183,691) in the same
  standalone viewer; mesh payloads byte-identical to the aligned trim viewer,
  so the comparison is points-only.
- Previews in goal files: `clean80_view.png`, `clean80_top.png`,
  `clean_before_view.png`, `clean_before_top.png`, `clean80_chair.png`,
  `clean_before_chair.png` (camera-up-aligned framing, shared before/after).
- Script: `experiments/continuity_clean.py`
  (`--strictness 0.80`, `--fine-below` for a dense sweep of the jump zone),
  ready to become a pipeline stage if approved; variant clouds +
  `continuity_summary.json` in `jobs/room2_clean_exp/`.
