# Real Redwood Bedroom, labeled PRIOR-assisted geometry
Left side of each comparison: 600-frame observed-only Mesh.
Right side: observed-only Mesh plus triangles from the supplied
pre-existing bedroom.ply where observed geometry has no nearby
support. For actual-camera views, each side also shows the photo.
Cyan/orange source-mask GLB: cyan = measured RGB-D reconstruction,
orange = explicitly pre-existing prior geometry.
This is NOT independent reconstruction, NOT a 100% result,
and does NOT pass #74 without human review.
Original observed-only baseline: ../redwood-bedroom-dense600/


## Actual visual review (2026-10-29)

**Negative result; do not present as a quality win.** The three real-camera
views still show a recognizable writing desk and furniture silhouettes, but
the geometry is coarse, missing thin structures, and the two free-camera
views remain substantially incomplete. Side-by-side prior-assisted images
are visually almost identical to the observed-only version. The reference
registered at 0.9995 ICP fitness and 0.0374 m inlier RMSE, yet only 791 of
170,000 reference faces survived the 0.09 m unsupported-geometry gate.
In the actual 3+2 preview PNGs, >10/255 pixel-channel differences affect
0.0%, 0.016%, 0.016% of the three same-camera images and 0.951%, 0.007%
of the two free-camera images. This is a description of *change*, not
a visual-quality metric or an independent geometry-accuracy test.

The 600-frame original has 553,208 triangles, but the published observed
GLB is reduced to 85,000 triangles and the preview renderer further
reduces to 30,000 triangles. Next geometry action: preserve the original
dense measured mesh and render a substantially higher-resolution copy
on the exact same 3+2 cameras. Do not optimize prior gap thresholds
on this near-identical baseline.
