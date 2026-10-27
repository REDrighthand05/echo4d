# Desk V2: fixed-Mesh residual diagnosis, 2026-10-27

This is an executed diagnostic, not a new reconstruction or a geometry PASS.
The original 15cm screen remains **FAIL**. No frames/pixels were removed from it.

## Frozen execution and independent checks

- Protocol and executable source committed at `c9e5b77e73f40f6f407d32d9282e45993aa9aa91` before residual execution.
- Actual tested merge checkout: `cf645157a84a1020960a51cbb114833e4020da5f`.
- [Real diagnostic run40404725134](https://github.com/REDrighthand05/echo4d/actions/runs/40404725134), job121859419015, completed/success; no COLMAP/FFmpeg/GPU or re-fusion.
- [Artifact12689728255](https://github.com/REDrighthand05/echo4d/actions/runs/40404725134/artifacts/12689728255), ZIP SHA256 `df328812537ea4e4667f36ec7cdd9f901ff5c7fe25a1b37022daf81c20538104`, independently downloaded and CRC/hash checked. Contains twelve original PNGs, lossless raster arrays, original input reports/Mesh, protocol and execution identity. Retention is 30 days; keep this artifact separately for later full reproduction.
- Protocol SHA256 `760a9ac42e895877e6854cab317d3a1b8b3e3f52807c2b9533fbee53e951b8a0`.
- Executed tool SHA256 `3316b8ed4ab6767a27338fc6f0901df941d10f61f2ec91e5ddd3e9f1d91c727c`.
- Exact report bytes SHA256 `45b972bdb0688de6a01cd4f2f142351c6c4fcc4e69d7614eb2ee47f8879d6b17`, preserved in the adjacent JSON.
- All twelve PNG and raster NPZ hashes independently checked. Recomputed counts, per-view RMSE, aggregate SSE and error-tail totals directly from saved measured/predicted arrays using a separate script, without calling the diagnostic partition implementation.
- Source-stage tests: four known-answer tests; full verify40404725171 Windows/Linux x Python3.12/3.13:4/4 PASS. These checks validate diagnostic arithmetic and execution, not physical truth.

## Observations

All169467 valid measured pixels retained;158851 overlap;10616 missing mesh surface.
Coverage93.73565355%; overlap RMSE0.1677790246288672m; MAE0.05789618122951465m.
The original values reproduce. Same Mesh SHA `320da61fcd421c0e55aa6fc014750aeaa3e6e2d98b607e9feaae268f83d2a939`.

| Predeclared partition | Observation |
| --- | --- |
| Absolute error >30cm, combining the two fixed signed tails |5463/158851 overlapping pixels (3.4391%) contribute91.2078% of all squared error. This is a tail diagnosis, never an exclusion rule.|
| Upper image third |57.7619% of squared error; grid and all twelve frames fixed before execution.|
| Depth-jump and invalid-neighbour proxy combined |17297/158851 overlapping pixels (10.8888%) contribute39.9734% of squared error.|
| Interior under the same proxy |60.0266% of squared error remains; the edge proxy cannot explain the full failure. Interior conditional RMSE13.7703cm is descriptive only, not a replacement score.|
| Measured Z 0.5–1m |Coverage99.1394%; conditional RMSE9.5312cm.|
| Measured Z 2–3m |Coverage76.4602%; conditional RMSE29.1624cm.|
| Measured Z 3–4m |Coverage46.2997%; conditional RMSE44.2111cm.|

The region, depth, and edge partitions overlap **across dimensions**; do not add
their contributions together. Within each dimension, counts and SSE conserve
the original totals. Signed residual is predicted Z minus measured Z: negative
tails correspond to a predicted surface in front of the measured depth, positive
tails to a predicted surface behind it. Neither sign proves a causal failure.

## What this resolves, and what it does not

The failure is concentrated in a small long tail and particular image/depth
regions. It is not justified to call every pixel inaccurate, or to attribute
the entire problem to depth discontinuities. This experiment cannot yet
distinguish pose drift, missing/incorrect fused surface, rasterization near
occlusions, depth outliers, or time misalignment as the cause.

Next: preregister a controlled projection/visibility and training-view support
check for the same saved inputs, retain all twelve original test views, and
change at most one reconstruction factor per new comparison. Do not tune on
these holdouts and then call the same set independent validation; any selected
fix needs a separately frozen validation scene/split.

The RGB-to-predicted-depth C branch remains a separate missing capability.
This diagnosis used physical sensor depth, estimated RGB camera poses and
previously fixed sensor-based metric scale; it does not establish RGB-only
dense reconstruction, independent scanned-surface truth or Hi3D improvement.
