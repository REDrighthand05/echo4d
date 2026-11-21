# ECHO4D Historical Research Branches Archive (Oct 2026)

This branch archives and consolidates all 64 historical research and experiment branches from late October 2026.
All commits remain reachable in GitHub git history and can be inspected or restored at any time via:
git checkout -b <branch_name> <commit_sha>

## Branch Catalog

| Branch Name | Commit SHA | Date | Author | Subject |
| :--- | :--- | :--- | :--- | :--- |
| research/echo4d-product-delivery-20261030 | 8e3ea75275 | Fri Oct 30 05:51:33 2026 +0800 | yang you | docs(product): bind geometry, timeline, viewer and long-term capture to one acceptance path |
| research/fountain-depth-ladder-20261031 | 6a5599b718 | Sat Oct 31 03:59:08 2026 +0800 | yang you | ci(reconstruction): run real Fountain 33/16/8 sensor-depth Mesh and five-view A/B |
| research/fountain-measured-prior-fusion-20261031 | 38d620f212 | Sat Oct 31 03:11:14 2026 +0800 | yang you | ci(mesh): run full prior RGB projection vs measured-only five-view geometry |
| research/fountain-rgb8-multiview-guard-20261031 | a3d07ab312 | Sat Oct 31 06:00:31 2026 +0800 | yang you | ci(reconstruction): execute real Fountain four-GLB CPU RGB visibility A/B |
| research/fountain16-plus8-rgb-cpu-20261031 | f682c01a12 | Sat Oct 31 05:00:10 2026 +0800 | yang you | ci(fountain): execute real 16 sensor plus 8 CPU RGB mesh comparison and upload five views |
| research/lounge-72sensor-24rgb-cpu-20261031 | 6a798b2b43 | Sat Oct 31 06:57:28 2026 +0800 | yang you | ci(mesh): run real Lounge 144/72/72+24 RGB CPU five-view GLB experiment |
| research/lounge-depth-gap-views-20261031 | 37e120e2da | Sat Oct 31 07:05:13 2026 +0800 | yang you | ci(mesh): execute real 144-vs-216 gap-targeted Lounge depth fusion |
| research/lounge-gap-targeted-rgb24-20261031 | 10a0e0f37a | Sat Oct 31 08:01:27 2026 +0800 | yang you | ci(lounge): avoid redundant intermediate runs and upload five real meshes |
| research/lounge-global-plane-tsdf-20261031 | ce9917ea1a | Sat Oct 31 08:59:13 2026 +0800 | yang you | ci(mesh): execute real Stanford Lounge global plane-constrained TSDF A/B |
| research/lounge-measured-first-carve-20261031 | 3abe9ad8af | Sat Oct 31 05:04:49 2026 +0800 | yang you | ci(mesh): run real Lounge scan-carving A/B and export five-view GLBs |
| research/lounge-planar-measured-holes-20261031 | 442b6c3882 | Sat Oct 31 08:02:46 2026 +0800 | yang you | research(mesh): add measured-support planar hole geometry A/B .github/workflows/lounge-planar-completion-real.yml |
| research/lounge-rgb-crossview24-20261031 | 2b0d26f097 | Sat Oct 31 08:59:13 2026 +0800 | yang you | ci(reconstruction): execute Lounge RGB cross-view Mesh ablation on real dataset |
| research/lounge-visibility-prior-20261031 | 81a553abac | Sat Oct 31 06:06:34 2026 +0800 | yang you | feat(mesh): lounge-visibility-prior-real.yml for real Lounge ray-aware PRIOR fusion |
| research/redwood-bedroom-depth-cutoff-ab-20261030 | eca8990db4 | Fri Oct 30 03:06:06 2026 +0800 | yang you | ci(real-room): run native 600-frame Redwood 3m vs 4.5m depth A/B and publish color Mesh |
| research/redwood-boardroom-prior-fusion-20261031 | 4f3e40aa90 | Sat Oct 31 03:06:11 2026 +0800 | yang you | fix(mesh): download real Redwood Google Drive binaries, reject HTML and validate poses before RGB-D |
| research/redwood-conservative-hole-caps-20261030 | 4202bd3c5c | Fri Oct 30 12:08:14 2026 +0800 | yang you | ci(real-room): run hole caps on published Redwood colored GLB and upload real result |
| research/redwood-depth-edge-guard-20261030 | caaff58cce | Fri Oct 30 06:07:48 2026 +0800 | yang you | ci(real-mesh): run original Redwood RGB-D edge-aware fusion and five-view A/B |
| research/redwood-measured-loop-pose-refusion-20261030 | 7ccb03fbad | Fri Oct 30 06:31:00 2026 +0800 | yang you | fix(test): index measured camera poses by original frame ID, matching real data |
| research/redwood-multiview-depth-consistency-20261030 | 6ae397b2c1 | Fri Oct 30 09:06:38 2026 +0800 | yang you | ci(mesh): run real 600-frame Redwood cross-view fusion and five-view A/B |
| research/redwood-observed-component-cleanup-20261030 | bafb24124e | Fri Oct 30 08:06:19 2026 +0800 | yang you | ci(real): run Redwood measured-mesh cleanup and export five-view comparison |
| research/redwood-occlusion-aware-depth-20261030 | 99623f5a0f | Fri Oct 30 07:00:49 2026 +0800 | yang you | fix(test): use original frame-indexed camera pose arrays in Redwood loop test |
| research/redwood-original-prior-detail-20261030 | 6ed37250a8 | Fri Oct 30 15:03:04 2026 +0800 | yang you | test(redwood): freeze three original and two free camera comparisons |
| research/redwood-prior-measured-hybrid-20261030 | 043664b5fb | Fri Oct 30 10:08:01 2026 +0800 | yang you | ci(real-scene): run prior-measured geometry on actual Redwood RGB-D |
| research/redwood-prior-measured-surface-fit-20261030 | bf306611e0 | Fri Oct 30 13:09:54 2026 +0800 | yang you | ci(redwood): run measured-supported prior surface fit on actual archived RGB-D geometry |
| research/redwood-prior-poisson-continuity-20261030 | 4d0382d575 | Fri Oct 30 16:10:13 2026 +0800 | yang you | ci(redwood): run real 500k prior Poisson continuity and upload 3+2 visual GLB artifacts |
| research/stanford-colmap-independent150-20261029 | bed14f985c | Fri Oct 30 02:01:46 2026 +0800 | yang you | research(room): run independent 150-frame COLMAP real RGB-D reconstruction |
| research/stanford-colmap-independent600-20261029 | 43cbd242d1 | Fri Oct 30 01:00:48 2026 +0800 | yang you | research(room): run independent 600-frame COLMAP RGB-D room after 300-frame visual recovery |
| research/stanford-colmap150-depth-pnp-20261030 | 318f52cb29 | Fri Oct 30 03:58:04 2026 +0800 | yang you | ci(room): execute real Stanford 150-frame PnP recovery and fixed-view mesh comparison |
| research/stanford-colmap150-exhaustive-20261030 | c67d737990 | Fri Oct 30 03:00:50 2026 +0800 | yang you | ci(room): execute real independent 150-frame COLMAP exhaustive matching and save five views |
| research/stanford-colmap300-da16-spatial-20261030 | f005a0b60d | Fri Oct 30 17:57:04 2026 +0800 | yang you | fix(room): describe eight additional RGB predictions accurately |
| research/stanford-colmap300-da2-track-filter-20261030 | 1a4afbc846 | Fri Oct 30 14:59:26 2026 +0800 | yang you | ci: execute real Stanford DA2 track-filter ablation with held-out sensor evaluation |
| research/stanford-colmap300-da8-spatial-20261030 | fb6795ba71 | Fri Oct 30 15:59:48 2026 +0800 | yang you | research(room): preserve three unscaled original RGB photos with the real five-view Mesh |
| research/stanford-colmap300-depth10-nested-20261030 | 48fc9e2e1f | Fri Oct 30 07:57:23 2026 +0800 | yang you | ci: run real Stanford 30/15/10 depth-only ablation from frozen 300 RGB COLMAP artifact |
| research/stanford-colmap300-depth10-pose-cover-20261030 | 5a7daeef50 | Fri Oct 30 08:59:03 2026 +0800 | yang you | research: use RGB-pose-only room-scale free orbit instead of camera-local close-ups |
| research/stanford-colmap300-depth10-sfm-cover-20261030 | 70fc238a9a | Fri Oct 30 09:52:47 2026 +0800 | yang you | ci(real): run SfM-observed ten-depth Stanford A/B and preserve five-view Mesh evidence |
| research/stanford-colmap300-depthanything2-20261030 | c635ef3fec | Fri Oct 30 13:55:28 2026 +0800 | yang you | ci: run real Stanford 8 measured + 2 CPU learned-depth Mesh A/B |
| research/stanford-colmap300-eight-measured-control-20261030 | fd537c52ae | Fri Oct 30 11:58:49 2026 +0800 | yang you | fix: use identical TSDF integration order for eight-only and 8+2 comparison |
| research/stanford-colmap300-four-depth-six-stereo-20261030 | 11715e2ec4 | Fri Oct 30 11:00:24 2026 +0800 | yang you | fix(room): store baseline five ray coverages once across both actual A/B panels |
| research/stanford-colmap300-mvs2-20261030 | 6029cc4cd8 | Fri Oct 30 12:59:39 2026 +0800 | yang you | ci: run real Stanford 8-measured vs 8+2 CPU RGB MVS geometry experiment |
| research/stanford-colmap300-rgb-pnp-rescue-20261030 | c0015f3eba | Fri Oct 30 07:01:09 2026 +0800 | yang you | research(room): track original COLMAP 3D observations with RGB optical flow before PnP |
| research/stanford-colmap300-temporal-overlap-20261030 | 8d4c3a1509 | Fri Oct 30 07:04:35 2026 +0800 | yang you | ci(room): run real independent early-overlap 300 RGB plus 30 measured-depth Mesh |
| research/stanford-colmap300rgb-75-30depth-20261030 | 6aca2a39b5 | Fri Oct 30 04:57:25 2026 +0800 | yang you | ci(room): execute real Stanford 300 RGB independent-pose depth ladder |
| research/stanford-colmap600rgb-150depth-20261030 | 75a7e19a65 | Fri Oct 30 04:02:09 2026 +0800 | yang you | ci(room): real 600 RGB and 150 measured depth ablation with no pose prior |
| research/stanford-colmap600rgb-75-30depth-20261030 | 0e7ce5cc6c | Fri Oct 30 04:09:01 2026 +0800 | yang you | ci(room): run real nested 150/75/30 depth fusion with 600 RGB SfM poses |
| research/stanford-da8-crossview-20261030 | 0632866860 | Fri Oct 30 17:02:03 2026 +0800 | yang you | ci(research): run same-input Stanford 8+8 measured-ray guard and upload five-view A/B |
| research/stanford-far-depth-coherence-20261030 | 434a6c26a3 | Fri Oct 30 02:00:44 2026 +0800 | yang you | ci(mesh): run coherent far-depth full 3000-frame A/B against unfiltered result |
| research/stanford-far-depth-refusion-20261030 | 6793d19e58 | Fri Oct 30 01:07:59 2026 +0800 | yang you | ci(mesh): run full real 3000-frame extended-depth TSDF and five-view A/B |
| research/stanford-lounge-spatial-prior-20261031 | ea6c7decb7 | Sat Oct 31 03:59:17 2026 +0800 | yang you | ci(mesh): run real Stanford Lounge 48 vs 144 spatial RGB-D with supplied scan PRIOR |
| research/stanford-measured-gap-patches-20261029 | 2ce9811124 | Fri Oct 30 00:02:56 2026 +0800 | yang you | fix(ci): interpolate GitHub workflow token and artifact run id |
| research/tum-desk-known-pose-open3d-20261030 | f4d79e841c | Fri Oct 30 20:02:48 2026 +0800 | yang you | ci(mesh): run real TUM Desk RGB-D fusion and export five-view GLB artifacts |
| research/tum-desk-lens-rectification-20261030 | 6918e76a20 | Fri Oct 30 21:03:07 2026 +0800 | yang you | ci(mesh): run real Freiburg1 Desk lens-corrected TSDF and five views |
| research/tum-desk-native640-24-plus-da12-20261030 | 7963392105 | Fri Oct 30 23:57:18 2026 +0800 | yang you | ci(desk): execute real 24 sensor plus 12 CPU RGB Mesh comparison and upload GLBs |
| research/tum-desk-native640-depth-ladder-20261030 | 353d049ee8 | Fri Oct 30 22:57:46 2026 +0800 | yang you | ci: run real Desk 57/24/12 depth Mesh and publish five-view artifacts |
| research/tum-desk-native640-island-prune-20261030 | 7ed24b48df | Fri Oct 30 23:05:39 2026 +0800 | yang you | fix(ci): render measured Desk mesh under Xvfb software OpenGL |
| research/tum-desk-native640-mvs23-20261031 | 58ce9716dc | Sat Oct 31 00:55:42 2026 +0800 | yang you | ci(reconstruction): run real Desk RGB-MVS replacement and export three GLBs |
| research/tum-desk-native640-tsdf-20261030 | 3055bcc2b3 | Fri Oct 30 22:04:21 2026 +0800 | yang you | test(mesh): verify native 640 to 320 ray calibration and fixed view poses |
| research/tum-room-loop-coverage-20261031 | 33d3d72775 | Sat Oct 31 01:09:31 2026 +0800 | yang you | feat(mesh): reject TUM Room measured cross-view foreground conflicts |
| research/tum-room-pose-depth36-20261031 | 3ccaff850a | Sat Oct 31 02:59:28 2026 +0800 | yang you | test(mesh): verify 36-frame pose cohort nested and original RGB-D fusion reused |
| research/tum-room-pose-spread-depth72-20261031 | 61b5e259f3 | Sat Oct 31 01:59:41 2026 +0800 | yang you | ci(room): run real 145-vs-72 GLB and five-view reconstruction |
| research/tum-xyz-depth-48-24-12-20261030 | 9cca4f418b | Fri Oct 30 18:58:39 2026 +0800 | yang you | ci: verify five real 48/24/12/6/3 Mesh levels and 3+2 views |
| research/tum-xyz-open3d-320-measured-20261030 | 466f2cbdd2 | Fri Oct 30 19:08:27 2026 +0800 | yang you | feat(mesh): fuse 96 interleaved measured TUM views against 48-frame baseline |
| research/tum-xyz-six-da6-local-bias-20261030 | 838b746c36 | Fri Oct 30 21:03:06 2026 +0800 | yang you | ci: isolate real-run concurrency and verify three exported colored meshes |
| research/tum-xyz-six-measured-six-da2-20261030 | cec2ddf3e0 | Fri Oct 30 19:56:58 2026 +0800 | yang you | ci(tum): execute real 6 measured + 6 CPU Depth Anything mesh and five-view comparison |
| research/tum-xyz-six-measured-twelve-ai-20261030 | 9078732aaf | Fri Oct 30 22:01:21 2026 +0800 | yang you | fix(research): initialize TUM importer command receipt before real reconstruction |
