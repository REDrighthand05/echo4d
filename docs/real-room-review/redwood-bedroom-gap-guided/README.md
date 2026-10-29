# Redwood Bedroom: real missing-surface depth fusion

640x480 original real Redwood RGB-D, measured depth, recorded poses.
OLD: observed-only 600-frame full-density 553,208-face Mesh.
NEW: same TSDF algorithm and 600 frames plus up to 120 real
intermediate frames selected for measured surfaces missing from OLD.
OLD Mesh is used ONLY for selecting frames, not fused into NEW.
Existing supplied reference/laser scan is NOT an input.
In the three real-camera comparisons, left is the original RGB,
center is the old observed Mesh, right is the newly fused Mesh.
In the two free views, left is old and right is new.
These pictures and GLB require human review; no automatic pass.
Issue #74 remains FAIL, old Desk V2 FAIL unchanged.
Source: https://redwood-data.org/indoor_lidar_rgbd/
Baseline: https://github.com/REDrighthand05/echo4d/tree/research/redwood-dense-native-mesh-20261029/docs/real-room-review/redwood-bedroom-native600

## Actual five-view inspection, 2026-10-29 (negative visual result)

Real first camera: writing desk recognizable in both OLD and NEW; no
clear improvement to its shelving or chair. Middle: furniture/cabinet and
curtain geometry remain coarse; no meaningful visual difference. Final:
bed and bedside table remain recognizable, but thin lamp and edges are
still broken. Both free cameras still show severe black voids and incomplete
surfaces; side-by-side OLD and NEW look nearly identical. This is NOT
stable free-roaming geometry.

600 -> 720 real measured RGB-D frames; faces 553,208 -> 561,887
(+8,679, 1.57%). Native 320x240 raycast covered pixels:
recorded 76,788 -> 76,793; 76,491 -> 76,531; 76,770 -> 76,781;
free 74,840 -> 75,260 and 71,828 -> 72,074. Small coverage increases
are NOT proof of corrected surfaces or independent geometric accuracy.
No visual pass; #74 and Desk V2 FAIL remain unchanged.

Next change should not simply add more frames. Test geometry hole
boundaries / disconnected fragments using a different mature surface
fusion or confidence-aware integration, with the SAME real 3+2 cameras.
