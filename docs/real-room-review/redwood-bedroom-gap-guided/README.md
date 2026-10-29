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
