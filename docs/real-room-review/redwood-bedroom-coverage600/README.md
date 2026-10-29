# Redwood Bedroom: 600-frame measured coverage vs 600 uniform frames

Real original 640x480 RGB-D and logged metric camera poses, Open3D
ScalableTSDFVolume 2.5cm voxel and 4.5m depth limit on both sides.
The new 600-frame subset is selected from actual measured world-space
surface coverage; 60 evenly spaced anchors preserve trajectory range.
The frozen baseline observed-only Mesh is used ONLY for A/B images.
Neither baseline nor provided reference scan is fused into the output.
In three recorded views: left = real RGB, middle = original 600,
right = coverage-selected 600. Two free views: left = old, right = new.
No claim of visual improvement or #74 PASS without human review.
#74 FAIL and old Desk 16.7779cm > 15cm FAIL unchanged.
Dataset: https://redwood-data.org/indoor_lidar_rgbd/
Baseline: https://github.com/REDrighthand05/echo4d/actions/runs/40778686280
