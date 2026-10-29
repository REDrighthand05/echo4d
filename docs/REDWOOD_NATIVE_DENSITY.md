# Native-density measured Redwood Bedroom

Use actual 600-frame Redwood RGB-D Open3D reconstruction from
https://github.com/REDrighthand05/echo4d/actions/runs/40778686280

Instead of discarding measured faces to make the 85,000-face GLB and
30,000-face preview, preserve all 553,208 observed triangles and use
Open3D CPU raycasting to produce the same three recorded-camera and two
free-camera views at 320x240. Display the old and new views side by side.
The official dataset supplies camera poses. Existing pre-built bedroom
Mesh is never fused. No invented geometry or hidden RGB texture.

Source and licensing: https://redwood-data.org/indoor_lidar_rgbd/
Engine: https://github.com/isl-org/Open3D

This is an actual mesh-detail visualization test, not a new reconstruction
or a claim of improved completeness. Issue #74 and Desk V2 FAIL unchanged.
