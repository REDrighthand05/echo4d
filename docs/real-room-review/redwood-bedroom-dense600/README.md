# Redwood Bedroom: 600 real RGB-D frames, fixed review cameras

Same 640x480 original RGB-D, logged poses, 0.025m voxel, stock
Open3D TSDF as the 360-frame baseline. Only source frame count
changes: 600 uniformly spread frames rather than 360.
Three photo/mesh and two free-camera views use the SAME recorded
comparison camera indices and free-camera definitions as the
360-frame baseline. Do not mistake 5 review images for a full
3D completeness test. No supplied prior used in fusion.

Baseline: ../redwood-bedroom/
Data: https://redwood-data.org/indoor_lidar_rgbd/
