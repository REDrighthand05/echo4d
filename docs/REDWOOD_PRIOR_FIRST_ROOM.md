# Full prior geometry first: real Redwood Bedroom

Direct experiment: unlike prior-gapfill's 791 retained triangles, use
the entire official **pre-existing** Bedroom room model as the scene
geometry. Register it to the independent 720-frame RGB-D Mesh with
stock Open3D ICP. Transfer colors only from nearby observed surfaces
whose normals agree. Unknown areas are neutral gray; never pretend
they were captured in the video. The result is a **prior-based scene**,
not RGB-video recovery.

Compare the identical three real-camera and two shifted free-camera
views of measured-only versus full-prior. The output GLB contains the
complete existing model with observed colors where available; source
mask makes unsupported color visible. The separate RGB-D-only Mesh is
never overwritten. This experiment is about whether a mature
pre-existing geometry source gives a usable free-walk baseline,
not whether we independently reconstruct from less information.

Run:
    pip install -r requirements.lock
    pip install --no-deps -e .
    pip install 'open3d==0.19.0' 'trimesh>=4,<5'
    PYTHONPATH=tools python tools/redwood_prior_first_room.py \
      --observed docs/real-room-review/redwood-bedroom-gap-guided/gap-guided-observed.glb \
      --cache /tmp/redwood --output /tmp/prior-first

Dataset: https://redwood-data.org/indoor_lidar_rgbd/
Official Open3D source: https://github.com/isl-org/Open3D
The modality used to create the pre-existing supplied bedroom.ply
is not asserted to be an independently surveyed laser scan.
#74 and historical Desk V2 FAIL remain unchanged.
