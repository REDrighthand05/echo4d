# Second real room: Stanford Lounge

Open3D 0.19 supplies 3000 real RGB-D images, logged camera poses, and a
separately labeled pre-existing reconstruction (LoungeRGBDImages).
Run: PYTHONPATH=tools python tools/stanford_lounge_second_room.py
--frames 360 --cache /tmp/lounge-cache --output /tmp/lounge-output

OBSERVED-viewable.glb is fused from original depth/color + logged camera poses.
PREEXISTING-PRIOR-RGB.glb, if registration passes, contains ONLY old
reference-model faces, colored by nearby compatible measured RGB-D.
Unsupported colors remain neutral grey. Failed registration does not
silently paste wrong geometry. Images: 3 real-camera + 2 free-camera
views; A/B only when prior alignment passes. No source RGB-D archives
are committed. Human recognition, free walk, #74 and Desk FAIL unchanged.

Source: https://www.open3d.org/docs/0.19.0/python_api/open3d.data.LoungeRGBDImages.html
Open3D implementation (MIT): https://github.com/isl-org/Open3D
