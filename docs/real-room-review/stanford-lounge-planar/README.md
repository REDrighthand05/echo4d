# Stanford Lounge: observed geometry with labeled planar gap inference

Input: **actual** Stanford Lounge 640x480 RGB-D TSDF, 360 measured frames,
calibrated logged poses, previously published on the
research/stanford-lounge-second-real-room-20261029 branch. This iteration
uses its independently fused 84,999-face *viewable preview* GLB, not the
3.15-million-face pre-existing reference mesh and not the original full
426,789-face native PLY.

Method: fit large floor/wall planes to the observed vertex normals; fill
only nearby small interior planar holes, with no extension beyond measured
support footprint. Original measured faces are preserved, inferred faces
are separate. Orange = inferred, cyan = measured. Inferred is **not** sensor
truth, a watertight room, or proof of geometric accuracy.

To run with the existing published real observation:

    pip install -r requirements.lock
    pip install --no-deps -e .
    pip install 'trimesh>=4,<5'
    PYTHONPATH=tools python tools/stanford_planar_support.py \
      --source docs/real-room-review/stanford-lounge-second-room/OBSERVED-viewable.glb \
      --output /tmp/stanford-planar

Results: OBSERVED-plus-INFERRED.glb, INFERRED-planar-only.glb,
OBSERVED-vs-INFERRED-provenance.glb, result.json. CI also downloads the
original Lounge RGB-D archive to render 3 original camera images plus 2
fixed free viewpoints using Open3D's full-mesh raycaster. The image comparison
is not a self-graded acceptance test.

No #74 status change. Old Desk 16.7779 cm > 15 cm remains FAIL.
