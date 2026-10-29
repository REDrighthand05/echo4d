# Actual real Redwood Bedroom: native observed 553,208-face Mesh

600 measured RGB-D frames at 640x480 with recorded camera poses;
original Open3D fusion from successful run 40778686280.
The colored GLB has all 553,208 observed triangles (287,882 vertices).
It is NOT the pre-existing supplied bedroom reference model.

Photo-vs-mesh: LEFT real source photo, CENTER prior 30,000-face
preview, RIGHT the same observed Mesh at full density using
CPU raycasting. Free cameras: LEFT prior 30,000-face preview,
RIGHT full observed Mesh; neither is an original input camera.

**Actual human visual review, 2026-10-29:**
Full-density views clearly restore the writing desk's shelving,
bedcover texture and furniture contours relative to the coarse
30k preview. The first camera's desk and the final camera's bed
are more recognizable. Some thin objects and boundaries remain
jagged, with black holes and detached fragments. Both free
views still have major missing/occluded geometry. This is a
meaningful visual-detail gain, NOT a completeness gain and
NOT a successful stable free-roaming room.

This is a display/export fix; the underlying 600-frame geometry
is unchanged. The separate supplied reference PRIOR was not used.
Issue #74 and old Desk V2 FAIL remain unchanged.

Native-density run:
https://github.com/REDrighthand05/echo4d/actions/runs/40792115358
Source: https://redwood-data.org/indoor_lidar_rgbd/
