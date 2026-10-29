"""Blender integration assertion after reopening the exported native scene."""
import bpy
import json
import math
import sys
from pathlib import Path
from mathutils import Vector

root = Path(sys.argv[sys.argv.index('--')+1])
bpy.ops.wm.open_mainfile(filepath=str(root/'scene.blend'), load_ui=False)
scene = bpy.context.scene
assert scene.unit_settings.system == 'NONE'
assert scene['echo4d_scale'] == 'unknown'
assert scene.camera is not None
timeline = json.loads(bpy.data.texts['echo4d-camera-timeline.json'].as_string())
assert len(timeline['frames']) == 3
for entry in timeline['frames']:
    frame = 1+30*entry['t_s']
    scene.frame_set(int(frame), subframe=frame-int(frame))
    matrix = entry['camera_to_sfm_world']
    x,y,z = matrix[12:15]
    assert (scene.camera.location-Vector((x,-z,y))).length < 1e-5
    assert (scene.camera.matrix_world.to_quaternion() @ Vector((0,0,-1))-Vector((0,-1,0))).length < 1e-5
assert any(m.use_nodes for m in bpy.data.materials)
(root/'timeline-check.json').write_text(json.dumps({'status':'PASS', 'frames':3, 'units':'unknown', 'reopened':True}))
