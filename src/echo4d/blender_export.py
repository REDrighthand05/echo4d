"""Import a real GLB into Blender, save native .blend, and verify reopening."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy
import math
from mathutils import Matrix, Vector

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def geometry():
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh was imported")
    for obj in meshes:
        obj.data.calc_loop_triangles()
    return {
        "mesh_objects": len(meshes),
        "vertices": sum(len(obj.data.vertices) for obj in meshes),
        "triangles": sum(len(obj.data.loop_triangles) for obj in meshes),
        "materials": sum(len(obj.data.materials) for obj in meshes),
        "vertex_color_layers": sum(len(obj.data.color_attributes) for obj in meshes),
    }

def add_timeline(path):
    """Retain source seconds; COLMAP/OpenCV camera axes -> Blender camera axes."""
    data = json.loads(Path(path).read_text())
    frames = data['frames']
    if not frames:
        raise ValueError('Empty recorded camera timeline')
    camera_data = bpy.data.cameras.new('Recorded camera')
    camera = bpy.data.objects.new('Recorded camera', camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.rotation_mode = 'QUATERNION'
    axes = Matrix.Diagonal((1., -1., -1., 1.))
    # glTF import rotates mesh world axes from Y-up to Blender Z-up.
    # OpenMVS writes native XYZ directly; apply the identical transform to poses.
    world_axes = Matrix(((1.,0.,0.,0.),(0.,0.,-1.,0.),(0.,1.,0.,0.),(0.,0.,0.,1.)))
    previous = -1.
    previous_q = None
    for entry in frames:
        seconds = float(entry['t_s'])
        values = entry['camera_to_sfm_world']
        if not math.isfinite(seconds) or seconds < 0 or seconds <= previous or len(values) != 16 or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid camera timeline')
        previous = seconds
        pose = Matrix([[values[c*4+r] for c in range(4)] for r in range(4)])
        camera.matrix_world = world_axes @ pose @ axes
        if previous_q is not None:
            camera.rotation_quaternion.make_compatible(previous_q)
        previous_q = camera.rotation_quaternion.copy()
        frame = 1+seconds*30
        camera.keyframe_insert(data_path='location', frame=frame)
        camera.keyframe_insert(data_path='rotation_quaternion', frame=frame)
        calibration = data.get('cameras', {}).get(str(entry['camera_id']))
        if calibration:
            width, height, fx, fy, cx, cy = calibration
            camera_data.sensor_fit = 'HORIZONTAL'
            camera_data.lens = fx/width*camera_data.sensor_width
            camera_data.shift_x = (.5*width-cx)/width
            camera_data.shift_y = (cy-.5*height)/width*(fx/fy)
            for property_name in ('lens','shift_x','shift_y'):
                camera_data.keyframe_insert(data_path=property_name, frame=frame)
            bpy.context.scene.render.resolution_x = int(width)
            bpy.context.scene.render.resolution_y = int(height)
            bpy.context.scene.render.pixel_aspect_y = fx/fy
    # A data block preserves exact timestamps even between 30fps display frames.
    bpy.data.texts.new('echo4d-camera-timeline.json').write(json.dumps(data))
    bpy.context.scene.render.fps = 30
    bpy.context.scene.frame_end = max(1, math.ceil(1+previous*30))
    bpy.context.scene.frame_set(1)
    return len(frames)


def render_views(output, trajectory):
    from mathutils import Vector
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 8
    # Ubuntu Blender is built without OpenImageDenoiser; CPU rendering must not
    # inherit the factory denoise setting and fail after a successful save.
    scene.cycles.use_denoising = False
    for layer in scene.view_layers:
        layer.cycles.use_denoising = False
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.world = bpy.data.worlds.new('Review environment')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.8,.8,.8,1)
    frames = json.loads(Path(trajectory).read_text())['frames']
    views = output.parent/'views'
    views.mkdir()
    original_camera = scene.camera
    for index in sorted({0,len(frames)//2,len(frames)-1}):
        frame = 1+30*frames[index]['t_s']
        scene.frame_set(int(frame),subframe=frame-int(frame))
        scene.render.filepath = str(views/f'recorded-{index:05d}.png')
        bpy.ops.render.render(write_still=True)
    points=[obj.matrix_world @ Vector(p) for obj in scene.objects if obj.type=='MESH' for p in obj.bound_box]
    center=sum(points,Vector())/len(points)
    radius=max((p-center).length for p in points)
    free=bpy.data.objects.new('Free review camera',bpy.data.cameras.new('Free review camera'))
    scene.collection.objects.link(free)
    scene.camera=free
    free.data.lens=28
    free.data.clip_end=max(1000,radius*20)
    for index,offset in enumerate([(1,-1,.65),(-1,-.5,.8)],1):
        free.location=center+Vector(offset).normalized()*radius*2.2
        free.rotation_euler=(center-free.location).to_track_quat('-Z','Y').to_euler()
        scene.render.filepath=str(views/f'free-{index}.png')
        bpy.ops.render.render(write_still=True)
    scene.camera=original_camera
    scene.frame_set(1)


def run(source, output, provenance, units='metric', trajectory=None, render=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file() or source.suffix.lower() != ".glb":
        raise ValueError("A real GLB file is required")
    if output.suffix.lower() != ".blend" or output.exists():
        raise ValueError("Use a fresh .blend output filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    # Vertex-colored GLBs may legally omit materials.
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        attributes = obj.data.color_attributes
        if len(attributes) and not len(obj.data.materials):
            material = bpy.data.materials.new(name="Recorded vertex colors")
            material.use_nodes = True
            nodes = material.node_tree.nodes
            shader = nodes.get("Principled BSDF")
            color = nodes.new("ShaderNodeVertexColor")
            color.layer_name = attributes[0].name
            material.node_tree.links.new(color.outputs["Color"], shader.inputs["Base Color"])
            obj.data.materials.append(material)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.color_type = "VERTEX"
    before = geometry()
    bpy.context.scene.unit_settings.system = "METRIC" if units == 'metric' else 'NONE'
    bpy.context.scene.unit_settings.scale_length = 1.0
    bpy.context.scene["echo4d_source_glb_sha256"] = sha(source)
    bpy.context.scene["echo4d_input_provenance"] = provenance
    bpy.context.scene['echo4d_scale'] = units
    camera_frames = add_timeline(trajectory) if trajectory else 0
    for picture in bpy.data.images:
        if picture.type == 'IMAGE' and picture.has_data and not picture.packed_file:
            picture.pack()
    # Preserve source texture appearance and open on the actual model bounds.
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.exposure = 0
    bpy.context.scene.view_settings.gamma = 1
    points=[obj.matrix_world @ Vector(p) for obj in bpy.context.scene.objects
            if obj.type == 'MESH' for p in obj.bound_box]
    center=sum(points,Vector())/len(points)
    radius=max((p-center).length for p in points)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.type = 'MATERIAL'
                region=area.spaces.active.region_3d
                region.view_location=center
                region.view_distance=max(radius*2.8,.01)
                if bpy.context.scene.camera:
                    region.view_rotation=bpy.context.scene.camera.matrix_world.to_quaternion()
    bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True)
    bpy.ops.wm.open_mainfile(filepath=str(output), load_ui=False)
    after = geometry()
    if before != after:
        raise RuntimeError("Geometry/material counts changed after .blend reopening")
    if render:
        if not trajectory:
            raise ValueError('Rendering recorded views requires a camera timeline')
        render_views(output, trajectory)
    receipt = {
        "blender_version": bpy.app.version_string,
        "input_provenance": provenance,
        "source_glb_sha256": sha(source),
        "blend_sha256": sha(output),
        "imported": before,
        "reopened": after,
        "native_blend_reopen": "PASS",
        "scene_quality": "NOT EVALUATED by this compatibility check",
        "scale": units,
        "recorded_camera_frames": camera_frames,
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance", default="user-provided GLB")
    parser.add_argument('--units', choices=['metric','unknown'], default='metric')
    parser.add_argument('--trajectory', type=Path)
    parser.add_argument('--render-views', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    run(args.input, args.output, args.provenance, args.units, args.trajectory, args.render_views)


