"""Import a real GLB into Blender, save native .blend, and verify reopening."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy

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

def run(source, output, provenance):
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
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    bpy.context.scene["echo4d_source_glb_sha256"] = sha(source)
    bpy.context.scene["echo4d_input_provenance"] = provenance
    bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True)
    bpy.ops.wm.open_mainfile(filepath=str(output), load_ui=False)
    after = geometry()
    if before != after:
        raise RuntimeError("Geometry/material counts changed after .blend reopening")
    receipt = {
        "blender_version": bpy.app.version_string,
        "input_provenance": provenance,
        "source_glb_sha256": sha(source),
        "blend_sha256": sha(output),
        "imported": before,
        "reopened": after,
        "native_blend_reopen": "PASS",
        "scene_quality": "NOT EVALUATED by this compatibility check",
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance", default="user-provided GLB")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    run(args.input, args.output, args.provenance)
