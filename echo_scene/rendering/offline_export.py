"""Offline high-quality path tracing export and standalone PBR renderer.

Supports Blender Cycles automated rendering scripts and a completely self-contained
standalone PBR offscreen renderer producing 3 original camera and 2 free exploration renders.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from .scene_manifest import CameraSpecification, SceneManifest, build_scene_manifest


@dataclass
class OfflineRenderConfig:
    """Configuration for offline photorealistic rendering."""
    samples: int = 128
    width: int = 1280
    height: int = 720
    denoise: bool = True
    format: str = "png"
    render_free_views: bool = True

    @classmethod
    def from_preset(cls, preset: str = "standard") -> "OfflineRenderConfig":
        presets = {
            "draft": cls(samples=32, width=960, height=540, denoise=True),
            "standard": cls(samples=128, width=1280, height=720, denoise=True),
            "high": cls(samples=512, width=1920, height=1080, denoise=True),
        }
        return presets.get(preset.lower(), presets["standard"])


def generate_blender_cycles_script(
    manifest: SceneManifest,
    config: OfflineRenderConfig,
    script_path: Path,
) -> Path:
    """Generate self-contained Blender script for high-end Cycles path tracing."""
    script_path = Path(script_path)
    script_path.parent.mkdir(parents=True, exist_ok=True)

    obs_p = manifest.geometry.get("observed_mesh", "").replace("\\", "/")
    struct_p = manifest.geometry.get("structural_mesh", "").replace("\\", "/")
    out_dir = str(script_path.parent / "renders").replace("\\", "/")

    script_content = f"""# Auto-generated Blender Cycles Rendering Script
import bpy
import os

# Clean default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Configure Cycles Engine
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = {config.samples}
scene.cycles.use_denoising = {config.denoise}
scene.render.resolution_x = {config.width}
scene.render.resolution_y = {config.height}
scene.render.image_settings.file_format = '{config.format.upper()}'

# Device setup
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = 'GPU'
except Exception:
    scene.cycles.device = 'CPU'

# Import Meshes
os.makedirs('{out_dir}', exist_ok=True)
if os.path.exists('{obs_p}'):
    bpy.ops.wm.ply_import(filepath='{obs_p}')
if os.path.exists('{struct_p}'):
    bpy.ops.wm.ply_import(filepath='{struct_p}')

# Lighting
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_data.energy = 3.5
light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.rotation_euler = (0.78, 0.45, 0.3)

# Camera
cam_data = bpy.data.cameras.new("RenderCam")
cam_obj = bpy.data.objects.new("RenderCam", cam_data)
bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Camera views loop
cameras_data = {json.dumps([c.to_dict() for c in manifest.cameras])}
for cam_info in cameras_data:
    c_pos = cam_info['center']
    cam_obj.location = (c_pos[0], c_pos[1], c_pos[2])
    # Look toward scene center
    cam_obj.rotation_euler = (1.2, 0.0, 0.8)
    scene.render.filepath = os.path.join('{out_dir}', cam_info['name'] + '.{config.format}')
    bpy.ops.render.render(write_still=True)
print("Blender Cycles Rendering Complete.")
"""
    script_path.write_text(script_content, encoding="utf-8")
    return script_path


def render_offline_standalone(
    manifest: SceneManifest,
    config: OfflineRenderConfig,
    out_dir: Path,
) -> List[Path]:
    """Execute offline rendering using Open3D headless PBR offscreen renderer."""
    import open3d as o3d
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered_files: List[Path] = []
    w, h = config.width, config.height

    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    scene = renderer.scene
    scene.set_background([0.12, 0.12, 0.14, 1.0])

    # PBR Material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.85, 0.85, 0.88, 1.0]
    mat.base_roughness = 0.4
    mat.base_reflectance = 0.5

    # Load Observed Mesh
    scene_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    obs_p = manifest.geometry.get("observed_mesh")
    if obs_p and Path(obs_p).exists():
        mesh_obs = o3d.io.read_triangle_mesh(obs_p)
        mesh_obs.compute_vertex_normals()
        scene_center = np.asarray(mesh_obs.get_center())
        scene.add_geometry("observed_mesh", mesh_obs, mat)

    # Load Structural Mesh if exists
    struct_p = manifest.geometry.get("structural_mesh")
    if struct_p and Path(struct_p).exists():
        struct_mat = o3d.visualization.rendering.MaterialRecord()
        struct_mat.shader = "defaultLit"
        struct_mat.base_color = [0.70, 0.78, 0.86, 1.0]
        struct_mat.base_roughness = 0.6
        mesh_struct = o3d.io.read_triangle_mesh(struct_p)
        mesh_struct.compute_vertex_normals()
        scene.add_geometry("structural_mesh", mesh_struct, struct_mat)

    # Setup Directional Sun Light
    scene.scene.set_sun_light(
        [-0.5, -1.0, -0.6], [1.0, 0.98, 0.95], 100000.0
    )
    scene.scene.enable_sun_light(True)

    # Render each camera view in the manifest (up to 3 source + 2 free views)
    for cam in manifest.cameras:
        if cam.is_free_view:
            c_pos = scene_center + np.array(cam.center, dtype=np.float64)
        else:
            c_pos = np.array(cam.center, dtype=np.float64)
        target = scene_center
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        renderer.setup_camera(float(cam.fov_degrees), target, c_pos, up)
        img_arr = np.asarray(renderer.render_to_image())
        if img_arr.size > 0:
            out_file = out_dir / f"{cam.name}.{config.format}"
            Image.fromarray(img_arr).save(out_file)
            rendered_files.append(out_file)

    del renderer
    return rendered_files


def execute_offline_render(
    job_dir: Path,
    preset: str = "standard",
) -> Dict[str, Any]:
    """Execute complete offline render pipeline and record execution receipt."""
    job_dir = Path(job_dir)
    manifest_p = job_dir / "scene_manifest.json"
    if not manifest_p.exists():
        manifest = build_scene_manifest(job_dir)
    else:
        manifest = SceneManifest.load(manifest_p)

    config = OfflineRenderConfig.from_preset(preset)
    renders_dir = job_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # 1. Always generate the Blender Cycles script for studio handoff
    script_p = renders_dir / "render_cycles.py"
    generate_blender_cycles_script(manifest, config, script_p)

    # 2. Check if Blender CLI exists
    blender_bin = shutil.which("blender")
    used_engine = "standalone_pbr"
    rendered_images: List[str] = []

    if blender_bin:
        try:
            cmd = [blender_bin, "-b", "-P", str(script_p)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                used_engine = "blender_cycles"
                rendered_images = [str(p) for p in renders_dir.glob(f"*.{config.format}")]
        except Exception:
            pass

    # 3. If Blender was not run or produced no outputs, run standalone PBR renderer
    if not rendered_images:
        out_paths = render_offline_standalone(manifest, config, renders_dir)
        rendered_images = [str(p) for p in out_paths]
        used_engine = "standalone_pbr"

    receipt = {
        "engine": used_engine,
        "preset": preset,
        "samples": config.samples,
        "resolution": [config.width, config.height],
        "rendered_views_count": len(rendered_images),
        "rendered_images": rendered_images,
        "cycles_script": str(script_p),
    }
    (renders_dir / "render_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt
