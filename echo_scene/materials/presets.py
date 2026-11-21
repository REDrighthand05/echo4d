"""Library of 8 basic indoor and 8 advanced specialized procedural PBR materials.

Implements parameter configurations, texture generation synthesis, and GLB test asset creation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .patterns import (
    cellular_worley_noise,
    crack_network_pattern,
    directional_noise,
    fbm_noise,
    hexagonal_honeycomb_pattern,
    twill_weave_pattern,
)
from .pbr_maps import bake_pbr_textures
from .procedural import (
    MaterialDescriptor,
    MaterialFamily,
    PhysicalProperties,
)

INDOOR_PRESETS = [
    "painted_wall",
    "plaster_concrete",
    "wood_furniture",
    "ceramic_tile",
    "fabric",
    "glass",
    "plastic",
    "common_metal",
]

ADVANCED_PRESETS = [
    "brushed_metal",
    "carbon_fiber_twill",
    "acoustic_honeycomb",
    "cast_aluminum",
    "thermal_barrier_coating",
    "thermal_ablation_tint",
    "soot_oil_grime",
    "clean_stainless_steel",
]


def create_material(
    mat_id: str,
    family: MaterialFamily,
    base_color: List[float],
    metallic: float,
    roughness: float,
    normal_strength: float = 1.0,
    seed: int = 42,
) -> MaterialDescriptor:
    """Helper to instantiate MaterialDescriptor."""
    return MaterialDescriptor(
        id=mat_id,
        family=family,
        procedural_seed=seed,
        physical=PhysicalProperties(
            base_color=base_color,
            metallic=metallic,
            roughness=roughness,
            normal_strength=normal_strength,
        ),
    )


def synthesize_material_textures(
    mat: MaterialDescriptor,
    out_dir: Path,
    res: int = 256,
) -> MaterialDescriptor:
    """Generate height, roughness, and color variation fields and bake PBR textures."""
    shape = (res, res)
    seed = mat.procedural_seed
    mat_id = mat.id

    if mat_id == "painted_wall":
        height = fbm_noise(shape, octaves=3, seed=seed) * 0.15
        rough = np.full(shape, 0.85, dtype=np.float32)
        var = None
    elif mat_id == "plaster_concrete":
        height = fbm_noise(shape, octaves=5, seed=seed) * 0.4 + cellular_worley_noise(shape, seed=seed) * 0.2
        rough = np.clip(0.8 + height * 0.15, 0.0, 1.0)
        var = height
    elif mat_id == "wood_furniture":
        height = directional_noise(shape, angle_rad=0.05, freq_along=0.01, freq_across=0.1, seed=seed) * 0.3
        rough = np.full(shape, 0.40, dtype=np.float32)
        var = height
    elif mat_id == "ceramic_tile":
        height = fbm_noise(shape, octaves=2, seed=seed) * 0.05
        rough = np.full(shape, 0.15, dtype=np.float32)
        var = None
    elif mat_id == "fabric":
        h1 = directional_noise(shape, 0.0, freq_across=0.4, seed=seed)
        h2 = directional_noise(shape, math.pi / 2.0, freq_across=0.4, seed=seed + 1)
        height = (h1 + h2) * 0.25
        rough = np.full(shape, 0.80, dtype=np.float32)
        var = height
    elif mat_id == "glass":
        height = np.zeros(shape, dtype=np.float32)
        rough = np.full(shape, 0.05, dtype=np.float32)
        var = None
    elif mat_id == "plastic":
        height = fbm_noise(shape, octaves=2, seed=seed) * 0.08
        rough = np.full(shape, 0.35, dtype=np.float32)
        var = None
    elif mat_id == "common_metal":
        height = fbm_noise(shape, octaves=4, seed=seed) * 0.2
        rough = np.full(shape, 0.30, dtype=np.float32)
        var = None
    elif mat_id == "brushed_metal":
        height = directional_noise(shape, angle_rad=0.0, freq_along=0.002, freq_across=0.5, seed=seed) * 0.5
        rough = np.clip(0.28 + height * 0.1, 0.0, 1.0)
        var = None
    elif mat_id == "carbon_fiber_twill":
        height, _ = twill_weave_pattern(shape, yarn_count=12)
        rough = np.where(height > 0.5, 0.20, 0.28).astype(np.float32)
        var = height
    elif mat_id == "acoustic_honeycomb":
        height = hexagonal_honeycomb_pattern(shape, cell_freq=8.0) * 0.7
        rough = np.where(height > 0.3, 0.35, 0.75).astype(np.float32)
        var = height
    elif mat_id == "cast_aluminum":
        height = cellular_worley_noise(shape, num_cells=120, seed=seed) * 0.4 + fbm_noise(shape, 4, seed=seed) * 0.2
        rough = np.full(shape, 0.55, dtype=np.float32)
        var = height
    elif mat_id == "thermal_barrier_coating":
        height = crack_network_pattern(shape, num_seeds=30, seed=seed) * 0.6
        rough = np.clip(0.82 + height * 0.1, 0.0, 1.0)
        var = 1.0 - height * 0.5
    elif mat_id == "thermal_ablation_tint":
        y = np.linspace(0, 1, res, dtype=np.float32)[:, None]
        height = fbm_noise(shape, octaves=3, seed=seed) * 0.1
        rough = np.clip(0.40 + y * 0.2, 0.0, 1.0)
        var = y
    elif mat_id == "soot_oil_grime":
        height = fbm_noise(shape, octaves=4, seed=seed) * 0.4
        rough = np.where(height > 0.5, 0.30, 0.75).astype(np.float32)
        var = height
    elif mat_id == "clean_stainless_steel":
        height = directional_noise(shape, angle_rad=0.0, freq_along=0.001, freq_across=0.3, seed=seed) * 0.15
        rough = np.full(shape, 0.12, dtype=np.float32)
        var = None
    else:
        height = np.zeros(shape, dtype=np.float32)
        rough = np.full(shape, mat.physical.roughness, dtype=np.float32)
        var = None

    return bake_pbr_textures(mat, out_dir, height, roughness_map=rough, color_variation=var)


def get_material_preset(name: str, seed: int = 42) -> MaterialDescriptor:
    """Retrieve configured MaterialDescriptor by name."""
    catalog: Dict[str, MaterialDescriptor] = {
        "painted_wall": create_material("painted_wall", MaterialFamily.PAINT, [0.92, 0.91, 0.88], 0.0, 0.85, 0.5, seed),
        "plaster_concrete": create_material("plaster_concrete", MaterialFamily.CONCRETE, [0.65, 0.65, 0.64], 0.02, 0.90, 1.2, seed),
        "wood_furniture": create_material("wood_furniture", MaterialFamily.WOOD, [0.55, 0.35, 0.20], 0.0, 0.40, 0.8, seed),
        "ceramic_tile": create_material("ceramic_tile", MaterialFamily.CERAMIC, [0.95, 0.95, 0.94], 0.05, 0.15, 0.4, seed),
        "fabric": create_material("fabric", MaterialFamily.FABRIC, [0.45, 0.48, 0.52], 0.0, 0.80, 1.0, seed),
        "glass": create_material("glass", MaterialFamily.GLASS, [0.92, 0.95, 0.95], 0.1, 0.05, 0.1, seed),
        "plastic": create_material("plastic", MaterialFamily.PLASTIC, [0.25, 0.25, 0.26], 0.0, 0.35, 0.6, seed),
        "common_metal": create_material("common_metal", MaterialFamily.METAL, [0.82, 0.82, 0.84], 0.95, 0.30, 0.7, seed),
        "brushed_metal": create_material("brushed_metal", MaterialFamily.METAL, [0.85, 0.85, 0.87], 0.95, 0.28, 1.2, seed),
        "carbon_fiber_twill": create_material("carbon_fiber_twill", MaterialFamily.COMPOSITE, [0.08, 0.08, 0.09], 0.10, 0.22, 1.4, seed),
        "acoustic_honeycomb": create_material("acoustic_honeycomb", MaterialFamily.ADVANCED, [0.72, 0.74, 0.75], 0.80, 0.45, 1.5, seed),
        "cast_aluminum": create_material("cast_aluminum", MaterialFamily.METAL, [0.78, 0.80, 0.82], 0.92, 0.55, 1.3, seed),
        "thermal_barrier_coating": create_material("thermal_barrier_coating", MaterialFamily.CERAMIC, [0.75, 0.72, 0.62], 0.0, 0.82, 1.6, seed),
        "thermal_ablation_tint": create_material("thermal_ablation_tint", MaterialFamily.METAL, [0.45, 0.38, 0.62], 0.88, 0.45, 0.8, seed),
        "soot_oil_grime": create_material("soot_oil_grime", MaterialFamily.ADVANCED, [0.12, 0.11, 0.10], 0.05, 0.75, 1.0, seed),
        "clean_stainless_steel": create_material("clean_stainless_steel", MaterialFamily.METAL, [0.90, 0.91, 0.92], 0.98, 0.12, 0.5, seed),
    }
    if name not in catalog:
        raise ValueError(f"Unknown preset: {name}. Available: {list(catalog.keys())}")
    return catalog[name]


def list_all_presets() -> List[str]:
    return INDOOR_PRESETS + ADVANCED_PRESETS


def create_test_sphere_glb(mat: MaterialDescriptor, out_path: Path) -> Path:
    """Generate a standard UV sphere mesh with the material baked and export to GLB."""
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=32)
    mesh.compute_vertex_normals()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    return out_path
