"""Texture map baking and channel packing for glTF PBR compliance.

Generates tangent-space normal maps, glTF metallic-roughness-occlusion packed maps,
and sRGB base color maps from procedural height and property fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

from .procedural import MaterialDescriptor, linear_to_srgb


def height_to_normal_map(
    height: np.ndarray,
    normal_strength: float = 1.0,
) -> np.ndarray:
    """Compute tangent-space OpenGL normal map from height map using Sobel filters.

    Returns:
        RGB uint8 array of shape (H, W, 3) where [128, 128, 255] is flat.
    """
    h, w = height.shape
    # Sobel kernels
    # dx: difference along horizontal axis
    dx = (
        np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
    ) * 0.5
    # dy: difference along vertical axis (OpenGL convention: +Y is up)
    dy = (
        np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    ) * 0.5

    nx = -dx * normal_strength * 2.0
    ny = -dy * normal_strength * 2.0
    nz = np.ones_like(nx)

    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= norm
    ny /= norm
    nz /= norm

    # Map [-1, 1] to [0, 255]
    r = np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    g = np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    b = np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    return np.dstack([r, g, b])


def pack_metallic_roughness_map(
    roughness: np.ndarray,
    metallic: np.ndarray,
    occlusion: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pack properties into glTF standard ORM format (R: Occlusion, G: Roughness, B: Metallic)."""
    h, w = roughness.shape
    r_chan = (
        np.clip(occlusion * 255.0, 0, 255).astype(np.uint8)
        if occlusion is not None
        else np.full((h, w), 255, dtype=np.uint8)
    )
    g_chan = np.clip(roughness * 255.0, 0, 255).astype(np.uint8)
    b_chan = np.clip(metallic * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([r_chan, g_chan, b_chan])


def bake_pbr_textures(
    mat: MaterialDescriptor,
    out_dir: Path,
    height_map: np.ndarray,
    roughness_map: Optional[np.ndarray] = None,
    color_variation: Optional[np.ndarray] = None,
    resolution: int = 512,
) -> MaterialDescriptor:
    """Bake and save complete set of PBR texture maps for a material."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = height_map.shape

    # 1. Normal Map
    normal_img = height_to_normal_map(
        height_map, normal_strength=mat.physical.normal_strength
    )
    norm_path = out_dir / f"{mat.id}_normal.png"
    Image.fromarray(normal_img).save(norm_path)
    mat.textures.normal_map = str(norm_path)

    # 2. Base Color Map (sRGB)
    base_rgb = np.array(mat.physical.base_color, dtype=np.float32)
    if color_variation is not None:
        var = np.clip(color_variation[:, :, None], 0.0, 1.0)
        color_arr = base_rgb[None, None, :] * (0.8 + 0.4 * var)
    else:
        color_arr = np.tile(base_rgb[None, None, :], (h, w, 1))

    color_srgb = np.clip(color_arr * 255.0, 0, 255).astype(np.uint8)
    color_path = out_dir / f"{mat.id}_base_color.png"
    Image.fromarray(color_srgb).save(color_path)
    mat.textures.base_color_map = str(color_path)

    # 3. Metallic-Roughness Map
    if roughness_map is None:
        roughness_map = np.full((h, w), mat.physical.roughness, dtype=np.float32)
    metallic_map = np.full((h, w), mat.physical.metallic, dtype=np.float32)

    orm_img = pack_metallic_roughness_map(roughness_map, metallic_map)
    orm_path = out_dir / f"{mat.id}_metallic_roughness.png"
    Image.fromarray(orm_img).save(orm_path)
    mat.textures.metallic_roughness_map = str(orm_path)

    # 4. Height Map (Linear)
    h_norm = np.clip(height_map * 255.0, 0, 255).astype(np.uint8)
    h_path = out_dir / f"{mat.id}_height.png"
    Image.fromarray(h_norm).save(h_path)
    mat.textures.height_map = str(h_path)

    return mat
