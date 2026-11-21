"""Shared rendering assets, scene manifest, and offline renderer bridge."""

from .scene_manifest import (
    CameraSpecification,
    LightSpecification,
    SceneManifest,
    build_scene_manifest,
)
from .offline_export import (
    OfflineRenderConfig,
    execute_offline_render,
    generate_blender_cycles_script,
    render_offline_standalone,
)

__all__ = [
    "CameraSpecification",
    "LightSpecification",
    "SceneManifest",
    "build_scene_manifest",
    "OfflineRenderConfig",
    "generate_blender_cycles_script",
    "render_offline_standalone",
    "execute_offline_render",
]
