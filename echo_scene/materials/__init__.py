"""Procedural PBR materials package."""

from .procedural import (
    MappingSpec,
    MaterialDescriptor,
    MaterialFamily,
    MaterialProvenance,
    PhysicalProperties,
    TextureMaps,
)
from .patterns import (
    cellular_worley_noise,
    crack_network_pattern,
    directional_noise,
    fbm_noise,
    hexagonal_honeycomb_pattern,
    twill_weave_pattern,
)
from .pbr_maps import bake_pbr_textures
from .presets import (
    ADVANCED_PRESETS,
    INDOOR_PRESETS,
    create_test_sphere_glb,
    get_material_preset,
    list_all_presets,
)

__all__ = [
    "MappingSpec",
    "MaterialDescriptor",
    "MaterialFamily",
    "MaterialProvenance",
    "PhysicalProperties",
    "TextureMaps",
    "fbm_noise",
    "directional_noise",
    "cellular_worley_noise",
    "twill_weave_pattern",
    "hexagonal_honeycomb_pattern",
    "crack_network_pattern",
    "bake_pbr_textures",
    "INDOOR_PRESETS",
    "ADVANCED_PRESETS",
    "get_material_preset",
    "list_all_presets",
    "create_test_sphere_glb",
]
