"""Material data structures, color spaces, and parameter contracts for PBR."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class MaterialFamily(str, Enum):
    """Broad classification of material behavior."""
    PAINT = "paint"
    CONCRETE = "concrete"
    WOOD = "wood"
    CERAMIC = "ceramic"
    FABRIC = "fabric"
    GLASS = "glass"
    PLASTIC = "plastic"
    METAL = "metal"
    COMPOSITE = "composite"
    ADVANCED = "advanced"


@dataclass
class PhysicalProperties:
    """Core PBR physical shading parameters."""
    base_color: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])  # sRGB 0.0-1.0
    metallic: float = 0.0                                                    # 0.0 dielectric to 1.0 metal
    roughness: float = 0.5                                                   # 0.0 smooth to 1.0 rough
    normal_strength: float = 1.0                                             # multiplier
    occlusion_strength: float = 1.0                                          # ambient occlusion weight
    ior: Optional[float] = 1.5                                               # Index of Refraction


@dataclass
class TextureMaps:
    """Relative or absolute disk paths to baked PBR textures."""
    base_color_map: Optional[str] = None           # sRGB image
    normal_map: Optional[str] = None               # OpenGL tangent space linear
    metallic_roughness_map: Optional[str] = None   # glTF packed (G: roughness, B: metallic)
    occlusion_map: Optional[str] = None            # linear single channel or packed R
    height_map: Optional[str] = None               # linear height


@dataclass
class MappingSpec:
    """Texture projection and scale specification."""
    method: str = "triplanar"     # 'uv' or 'triplanar'
    world_scale: float = 1.0      # texture repeat scale in meters
    rotation_deg: float = 0.0


@dataclass
class MaterialProvenance:
    """Audit origin of material assignment."""
    assignment_source: str = "procedural_synthesizer"
    confidence_heuristic: float = 0.85
    user_override: bool = False


@dataclass
class MaterialDescriptor:
    """Complete specification of a procedural PBR material layer."""
    id: str
    family: MaterialFamily
    procedural_seed: int = 42
    physical: PhysicalProperties = field(default_factory=PhysicalProperties)
    textures: TextureMaps = field(default_factory=TextureMaps)
    mapping: MappingSpec = field(default_factory=MappingSpec)
    provenance: MaterialProvenance = field(default_factory=MaterialProvenance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family.value,
            "procedural_seed": self.procedural_seed,
            "physical": asdict(self.physical),
            "textures": asdict(self.textures),
            "mapping": asdict(self.mapping),
            "provenance": asdict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaterialDescriptor":
        return cls(
            id=data.get("id", "default_material"),
            family=MaterialFamily(data.get("family", "paint")),
            procedural_seed=data.get("procedural_seed", 42),
            physical=PhysicalProperties(**data.get("physical", {})),
            textures=TextureMaps(**data.get("textures", {})),
            mapping=MappingSpec(**data.get("mapping", {})),
            provenance=MaterialProvenance(**data.get("provenance", {})),
        )


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB array [0.0, 1.0] to sRGB."""
    linear = np.clip(linear, 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return srgb


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB array [0.0, 1.0] to linear RGB."""
    srgb = np.clip(srgb, 0.0, 1.0)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    )
    return linear
