"""Geometry processing and conservative structural completion package."""

from .provenance import (
    EvidenceSource,
    PlaneModel,
    SceneGeometry,
    StructuralCategory,
)
from .plane_detection import detect_structural_planes, fit_plane_ransac_huber
from .multiview_checks import MultiViewConsistencyChecker, project_points_to_camera
from .structural_completion import (
    ConservativeExtrapolator,
    execute_structural_completion,
)

__all__ = [
    "EvidenceSource",
    "StructuralCategory",
    "PlaneModel",
    "SceneGeometry",
    "detect_structural_planes",
    "fit_plane_ransac_huber",
    "MultiViewConsistencyChecker",
    "project_points_to_camera",
    "ConservativeExtrapolator",
    "execute_structural_completion",
]
