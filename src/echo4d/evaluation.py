"""Metrics against a caller-supplied independent, aligned reference point set."""
import math
import numpy as np
from scipy.spatial import cKDTree
from .model import DataError


def compare_points(reconstruction: np.ndarray, reference: np.ndarray, threshold: float = .05) -> dict:
    if not math.isfinite(threshold) or threshold <= 0:
        raise DataError("Comparison threshold must be positive and finite")
    for points in (reconstruction, reference):
        if (points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "fiu"
                or not len(points) or len(points) > 5_000_000 or not np.isfinite(points).all()):
            raise DataError("Point sets must be nonempty finite numeric Nx3 arrays, at most 5 million points")
    forward = cKDTree(reference).query(reconstruction, workers=1)[0]
    backward = cKDTree(reconstruction).query(reference, workers=1)[0]
    precision, recall = float(np.mean(forward <= threshold)), float(np.mean(backward <= threshold))
    return {"threshold_m": threshold, "reconstruction_points": len(reconstruction), "reference_points": len(reference),
            "point_rmse_m": float(np.sqrt(np.mean(forward ** 2))), "reference_rmse_m": float(np.sqrt(np.mean(backward ** 2))),
            "precision": precision, "coverage": recall,
            "f_score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "note": "Nearest-neighbour metrics depend on density and alignment. No registration performed."}
