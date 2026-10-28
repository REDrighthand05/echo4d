"""Conservative frame-to-model ICP for the real-room high-information experiment.

The reference TSDF is fused ONLY from training measured depth and known poses.
Refinement is bounded; it is not a ground-truth pose replacement or a way to
invent unobserved surfaces. Holdout frames are never loaded by this module.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from echo4d.holdout import read_echo_ply
from echo4d.model import DataError, load_recording


def correction_limits(initial: np.ndarray, candidate: np.ndarray, *,
                      max_translation_m: float = 0.025,
                      max_rotation_deg: float = 1.0) -> tuple[bool, float, float]:
    """Bound the WORLD-space pose correction, not the camera's absolute pose."""
    if (initial.shape != (4, 4) or candidate.shape != (4, 4)
            or not np.isfinite(initial).all() or not np.isfinite(candidate).all()):
        return False, float("inf"), float("inf")
    delta = candidate @ np.linalg.inv(initial)
    translation = float(np.linalg.norm(delta[:3, 3]))
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    valid = (np.allclose(candidate[3], [0, 0, 0, 1], atol=1e-6)
             and translation <= max_translation_m
             and rotation_deg <= max_rotation_deg)
    return bool(valid), translation, rotation_deg


def refine_training_poses(manifest_path: Path, reference_mesh: Path,
                          output_manifest: Path, *, stride: int = 8,
                          max_correspondence_m: float = 0.06) -> dict:
    """Align measured training depth to a training-only TSDF with strict bounds.

    ICP may overfit its own reference Mesh; acceptance here is exploratory.
    No held-out RGB, depth, or poses are read. The original manifest is retained.
    """
    if stride not in (4, 8, 16) or not 0.02 <= max_correspondence_m <= 0.10:
        raise DataError("Invalid bounded ICP settings")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise DataError("Open3D required for bounded room ICP") from exc

    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    recording = load_recording(manifest_path)
    vertices, faces = read_echo_ply(reference_mesh)
    if len(vertices) < 500 or len(faces) < 500:
        raise DataError("ICP requires a nonempty measured-depth reference Mesh")
    reference = o3d.geometry.TriangleMesh()
    reference.vertices = o3d.utility.Vector3dVector(
        np.asarray(vertices, dtype=np.float64))
    reference.triangles = o3d.utility.Vector3iVector(
        np.asarray(faces, dtype=np.int32))
    reference.compute_vertex_normals()
    target = o3d.geometry.PointCloud()
    target.points = reference.vertices
    target.normals = reference.vertex_normals
    target = target.voxel_down_sample(voxel_size=0.018)
    if len(target.points) < 500:
        raise DataError("Insufficient ICP reference points")
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=24))

    cam = recording.camera
    accepted, diagnostics = 0, []
    if len(manifest["frames"]) != len(recording.entries):
        raise DataError("Training manifest frame mismatch")
    for i, (_, frame, _) in enumerate(recording.iter_frames()):
        depth = np.asarray(frame.depth, dtype=np.float32)[::stride, ::stride]
        good = np.isfinite(depth) & (depth >= 0.20) & (depth <= 4.0)
        if frame.excluded is not None:
            good &= ~np.asarray(frame.excluded)[::stride, ::stride]
        rows, cols = np.nonzero(good)
        original = np.asarray(frame.camera_to_world, dtype=np.float64)
        record = {"frame": i, "sampled_depth_points": int(len(rows)),
                  "accepted": False}
        if len(rows) < 200:
            record["reason"] = "too_few_measured_points"
            diagnostics.append(record)
            continue
        z = depth[rows, cols].astype(np.float64)
        u = cols * stride
        v = rows * stride
        xyz = np.column_stack(((u - cam.cx) * z / cam.fx,
                               (v - cam.cy) * z / cam.fy, z))
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(xyz)
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_m, original,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=12))
        allowed, translation, angle = correction_limits(original, result.transformation)
        record.update({"fitness": float(result.fitness),
                       "inlier_rmse_m": float(result.inlier_rmse),
                       "correction_translation_m": translation,
                       "correction_rotation_deg": angle})
        if allowed and result.fitness >= 0.45 and result.inlier_rmse <= 0.035:
            manifest["frames"][i]["cameraToWorld"] = (
                result.transformation.flatten(order="F").tolist())
            record["accepted"] = True
            accepted += 1
        else:
            record["reason"] = "low_overlap_or_correction_exceeds_bounds"
        diagnostics.append(record)

    manifest["notes"] = (manifest.get("notes", "") +
        "; exploratory bounded training-only frame-to-TSDF ICP; original poses retained if rejected")
    output_manifest.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")
    return {"training_frames": len(diagnostics), "accepted_pose_corrections": accepted,
            "rejected_pose_corrections": len(diagnostics) - accepted,
            "max_translation_m": 0.025, "max_rotation_deg": 1.0,
            "reference": "training-only baseline TSDF, not independent ground truth",
            "sample_stride": stride, "per_frame": diagnostics}
