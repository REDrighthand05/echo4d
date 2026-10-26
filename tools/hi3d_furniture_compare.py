"""Fair same-object coarse vs proposed Hi3D mesh on FROZEN withheld RGB-D views.

Only a reviewed, independently calibrated single furniture instance is eligible.
This module never fits a transform to withheld points or changes either mesh.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from echo4d.holdout import read_echo_ply, rasterize_mesh_depth
from echo4d.model import Camera, DataError, file_sha256, rigid_matrix, safe_path
from hi3d_furniture_evidence import validate_ledger

EXPECTED_COUNT = 12


def dimensions(vertices: np.ndarray, axes: np.ndarray) -> np.ndarray:
    if axes.shape != (3, 3) or not np.isfinite(axes).all() or not np.allclose(
        axes.T @ axes, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(axes), 1.0, atol=1e-5
    ):
        raise DataError("Object measurement axes must be an audited right-handed rotation")
    local = vertices @ axes
    return np.ptp(local, axis=0)


def numeric_score(pred: np.ndarray, measured_m: np.ndarray, object_mask: np.ndarray,
                  min_z: float = .15, max_z: float = 4.) -> tuple[dict, np.ndarray]:
    if (pred.shape != measured_m.shape or pred.shape != object_mask.shape or pred.ndim != 2
            or object_mask.dtype != bool or not np.isfinite(pred).all()
            or not np.isfinite(measured_m).all() or np.any(measured_m < 0)
            or np.any(pred < 0)):
        raise DataError("Invalid aligned measured depth, mask or predicted depth")
    observed = object_mask & (measured_m >= min_z) & (measured_m <= max_z)
    n = int(np.count_nonzero(observed))
    if n == 0:
        raise DataError("Object mask contains no valid independently measured Z")
    predicted = (pred >= min_z) & (pred <= max_z)
    overlap = observed & predicted
    errors = (pred[overlap] - measured_m[overlap]).astype(np.float64)
    # Pixels with predicted surfaces outside the object mask are FLAGGED,
    # not blindly called hallucinations: this may be occlusion or label error.
    return ({
        "measured_object_pixels": n,
        "covered_pixels": int(errors.size),
        "missing_pixels": n - int(errors.size),
        "coverage": float(errors.size / n),
        "predicted_pixels_outside_object_mask_warning": int(np.count_nonzero(predicted & ~object_mask)),
        "rmse_m": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
        "mae_m": float(np.mean(np.abs(errors))) if errors.size else None,
        "p95_abs_m": float(np.percentile(np.abs(errors), 95)) if errors.size else None,
    }, errors)


def surface_distance(shape, points: np.ndarray, *, sample_count: int = 3000) -> dict:
    # Reuse the tested deterministic surface sampler from #43, no second sampling algorithm.
    from hi3d_geometry_gate import surface_sample
    vertices, faces = shape
    surface = surface_sample(vertices, faces, sample_count, seed=42)
    d_surface = cKDTree(points).query(surface, workers=1)[0]
    d_observed = cKDTree(surface).query(points, workers=1)[0]
    return {
        "surface_to_measured_rmse_m": float(np.sqrt(np.mean(d_surface**2))),
        "measured_to_surface_rmse_m": float(np.sqrt(np.mean(d_observed**2))),
        "measured_surface_coverage_3cm": float(np.mean(d_observed <= .03)),
        "surface_supported_fraction_3cm": float(np.mean(d_surface <= .03)),
        "measurement_to_surface_p95_m": float(np.percentile(d_observed, 95)),
        "sample_count": sample_count,
    }


def checked_file(root: Path, spec: dict, label: str) -> Path:
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise DataError(f"Missing file path/hash for {label}")
    file = safe_path(root, spec["path"])
    if file.stat().st_size > 100 * 1024**2 or file_sha256(file) != spec["sha256"]:
        raise DataError(f"File hash mismatch or oversized: {label}")
    return file


def _read_depth(path: Path, camera: Camera, divisor: float) -> np.ndarray:
    if divisor != 5000:
        raise DataError("TUM raw sensor PNG must be converted by exactly 5000 units per metre")
    with Image.open(path) as image:
        raw = np.asarray(image)
    if raw.shape != (camera.height, camera.width) or raw.dtype.kind not in "iu":
        raise DataError("Raw TUM depth must be integer PNG matching camera")
    value = raw.astype(np.float32) / 5000.0
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 100):
        raise DataError("Invalid sensor depth")
    return value


def evaluate(ledger_path: Path, proof_path: Path, root: Path) -> dict:
    root = root.resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    records = validate_ledger(ledger)
    source = {f["name"]: f for f in records}
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if (proof.get("version") != "echo-hi3d-furniture-object-proof-1" or
            proof.get("frozen_ledger_sha256") != file_sha256(ledger_path) or
            proof.get("reviewed_same_single_furniture") is not True or
            proof.get("alignment_origin") != "coarse_only_before_holdout" or
            proof.get("holdout_never_used_for_refinement_or_scale") is not True):
        raise DataError("Unreviewed object, mutated holdout, or post-hoc alignment")
    if (not isinstance(proof.get("object_identity_evidence"), str) or
            len(proof["object_identity_evidence"].strip()) < 30 or
            not isinstance(proof.get("independent_dimension_source"), str) or
            len(proof["independent_dimension_source"].strip()) < 30):
        raise DataError("Missing externally auditable furniture identity and physical measurement")
    measure = np.asarray(proof.get("independent_dimensions_m"), dtype=np.float64)
    if measure.shape != (3,) or not np.isfinite(measure).all() or np.any(measure <= 0):
        raise DataError("Unknown physical furniture dimensions")
    axes = np.asarray(proof.get("object_axes_world"), dtype=np.float64)
    # Not every plausible claim is genuine evidence; human check is separate.
    coarse_path = checked_file(root, proof["coarse_world_mesh"], "coarse")
    proposed_path = checked_file(root, proof["refined_world_mesh"], "refined")
    if coarse_path == proposed_path or coarse_path.suffix != ".ply" or proposed_path.suffix != ".ply":
        raise DataError("Two separate metric ECHO PLY assets required")
    coarse = read_echo_ply(coarse_path)
    refined = read_echo_ply(proposed_path)
    original_dimensions = dimensions(coarse[0], axes)
    proposed_dimensions = dimensions(refined[0], axes)
    views = proof.get("views")
    names = ledger["preregistered_protocol"]["holdout_names"]
    if not isinstance(views, list) or [v.get("name") for v in views] != names:
        raise DataError("All 12 original frozen holdout views required in order, no substitutions")
    camera = Camera.from_dict(proof.get("intrinsics"))
    per_view = []
    point_batches = []
    errors = {"coarse": [], "candidate": []}
    totals = {"coarse": {"valid": 0, "overlap": 0},
              "candidate": {"valid": 0, "overlap": 0}}
    for view in views:
        name = view["name"]
        record = source[name]
        if (view.get("original_rgb_t_s") != record["original_rgb_t_s"] or
                view.get("raw_depth_t_s") != record["raw_depth_t_s"] or
                view.get("pose_origin") != "TUM_mocap_calibrated_to_rgb_camera"):
            raise DataError("Holdout source timestamp/pose provenance changed")
        image = checked_file(root, view["original_photo"], name + " original photo")
        depth = checked_file(root, view["original_depth"], name + " original depth")
        if (file_sha256(image) != record["original_rgb_sha256"]
                or file_sha256(depth) != record["raw_depth_sha256"]):
            raise DataError("Holdout RGB/depth differs from frozen raw TUM bytes")
        with Image.open(image) as im:
            if im.size != (camera.width, camera.height):
                raise DataError("Photo resolution incompatible with calibration")
        mask_path = checked_file(root, view["object_mask"], name + " reviewed object mask")
        with Image.open(mask_path) as mask_image:
            mask_array = np.asarray(mask_image)
        if (mask_array.shape != (camera.height, camera.width) or
                mask_array.dtype != np.uint8 or
                not np.isin(mask_array, [0, 255]).all()):
            raise DataError("Only reviewed binary uint8 object masks are allowed")
        mask = mask_array == 255
        if np.count_nonzero(mask) < 32:
            raise DataError("Insufficient verified same-furniture mask pixels")
        pose = rigid_matrix(view.get("camera_to_world_column_major"))
        measured = _read_depth(depth, camera, 5000)
        yy, xx = np.where(mask & (measured >= .15) & (measured <= 4))
        zz = measured[yy, xx].astype(np.float64)
        point_camera = np.column_stack(((xx-camera.cx)*zz/camera.fx,
                                        (yy-camera.cy)*zz/camera.fy, zz))
        point_world = point_camera @ pose[:3, :3].T + pose[:3, 3]
        # deterministic spatial thinning, not per-model cherry picking.
        point_batches.append(point_world[::max(1, len(point_world)//12000)])
        result = {"name": name, "measured_object_pixels": len(zz)}
        for label, geometry in (("coarse", coarse), ("candidate", refined)):
            projected, _ = rasterize_mesh_depth(*geometry, camera, pose, far=4.)
            score, err = numeric_score(projected, measured, mask)
            result[label] = score
            errors[label].append(err)
            totals[label]["valid"] += score["measured_object_pixels"]
            totals[label]["overlap"] += score["covered_pixels"]
        per_view.append(result)
    truth = np.concatenate(point_batches, axis=0)
    if len(truth) < 32:
        raise DataError("Insufficient object-only reference measurements")
    output = {"version": "echo-hi3d-furniture-object-comparison-1",
              "source_ledger_sha256": file_sha256(ledger_path),
              "object_proof_sha256": file_sha256(proof_path),
              "coarse_sha256": file_sha256(coarse_path),
              "candidate_sha256": file_sha256(proposed_path),
              "review_claim_not_cryptographically_authenticated": True,
              "hi3d_inference_claim_not_verified_by_this_cpu_tool": True,
              "raw_depth_units_per_metre": 5000,
              "real_furniture_dimensions_source_claim": proof["independent_dimension_source"],
              "original_measured_dimensions_m": measure.tolist(),
              "per_frame": per_view,
              "unknown_surfaces_truth": "Not certified; outside-mask projections may be occlusion",
              "shapes": {}}
    for label, geometry, extents in (("coarse", coarse, original_dimensions),
                                     ("candidate", refined, proposed_dimensions)):
        all_errors = np.concatenate(errors[label])
        nvalid, noverlap = totals[label]["valid"], totals[label]["overlap"]
        output["shapes"][label] = {
            "bbox_dimensions_in_audited_axes_m": extents.tolist(),
            "dimension_relative_error": (np.abs(extents-measure)/measure).tolist(),
            "coverage": noverlap/nvalid,
            "overlap_rmse_m": float(np.sqrt(np.mean(all_errors**2))) if len(all_errors) else None,
            "overlap_mae_m": float(np.mean(np.abs(all_errors))) if len(all_errors) else None,
            "observed_pixels": nvalid, "overlap_pixels": noverlap,
            "missing_pixels": nvalid-noverlap,
            **surface_distance(geometry, truth)}
    before, after = output["shapes"]["coarse"], output["shapes"]["candidate"]
    cfg = ledger["preregistered_protocol"]
    output["candidate_is_not_worse_on_frozen_metrics"] = bool(
        after["coverage"] >= cfg["coverage_min"] and
        after["coverage"] >= before["coverage"] - cfg["max_coverage_regression"] and
        after["overlap_rmse_m"] is not None and before["overlap_rmse_m"] is not None and
        after["overlap_rmse_m"] <= before["overlap_rmse_m"] + cfg["max_holdout_rmse_increase_m"] and
        np.max(after["dimension_relative_error"]) <=
        np.max(before["dimension_relative_error"]) + cfg["max_dimension_relative_error_increase"] and
        after["measured_to_surface_rmse_m"] <= before["measured_to_surface_rmse_m"] + .005)
    output["conclusion"] = "PASS_RESEARCH_SCREEN_NOT_PHYSICAL_TRUTH" if output[
        "candidate_is_not_worse_on_frozen_metrics"] else "FAIL_OR_INCONCLUSIVE"
    return output


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--proof", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.out.exists():
            raise DataError("Never overwrite prior comparison")
        result = evaluate(a.ledger, a.proof, a.root)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"decision": result["conclusion"], "report": str(a.out)}))
        return 0 if result["candidate_is_not_worse_on_frozen_metrics"] else 3
    except (DataError, ValueError, OSError, KeyError, TypeError) as e:
        import sys
        print(json.dumps({"status": "rejected", "error": str(e)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
