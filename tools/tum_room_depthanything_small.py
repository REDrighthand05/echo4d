"""CPU Depth Anything V2 Small relative-Z -> known-pose RGB metric anchors -> real-room TSDF.

A child of #97. The learned model is NOT a metric sensor. Its scale/offset are
calibrated using independent RGB ZNCC matches and the already-known metric poses.
Neither measured depth pixels nor an existing Mesh enter candidate reconstruction.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

import tum_room_knownpose_multiview as base
import tum_room_multiview_zncc as zncc
from echo4d.model import DataError, file_sha256

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "870a35c76c2bc1d82fbde922d95015496cb7dd6c"
MODEL_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
_model = None


def calibrate_relative_depth(relative: np.ndarray, rgb_seed_m: np.ndarray
                             ) -> tuple[np.ndarray, dict]:
    """RANSAC fit 1/Z = a*relative + b, a>0; do not invent a metric scale.

    The sparse Z seeds come from 2+ independent RGB views and *known* TUM
    translations in metres. The learned relative depth alone is unscaled.
    """
    if (relative.shape != rgb_seed_m.shape or relative.ndim != 2
            or not np.isfinite(relative).all() or not np.isfinite(rgb_seed_m).all()
            or (rgb_seed_m < 0).any()):
        raise DataError("Expected finite, aligned nonnegative RGB-inferred Z")
    r = relative.astype(np.float64)
    lo, hi = np.percentile(r, [1, 99])
    if hi - lo < 1e-6:
        return np.zeros_like(rgb_seed_m, dtype=np.float32), {"status": "flat_relative_depth"}
    r = (r - lo) / (hi - lo)
    valid = (rgb_seed_m >= .45) & (rgb_seed_m <= 4.)
    indices = np.flatnonzero(valid)
    if len(indices) < 250:
        return np.zeros_like(rgb_seed_m, dtype=np.float32), {"status": "insufficient_rgb_metric_anchors",
                                                               "anchor_pixels": int(len(indices))}
    rng = np.random.default_rng(20261029)
    if len(indices) > 6000:
        indices = rng.choice(indices, 6000, replace=False)
    x = r.flat[indices]
    y = 1. / rgb_seed_m.flat[indices].astype(np.float64)
    best = None
    for _ in range(240):
        i, j = rng.integers(0, len(x), size=2)
        if abs(x[i] - x[j]) < .08:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        if not (.05 < a < 12.):
            continue
        b = y[i] - a * x[i]
        residual = np.abs(y - (a*x + b))
        inliers = residual < .12
        count = int(inliers.sum())
        score = (count, -float(np.median(residual[inliers])) if count else -1e9)
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[0][0] < max(200, int(.30 * len(x))):
        return np.zeros_like(rgb_seed_m, dtype=np.float32), {
            "status": "metric_calibration_failed", "anchor_pixels": int(len(x))}
    inliers = best[1]
    for _ in range(3):
        a, b = np.linalg.lstsq(
            np.column_stack((x[inliers], np.ones(inliers.sum()))),
            y[inliers], rcond=None)[0]
        if not (.05 < a < 12.):
            break
        inliers = np.abs(y - (a*x + b)) < .12
    if not (.05 < a < 12.) or inliers.sum() < max(200, int(.30 * len(x))):
        return np.zeros_like(rgb_seed_m, dtype=np.float32), {
            "status": "unstable_metric_calibration", "anchor_pixels": int(len(x))}
    inverse = a*r + b
    depth = np.zeros_like(rgb_seed_m, dtype=np.float32)
    plausible = (inverse >= .25) & (inverse <= 1/.45)
    # Predict only near RGB triangulation support; no extrapolated whole-room fill.
    distance = cv2.distanceTransform((~valid).astype(np.uint8), cv2.DIST_L2, 3)
    keep = plausible & (distance <= 18)
    depth[keep] = (1. / inverse[keep]).astype(np.float32)
    anchor_residual = np.abs(depth[valid] - rgb_seed_m[valid])
    return depth, {
        "status": "calibrated_from_known_pose_rgb_only",
        "anchor_pixels": int(len(indices)),
        "inlier_pixels": int(inliers.sum()),
        "inverse_depth_slope": float(a),
        "inverse_depth_intercept": float(b),
        "median_anchor_disagreement_m": float(np.median(anchor_residual)),
        "inferred_pixels": int(np.count_nonzero(depth)),
        "metric_scale_origin": "known TUM camera translations plus multi-view RGB matches",
        "unknown_pixels_not_filled": int(np.count_nonzero(~keep)),
        "relative_model": MODEL_ID, "model_revision": MODEL_REVISION}


def load_cpu_model():
    """Fetch a pinned, Apache-2.0 *Small* checkpoint, verify real bytes."""
    global _model
    if _model is None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        torch.set_num_threads(2)
        folder = Path(snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION,
            allow_patterns=["model.safetensors", "config.json", "preprocessor_config.json"]))
        if file_sha256(folder / "model.safetensors") != MODEL_SHA256:
            raise DataError("Depth Anything V2 Small checkpoint SHA256 mismatch")
        processor = AutoImageProcessor.from_pretrained(str(folder), local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(
            str(folder), local_files_only=True).to("cpu").eval()
        _model = (processor, model)
    return _model


def infer_relative(rgb: np.ndarray) -> np.ndarray:
    import torch
    from PIL import Image
    processor, model = load_cpu_model()
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
    with torch.inference_mode():
        pred = model(**inputs).predicted_depth
        resized = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic",
            align_corners=False).squeeze()
    return resized.cpu().numpy().astype(np.float32)


def learned_sweep(reference, sources, K, reference_pose, source_poses,
                  depths=base.DEPTHS_M):
    seed, receipt = zncc.sweep_zncc(
        reference, sources, K, reference_pose, source_poses, depths=depths)
    relative = infer_relative(reference)
    metric, fit = calibrate_relative_depth(relative, seed)
    return metric, {**receipt, "learned_depth_calibration": fit,
                    "depth_source": "Depth Anything V2 Small relative RGB + known-pose RGB ZNCC metric calibration; no sensor Z",
                    "valid_pixels": int(np.count_nonzero(metric))}


_original_build_manifest = base.build_manifest


def build_manifest_da2(imported, target, report):
    import json
    manifest = _original_build_manifest(imported, target, report)
    data = json.loads(manifest.read_text())
    data["name"] = "TUM room known-pose RGB Depth Anything V2 Small inferred Z"
    data["notes"] = (
        "DA2 Small relative-depth prediction; metric scale from independent "
        "known-pose RGB ZNCC anchors, never measured sensor depth")
    manifest.write_text(json.dumps(data, indent=2, allow_nan=False) + "\\n")
    return manifest


_original_run = base.run


def run_with_model(args, report):
    report["algorithm_variant"] = "DA2-Small relative depth, robust RGB-known-pose metric fit, TSDF"
    report["model_license"] = "Apache-2.0, Small only (not Base/Large/Giant)"
    report["model_checkpoint_sha256"] = MODEL_SHA256
    report["model_revision"] = MODEL_REVISION
    report["no_gpu_claim"] = True
    _original_run(args, report)
    import json
    # The inherited TSDF provider emits a plane-sweep-only provenance label.
    # Correct that label and its SHA without changing vertices or triangles.
    ply = args.report_dir / "multiview-mesh" / "mesh.ply"
    old = b"RGB multi-view plane-sweep inferred Z; known metric poses; no sensor Z"
    new = b"DA2 Small relative Z plus known-pose RGB metric anchors; no sensor Z"
    content = ply.read_bytes()
    if content.count(old) != 1:
        raise DataError("Expected exactly one inherited PLY provenance label")
    ply.write_bytes(content.replace(old, new, 1))
    fusion_path = args.report_dir / "multiview-mesh" / "report.json"
    fusion = json.loads(fusion_path.read_text())
    fusion["geometry_source"] = (
        "Depth Anything V2 Small relative depth with RGB-known-pose metric fit; no sensor Z")
    fusion["mesh_sha256"] = file_sha256(ply)
    fusion_path.write_text(json.dumps(fusion, indent=2, allow_nan=False) + "\\n")
    report["fusion"] = fusion
    report["new_mesh"]["sha256"] = file_sha256(ply)
    previous_dir = os.environ.get("ECHO_PREVIOUS_ZNCC")
    if previous_dir:
        import json
        import trimesh
        from echo4d.holdout import read_echo_ply
        from tum_graph_visual import read_vertex_rgb
        previous = Path(previous_dir)
        old_report = json.loads((previous / "report.json").read_text())
        old_mesh = previous / "multiview-mesh" / "mesh.ply"
        new_mesh = args.report_dir / "multiview-mesh" / "mesh.ply"
        if (file_sha256(old_mesh) != old_report["new_mesh"]["sha256"]
                or old_report["source_archive_sha256"] != report["source_archive_sha256"]
                or old_report["selection"]["expanded_indices"]
                   != report["selection"]["expanded_indices"]):
            raise DataError("Previous ZNCC control source/training/mesh mismatch")
        high = json.loads((args.baseline / "room-sparse-quality-report.json").read_text())
        report["visual_zncc_vs_da2"] = base.compare_multiview_five(
            args.work / "room-import", old_mesh, new_mesh, high,
            args.report_dir / "visual-zncc-vs-da2",
            "previous known-pose RGB ZNCC / no prior Mesh")
        vv, ff = read_echo_ply(old_mesh)
        cc = read_vertex_rgb(old_mesh, len(vv))
        old_glb = args.report_dir / "old-zncc-rgb.glb"
        trimesh.Trimesh(vertices=vv, faces=ff, vertex_colors=cc,
                        process=False).export(old_glb)
        report["old_zncc_control"] = {
            "mesh_sha256": file_sha256(old_mesh),
            "glb_sha256": file_sha256(old_glb),
            "role": "posthoc A/B only; never candidate input"}
    report["issue_74"] = "FAIL / NOT ACCEPTED until human reviews 3+2 actual Mesh"
    report["status"] = "real_room_DA2_small_mesh_generated_visual_review_required"


def main():
    base.sweep_depth = learned_sweep
    base.build_manifest = build_manifest_da2
    base.run = run_with_model
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
