"""Real TUM XYZ: six measured RGB-D frames + six CPU learned RGB-only depths.

Same 60 original RGBs, TUM *reference* poses, 320x240 calibration and 1.8cm
Open3D TSDF as PR #164. This branch depends on #164 and #161, NOT main.
The six target sensor depths are NEVER loaded. Metric calibration comes only
from projecting the six selected measured depth frames into each RGB target.
No old mesh is used as a fusion prior; the 6-only mesh is reconstructed anew.
Depth Anything V2 is relative, not an independent metric measurement.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from echo4d.model import Frame, DataError, rigid_matrix, safe_path
from tum_public_roundtrip import ROOT, URL
from tum_xyz_depth_budget_48_24_12 import make_recording, diverse_pose_order
from tum_xyz_open3d_320 import (
    TRAIN_INDICES, RECORDED_INDICES, export_glb, free_view, fuse, panel, render,
)

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REV = "32d03942121d29edb49de4e2cc15831558af3f36"
WEIGHTS_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
VOXEL_M = .018


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CPUDepthAnything:
    """The same SHA-pinned Apache-2.0 checkpoint proven runnable in PR #157."""
    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        torch.set_num_threads(2)
        path = Path(snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REV,
            allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"]))
        if sha256(path / "model.safetensors") != WEIGHTS_SHA256:
            raise RuntimeError("Depth Anything V2 checkpoint hash mismatch")
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(str(path))
        self.model = AutoModelForDepthEstimation.from_pretrained(
            str(path), use_safetensors=True).to("cpu").eval()

    def infer(self, rgb):
        import torch.nn.functional as F
        batch = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        with self.torch.inference_mode():
            prediction = self.model(**batch).predicted_depth.unsqueeze(1)
            prediction = F.interpolate(prediction, size=rgb.shape[:2],
                                       mode="bicubic", align_corners=False)
        result = prediction[0, 0].cpu().numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("Model returned nonfinite relative depth")
        return result


def measured_projection(frames, camera, target_pose, *, stride=3):
    """Reproject ONLY selected measured depth to target image and z-buffer it.

    Camera axes right/down/forward, cameraToWorld column-major in recording.
    This function cannot read target depth, even for validation.
    """
    h, w = camera.height, camera.width
    zbuffer = np.full(h * w, np.inf, dtype=np.float32)
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    xx, yy = xx.reshape(-1), yy.reshape(-1)
    target_world_to_camera = np.linalg.inv(target_pose)
    for frame in frames:
        depth = frame.depth[::stride, ::stride].reshape(-1)
        valid = np.isfinite(depth) & (depth > .4) & (depth < 3.8)
        if not np.any(valid):
            continue
        d, x, y = depth[valid], xx[valid], yy[valid]
        xyz = np.stack(((x - camera.cx) / camera.fx * d,
                        (y - camera.cy) / camera.fy * d, d), axis=1)
        relative = target_world_to_camera @ frame.camera_to_world
        projected = xyz @ relative[:3, :3].T + relative[:3, 3]
        z = projected[:, 2]
        positive = (z > .4) & (z < 3.8)
        z, projected = z[positive], projected[positive]
        u = np.rint(projected[:, 0] * camera.fx / z + camera.cx).astype(np.int32)
        v = np.rint(projected[:, 1] * camera.fy / z + camera.cy).astype(np.int32)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        np.minimum.at(zbuffer, v[inside] * w + u[inside], z[inside])
    return zbuffer.reshape(h, w)


def calibrate_from_six(relative, measured_reprojection, *, support_radius=55):
    """Robust relative-disparity -> inverse-metric-depth fit from six sensor anchors.

    Pixels outside projected measured support are deliberately excluded.  The
    target's own sensor depth and any pre-existing mesh are never accessed.
    """
    if relative.shape != measured_reprojection.shape or relative.ndim != 2:
        raise ValueError("Prediction and projected sensor depth must share image shape")
    anchors = np.isfinite(measured_reprojection) & (relative > 0) & np.isfinite(relative)
    y, x = np.where(anchors)
    if len(x) < 200:
        raise RuntimeError(f"Insufficient projected measured support: {len(x)}")
    # Spatially subsample to avoid overweighting highly overlapping anchor rays.
    x, y = x[::3], y[::3]
    z = measured_reprojection[y, x].astype(np.float64)
    p = relative[y, x].astype(np.float64)
    A = np.column_stack((p, np.ones_like(p)))
    target = 1.0 / z
    weight = np.ones_like(z)
    for _ in range(7):
        coef = np.linalg.lstsq(A * weight[:, None], target * weight, rcond=None)[0]
        residual = A @ coef - target
        robust = max(float(np.median(np.abs(residual))) * 1.4826, .02)
        weight = np.clip(1.5 * robust / np.maximum(np.abs(residual), 1e-8), .05, 1)
    if not np.isfinite(coef).all() or coef[0] <= 0:
        raise RuntimeError(f"Relative depth cannot be calibrated as disparity: {coef}")
    estimate = 1 / np.maximum(A @ coef, 1e-6)
    inliers = np.abs(estimate - z) < np.maximum(.28, .22 * z)
    if inliers.sum() < 150 or inliers.mean() < .45:
        raise RuntimeError(f"Six-anchor metric fit failed: {inliers.sum()}/{len(z)}")
    coef = np.linalg.lstsq(A[inliers], target[inliers], rcond=None)[0]
    if coef[0] <= 0:
        raise RuntimeError("Refitted inverse-depth slope is nonpositive")
    denominator = relative.astype(np.float64) * coef[0] + coef[1]
    supported = distance_transform_edt(~anchors) <= support_radius
    valid = supported & (denominator > 1 / 3.8) & (denominator < 1 / .4)
    metric = np.zeros_like(relative, dtype=np.float32)
    metric[valid] = (1 / denominator[valid]).astype(np.float32)
    if valid.mean() < .02:
        raise RuntimeError("Predicted target has less than 2% calibrated coverage")
    return metric, {
        "projected_sensor_support": int(anchors.sum()),
        "calibration_samples": len(z),
        "calibration_inliers": int(inliers.sum()),
        "inverse_depth_slope": float(coef[0]),
        "inverse_depth_intercept": float(coef[1]),
        "valid_fraction": float(valid.mean()),
        "support_radius_px": support_radius,
        "calibration": "six selected measured depth frames reprojected using TUM reference poses",
        "target_sensor_depth_read": False,
    }


def test_calibration():
    h, w = 80, 100
    y, x = np.indices((h, w))
    true = (1.2 + .004 * x + .002 * y).astype(np.float32)
    relative = ((1 / true - .09) / .2).astype(np.float32)
    observed = np.full((h, w), np.inf, np.float32)
    observed[::3, ::3] = true[::3, ::3]
    metric, info = calibrate_from_six(relative, observed)
    valid = metric > 0
    assert valid.mean() > .8, info
    assert np.median(np.abs(metric[valid] - true[valid])) < .02, info
    # Geometry sanity: a same-pose projection must preserve sampled depths.
    from echo4d.model import Camera
    cam = Camera(w, h, 100., 100., 49.5, 39.5)
    f = Frame(0., np.eye(4), true, np.zeros((h, w, 3), np.uint8))
    projected = measured_projection([f], cam, np.eye(4), stride=3)
    assert np.isfinite(projected).sum() > 800
    assert np.max(np.abs(projected[::3, ::3] - true[::3, ::3])) < 1e-5
    print("PASS: six-anchor calibration and measured-only reprojection")


def experiment(args, receipt):
    import open3d as o3d
    import trimesh

    record, imported = make_recording(args, receipt)
    entries = record.entries
    order = diverse_pose_order(entries, TRAIN_INDICES)
    measured_ids = tuple(sorted(order[:6]))
    inferred_ids = tuple(sorted(order[6:12]))
    assert not (set(measured_ids) & set(inferred_ids))
    assert set(measured_ids).isdisjoint(set(range(0, 60, 5)))
    receipt.update(measured_indices=measured_ids, rgb_only_predicted_indices=inferred_ids,
                   source_rgb_frames=60, reference_poses="TUM provided (not RGB estimated)",
                   metric_scale="six measured TUM sensor depth frames + TUM reference poses",
                   mesh_prior=False, target_sensor_depth_read=False, gpu=False,
                   checkpoint=MODEL_ID, checkpoint_revision=MODEL_REV,
                   weights_sha256=WEIGHTS_SHA256, checkpoint_license="Apache-2.0")
    record.entries = [entries[i] for i in measured_ids]
    measured = [frame for _, frame, _ in record.iter_frames()]
    record.entries = entries
    if len(measured) != 6:
        raise RuntimeError("Expected exactly six real measured depth frames")
    baseline, _ = fuse(o3d, measured, record.camera, pixel_stride=1, voxel=VOXEL_M)
    args.output.mkdir(parents=True, exist_ok=True)
    export_glb(baseline, args.output / "tum-measured6-control.glb")
    predictor = CPUDepthAnything()
    predicted = []
    for index in inferred_ids:
        spec = entries[index]
        # Only the imported RGB path is opened. Never call iter_frames on a target.
        with Image.open(safe_path(record.root, spec["color"])) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (record.camera.height, record.camera.width, 3):
            raise DataError("Target RGB calibration differs from source camera")
        pose = rigid_matrix(spec["cameraToWorld"])
        relative = predictor.infer(rgb)
        projected = measured_projection(measured, record.camera, pose)
        metric, details = calibrate_from_six(relative, projected)
        predicted.append(Frame(float(spec["t"]), pose, metric, rgb).validate(record.camera))
        Image.fromarray(np.clip(metric * 1000, 0, 65535).astype(np.uint16)).save(
            args.output / f"predicted-{index:02d}-mm.png")
        receipt.setdefault("predictions", []).append(dict(index=index, **details))
        print(f"RGB-only target {index}: calibrated {details['valid_fraction']:.1%} pixels", flush=True)
    # The first six TSDF integrations are exactly the same as the control.
    hybrid, _ = fuse(o3d, measured + predicted, record.camera,
                     pixel_stride=1, voxel=VOXEL_M)
    export_glb(hybrid, args.output / "tum-measured6-plus-da6.glb")
    for name in ("tum-measured6-control.glb", "tum-measured6-plus-da6.glb"):
        obj = trimesh.load(str(args.output / name), force="mesh", process=False)
        if len(obj.faces) < 100 or obj.visual.kind != "vertex":
            raise RuntimeError(f"Exported GLB invalid: {name}")
        receipt.setdefault("meshes", {})[name] = {
            "faces": len(obj.faces), "sha256": sha256(args.output / name)}
    association = json.loads((imported / "association.json").read_text())
    original_root = args.work / "dataset" / ROOT
    rows = []
    views = []
    for index in RECORDED_INDICES:
        pose = rigid_matrix(entries[index]["cameraToWorld"])
        with Image.open(original_root / association["frames"][index]["rgb_path"]) as img:
            original = img.convert("RGB").copy()
        original.save(args.output / f"real-original-{index:02d}-640x480.png")
        photo = np.asarray(original.resize((320, 240), Image.Resampling.LANCZOS))
        old = render(o3d, baseline, record.camera, pose)
        new = render(o3d, hybrid, record.camera, pose)
        name = f"recorded-{index:02d}-6-vs-6plus6.png"
        rows.append(panel(args.output / name, [photo, old, new],
                          ["REAL RGB", "6 measured", "6 measured + 6 AI"]))
        views.append(dict(kind="recorded", index=index, file=name))
        gc.collect()
    for side, index in ((-1., 19), (1., 39)):
        pose = free_view(rigid_matrix(entries[index]["cameraToWorld"]), side)
        old = render(o3d, baseline, record.camera, pose)
        new = render(o3d, hybrid, record.camera, pose)
        name = f"free-{'left' if side < 0 else 'right'}-6-vs-6plus6.png"
        rows.append(panel(args.output / name,
                          [np.full_like(old, 20), old, new],
                          ["FREE VIEW", "6 measured", "6 measured + 6 AI"]))
        views.append(dict(kind="free", reference_index=index, file=name,
                          offset_m=.22, yaw_deg=side * 12))
        gc.collect()
    board = Image.new("RGB", (960, 267 * len(rows)))
    for i, row in enumerate(rows):
        board.paste(row, (0, 267 * i))
    board.save(args.output / "tum-six-vs-six-plus-da6-five-views.png")
    receipt.update(views=views, visual_verdict="PENDING real image inspection",
                   full_room="FAIL until continuous free-traversable scene shown",
                   historical_fail="desk 16.7779cm > 15cm and #74 remain FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        test_calibration()
        return 0
    if not args.work or not args.output:
        parser.error("--work and --output required")
    receipt = {
        "status": "running", "scene": "TUM Freiburg1 XYZ",
        "source": URL, "dependency": "PR #164 and high-info PR #161",
        "run_url": f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
        if os.getenv("GITHUB_RUN_ID") else None,
        "commands": [],
    }
    try:
        experiment(args, receipt)
        receipt["status"] = "REAL_TWO_MESHES_EXPORTED_VISUAL_REVIEW_PENDING"
        return 0
    except Exception as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": receipt["status"],
                          "error": receipt.get("error"),
                          "meshes": receipt.get("meshes")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
