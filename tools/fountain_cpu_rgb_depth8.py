"""Real Fountain: 16 measured RGB-D frames + 8 RGB-only CPU depth estimates.

Same official 33-view scene and calibrated camera poses as #179/#180.
The eight target sensor-depth files are NEVER opened until after both GLBs
have been exported. The supplied scan PRIOR is never used.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from echo4d.model import Camera, DataError, Frame
from fountain_depth_ladder import choose_fountain_indices, SOURCE_INDICES, FREE_INDICES, VOXEL_M
from tum_xyz_open3d_320 import export_glb, free_view, fuse, panel, render

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REV = "32d03942121d29edb49de4e2cc15831558af3f36"
WEIGHTS_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
DEPTH_MIN, DEPTH_MAX = .4, 3.8


def select_rgb_targets(measured, n=33, count=8):
    """Deterministic temporal spread, disjoint from sensor anchors."""
    remaining = tuple(i for i in range(n) if i not in set(measured))
    if len(measured) != 16 or len(remaining) < count or count < 2:
        raise ValueError("Need 16 distinct measured anchors and enough RGB views")
    targets = tuple(remaining[round(j * (len(remaining)-1) / (count-1))]
                    for j in range(count))
    if len(set(targets)) != count or set(targets) & set(measured):
        raise ValueError("RGB-only target selection overlaps sensor frames")
    return targets


def project_sensor_anchors(measured, camera, target_pose, stride=3):
    """Only retained sensor rays, reprojected into target RGB camera."""
    h, w = camera.height, camera.width
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    xx, yy = xx.ravel(), yy.ravel()
    zbuffer = np.full(h*w, np.inf, np.float32)
    target_inv = np.linalg.inv(target_pose)
    for frame in measured:
        z = frame.depth[::stride, ::stride].ravel()
        valid = np.isfinite(z) & (z > DEPTH_MIN) & (z < DEPTH_MAX)
        if not np.any(valid):
            continue
        depth = z[valid]
        xyz = np.column_stack(((xx[valid]-camera.cx)*depth/camera.fx,
                               (yy[valid]-camera.cy)*depth/camera.fy, depth))
        t = target_inv @ frame.camera_to_world
        xyz = xyz @ t[:3, :3].T + t[:3, 3]
        zz = xyz[:, 2]
        good = (zz > DEPTH_MIN) & (zz < DEPTH_MAX)
        xyz, zz = xyz[good], zz[good]
        u = np.rint(xyz[:, 0]*camera.fx/zz + camera.cx).astype(np.int32)
        v = np.rint(xyz[:, 1]*camera.fy/zz + camera.cy).astype(np.int32)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        np.minimum.at(zbuffer, v[inside]*w + u[inside], zz[inside])
    return zbuffer.reshape(h, w)


def metric_from_relative(relative, anchors, radius=70):
    """Robust inverse-depth scale fit, only from retained measured rays."""
    if relative.shape != anchors.shape or relative.ndim != 2:
        raise ValueError("Depth image size mismatch")
    support = np.isfinite(anchors) & np.isfinite(relative) & (relative > 0)
    y, x = np.where(support)
    if len(x) < 180:
        raise RuntimeError(f"Too few sensor reprojections: {len(x)}")
    x, y = x[::2], y[::2]
    z = anchors[y, x].astype(np.float64)
    p = relative[y, x].astype(np.float64)
    A = np.column_stack((p, np.ones_like(p)))
    target = 1. / z
    weights = np.ones(len(z))
    for _ in range(7):
        coef = np.linalg.lstsq(A*weights[:, None], target*weights, rcond=None)[0]
        residual = A @ coef - target
        sigma = max(float(np.median(np.abs(residual)))*1.4826, .02)
        weights = np.clip(1.5*sigma / np.maximum(np.abs(residual), 1e-8), .05, 1.)
    if not np.isfinite(coef).all() or coef[0] <= 0:
        raise RuntimeError("Invalid RGB relative-depth metric scale")
    estimated = 1. / np.maximum(A @ coef, 1e-6)
    inlier = np.abs(estimated-z) < np.maximum(.28, .22*z)
    if inlier.sum() < 120 or inlier.mean() < .40:
        raise RuntimeError(f"Insufficient sensor-consistent support: {inlier.sum()}")
    coef = np.linalg.lstsq(A[inlier], target[inlier], rcond=None)[0]
    if coef[0] <= 0:
        raise RuntimeError("Negative inverse-depth scale")
    inv = relative.astype(np.float64)*coef[0] + coef[1]
    valid = ((distance_transform_edt(~support) <= radius) &
             (inv > 1./DEPTH_MAX) & (inv < 1./DEPTH_MIN))
    metric = np.zeros(relative.shape, np.float32)
    metric[valid] = (1./inv[valid]).astype(np.float32)
    # Reject observed sensor conflicts rather than inventing a second surface.
    conflicts = support & (metric > 0) & (
        np.abs(metric-anchors) > np.maximum(.18, .15*anchors))
    metric[conflicts] = 0
    if np.count_nonzero(metric) < metric.size*.02:
        raise RuntimeError("Less than 2% trustworthy predicted depth")
    return metric, {"sensor_projection_pixels": int(support.sum()),
                    "calibration_inliers": int(inlier.sum()),
                    "predicted_valid_fraction": float(np.mean(metric > 0)),
                    "rejected_sensor_conflicts": int(conflicts.sum())}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CPUDepthAnything:
    """Previously executed Apache-2.0 checkpoint, pinned and hash-checked."""
    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        torch.set_num_threads(2)
        root = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REV,
            allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"]))
        if file_sha256(root / "model.safetensors") != WEIGHTS_SHA256:
            raise RuntimeError("Depth Anything checkpoint SHA256 mismatch")
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(str(root))
        self.model = AutoModelForDepthEstimation.from_pretrained(
            str(root), use_safetensors=True).to("cpu").eval()

    def infer(self, rgb):
        import torch.nn.functional as F
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        with self.torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth.unsqueeze(1)
            prediction = F.interpolate(prediction, size=rgb.shape[:2],
                                       mode="bicubic", align_corners=False)
        output = prediction[0, 0].cpu().numpy().astype(np.float32)
        if not np.isfinite(output).all():
            raise RuntimeError("Nonfinite CPU predicted depth")
        return output


def run(work, output, receipt):
    import open3d as o3d
    import trimesh
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    dataset = o3d.data.SampleFountainRGBDImages(data_root=str(work))
    trajectory = o3d.io.read_pinhole_camera_trajectory(dataset.keyframe_poses_log_path)
    params = trajectory.parameters
    if not (len(dataset.color_paths) == len(dataset.depth_paths) == len(params) == 33):
        raise DataError("Official Fountain RGB/depth/pose alignment changed")
    intrinsic = params[0].intrinsic
    k = intrinsic.intrinsic_matrix
    full_camera = Camera(intrinsic.width, intrinsic.height, float(k[0,0]),
                         float(k[1,1]), float(k[0,2]), float(k[1,2]))
    camera = full_camera.scaled(2)
    poses = [np.linalg.inv(np.asarray(p.extrinsic, np.float64)) for p in params]
    _, measured_ids = choose_fountain_indices(poses)
    targets = select_rgb_targets(measured_ids)
    holdouts = tuple(i for i in range(33) if i not in set(measured_ids) | set(targets))
    # Load RGB for all 33 frames. Open sensor depth ONLY for 16 retained anchors.
    originals, colors = {}, []
    for i, path in enumerate(dataset.color_paths):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), np.uint8)
        if rgb.shape != (full_camera.height, full_camera.width, 3):
            raise DataError(f"Original RGB {i} has wrong shape")
        if i in SOURCE_INDICES:
            originals[i] = rgb.copy()
        colors.append(rgb)
    measured = []
    for i in measured_ids:
        with Image.open(dataset.depth_paths[i]) as image:
            raw = np.asarray(image)
        if raw.shape != (full_camera.height, full_camera.width):
            raise DataError(f"Measured sensor depth {i} size mismatch")
        depth = np.ascontiguousarray(raw[::2, ::2].astype(np.float32) / 1000.)
        depth[(depth < .2) | (depth > 4.)] = 0
        measured.append(Frame(float(i), poses[i], depth,
                              np.ascontiguousarray(colors[i][::2, ::2])).validate(camera))
    receipt.update(scene="Open3D official real Fountain", source_resolution=[
        full_camera.width, full_camera.height], fusion_resolution=[camera.width,camera.height],
        retained_sensor_indices=list(measured_ids), rgb_only_target_indices=list(targets),
        independent_holdout_indices=list(holdouts), reference_poses="Open3D supplied",
        metric_scale="16 retained real sensor depths, calibrated known poses",
        model=MODEL_ID, model_revision=MODEL_REV, model_weights_sha256=WEIGHTS_SHA256,
        model_license="Apache-2.0", cpu_only=True, supplied_scan_prior_used=False,
        target_sensor_depth_used_in_reconstruction=False,
        historical_fail="#74 full-room FAIL; Desk 16.7779cm >15cm FAIL",
        predictions=[], meshes={}, views=[])
    view_poses = [poses[i] for i in SOURCE_INDICES]
    view_poses += [free_view(poses[i], side) for i, side in FREE_INDICES]
    renderings = {}
    for label, frames in (("measured16", measured),):
        mesh, _ = fuse(o3d, frames, camera, pixel_stride=1, voxel=VOXEL_M)
        name = f"fountain-{label}.glb"
        export_glb(mesh, output / name)
        renderings[label] = [render(o3d, mesh, camera, pose) for pose in view_poses]
        receipt["meshes"][label] = {"glb":name, "sensor_frames":len(frames),
                                    "rgb_predicted_frames":0}
        del mesh
        gc.collect()
    predictor = CPUDepthAnything()
    predicted_frames = []
    predictions = {}
    for i in targets:
        relative_full = predictor.infer(colors[i])
        relative = np.ascontiguousarray(relative_full[::2, ::2])
        projected = project_sensor_anchors(measured, camera, poses[i])
        try:
            metric, details = metric_from_relative(relative, projected)
        except RuntimeError as error:
            receipt["predictions"].append({"index":i, "used":False, "reason":str(error)})
            print(f"Skip RGB-only frame {i}: {error}", flush=True)
            continue
        predicted_frames.append(Frame(float(i), poses[i], metric,
            np.ascontiguousarray(colors[i][::2, ::2])).validate(camera))
        predictions[i] = metric
        Image.fromarray(np.clip(metric*1000, 0, 65535).astype(np.uint16)).save(
            output / f"rgb-predicted-depth-{i:02d}-millimetres.png")
        receipt["predictions"].append({"index":i, "used":True, **details})
        print(f"CPU RGB frame {i}: {details['predicted_valid_fraction']:.1%} supported", flush=True)
    del predictor
    gc.collect()
    if len(predicted_frames) < 3:
        raise RuntimeError(f"Only {len(predicted_frames)} RGB frames had sensor-supported scale")
    hybrid, _ = fuse(o3d, measured+predicted_frames, camera,
                     pixel_stride=1, voxel=VOXEL_M)
    export_glb(hybrid, output / "fountain-measured16-plus-rgb8.glb")
    renderings["hybrid"] = [render(o3d, hybrid, camera, pose) for pose in view_poses]
    receipt["meshes"]["hybrid"] = {"glb":"fountain-measured16-plus-rgb8.glb",
                                   "sensor_frames":16,
                                   "rgb_predicted_frames":len(predicted_frames)}
    del hybrid
    gc.collect()
    # Export same-scene full-sensor upper reference; NO supplied mesh prior.
    all_frames = measured.copy()
    for i in range(33):
        if i in measured_ids:
            continue
        with Image.open(dataset.depth_paths[i]) as image:
            raw = np.asarray(image)
        depth = np.ascontiguousarray(raw[::2, ::2].astype(np.float32)/1000.)
        depth[(depth < .2) | (depth > 4.)] = 0
        all_frames.append(Frame(float(i), poses[i], depth,
            np.ascontiguousarray(colors[i][::2, ::2])).validate(camera))
    upper, _ = fuse(o3d, all_frames, camera, pixel_stride=1, voxel=VOXEL_M)
    export_glb(upper, output / "fountain-measured33-upper.glb")
    renderings["measured33"] = [render(o3d, upper, camera, pose) for pose in view_poses]
    receipt["meshes"]["measured33"] = {"glb":"fountain-measured33-upper.glb",
                                       "sensor_frames":33, "rgb_predicted_frames":0,
                                       "upper_reference_only":True}
    del upper, all_frames
    gc.collect()
    # Target sensor depth files are opened only AFTER BOTH reduced-input GLBs
    # have been exported. The 33-depth upper reference is a separate control.
    checks = []
    for i, predicted in predictions.items():
        with Image.open(dataset.depth_paths[i]) as image:
            sensor = np.asarray(image)[::2, ::2].astype(np.float32)/1000.
        valid = (sensor > DEPTH_MIN) & (sensor < DEPTH_MAX) & (predicted > 0)
        if valid.any():
            checks.append({"index":i, "median_abs_depth_error_m":
                           float(np.median(np.abs(sensor[valid]-predicted[valid]))),
                           "pixels":int(valid.sum())})
    receipt["posthoc_target_sensor_check"] = checks
    for spec in receipt["meshes"].values():
        path = output / spec["glb"]
        loaded = trimesh.load(str(path), force="mesh", process=False)
        if len(loaded.faces) < 1000 or loaded.visual.kind != "vertex":
            raise DataError(f"Invalid real colored Mesh: {path}")
        spec["sha256"] = file_sha256(path)
        spec["triangles"] = len(loaded.faces)
    boards = []
    for row, pose in enumerate(view_poses):
        if row < 3:
            i = SOURCE_INDICES[row]
            Image.fromarray(originals[i]).save(output / f"source-{i:02d}-native.png")
            photo = np.asarray(Image.fromarray(originals[i]).resize(
                (camera.width, camera.height), Image.Resampling.LANCZOS))
            kind = "recorded"
        else:
            i = FREE_INDICES[row-3][0]
            photo = np.full((camera.height,camera.width,3), 20, np.uint8)
            kind = "free"
        name = f"{kind}-{i:02d}-rgb-depth-compare.png"
        boards.append(panel(output / name, [photo, renderings["measured33"][row],
            renderings["measured16"][row], renderings["hybrid"][row]],
            ["REAL RGB" if kind=="recorded" else "FREE / NO PHOTO",
             "33 SENSOR", "16 SENSOR", "16 SENSOR + CPU RGB"]))
        receipt["views"].append({"kind":kind,"index":i,"file":name})
    board = Image.new("RGB",(camera.width*4,(camera.height+27)*5))
    for row, image in enumerate(boards):
        board.paste(image,(0,row*(camera.height+27)))
    board.save(output / "fountain-33-vs-16-vs-16plus8rgb-five-views.png")
    receipt["visual_verdict"] = "PENDING real five-view human inspection; no full-room claim"
    receipt["status"] = "real_three_glbs_exported_visual_review_pending"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    receipt = {"status":"running","run_url":
        f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
        if os.getenv("GITHUB_RUN_ID") else None}
    try:
        run(args.work,args.output,receipt)
        code = 0
    except Exception as error:
        traceback.print_exc()
        receipt.update(status="FAIL",error=f"{type(error).__name__}: {error}")
        code = 1
    finally:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/"result.json").write_text(
            json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
        print(json.dumps({"status":receipt["status"],"meshes":receipt.get("meshes"),
                          "error":receipt.get("error")}),flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
