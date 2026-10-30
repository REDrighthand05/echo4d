"""Real Fountain RGB depth ablation: retain 16 sensors, guard 8 RGB predictions.

A second, geometrically different estimator checks predicted target surfaces
against *only the 16 retained* depth/RGB cameras. It removes multi-view
contradicted foreground and snaps well-supported surfaces to measured geometry.
The original unguarded 16+8 reconstruction is generated in the same run.
Target sensor depth is opened only after BOTH reduced-input meshes are saved.
No supplied scan PRIOR, no target depth in either estimate, no GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from echo4d.model import Camera, DataError, Frame
from fountain_cpu_rgb_depth8 import (
    CPUDepthAnything, DEPTH_MIN, DEPTH_MAX, file_sha256,
    metric_from_relative, project_sensor_anchors, select_rgb_targets,
)
from fountain_depth_ladder import choose_fountain_indices, SOURCE_INDICES, FREE_INDICES, VOXEL_M
from tum_xyz_open3d_320 import export_glb, free_view, fuse, panel, render


def multiview_guard(metric, target_rgb, target_pose, measured, camera):
    """Conservative RGB-D reprojection test; never accesses target sensor depth.

    A predicted point that lies in front of two or more *observed* surfaces,
    with no corroborating source view, is discarded. A point agreeing in
    depth AND color with >=2 measured views can be snapped to a nearby
    sensor-reprojected depth to prevent duplicated walls. Unobserved pixels
    remain candidates, not fabricated confirmations.
    """
    h, w = metric.shape
    if target_rgb.shape != (h, w, 3) or not measured:
        raise ValueError("Need target RGB and retained sensor views")
    if camera.width != w or camera.height != h:
        raise ValueError("Camera/depth resolution mismatch")
    valid = np.isfinite(metric) & (metric > DEPTH_MIN) & (metric < DEPTH_MAX)
    yy, xx = np.indices((h, w), dtype=np.float32)
    z = np.where(valid, metric, 1.).astype(np.float32)
    xyz = np.stack(((xx-camera.cx)*z/camera.fx,
                    (yy-camera.cy)*z/camera.fy, z), axis=-1).reshape(-1, 3)
    valid_flat = valid.ravel()
    target_colors = target_rgb.reshape(-1, 3).astype(np.int16)
    confirmations = np.zeros(h*w, np.uint8)
    contradictions = np.zeros(h*w, np.uint8)
    tested = np.zeros(h*w, np.uint8)
    for source in measured:
        if source.depth.shape != (h, w) or source.color.shape != (h, w, 3):
            raise ValueError("Measured frame has inconsistent dimensions")
        t = np.linalg.inv(source.camera_to_world) @ target_pose
        projected = xyz @ t[:3, :3].T + t[:3, 3]
        zs = projected[:, 2]
        front = valid_flat & (zs > DEPTH_MIN) & (zs < DEPTH_MAX)
        safe_z = np.maximum(zs, .001)
        u = np.rint(projected[:, 0]*camera.fx/safe_z+camera.cx).astype(np.int32)
        v = np.rint(projected[:, 1]*camera.fy/safe_z+camera.cy).astype(np.int32)
        inside = front & (u>=0) & (u<w) & (v>=0) & (v<h)
        ids = np.flatnonzero(inside)
        if not len(ids):
            continue
        depth = source.depth[v[ids], u[ids]]
        has_sensor = np.isfinite(depth) & (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
        ids = ids[has_sensor]
        depth = depth[has_sensor]
        if not len(ids):
            continue
        tested[ids] += 1
        colors = source.color[v[ids], u[ids]].astype(np.int16)
        color_error = np.abs(colors-target_colors[ids]).mean(axis=1)
        dz = zs[ids]-depth
        same_surface = (np.abs(dz) <= np.maximum(.10, .075*depth))
        confirmations[ids] += (same_surface & (color_error < 42)).astype(np.uint8)
        # Predicted point is in front of an actually measured surface.
        # Large disagreement is geometric evidence even on plain-color walls.
        in_front = dz < -np.maximum(.17, .10*depth)
        strong = dz < -np.maximum(.33, .18*depth)
        contradictions[ids] += (in_front & ((color_error > 35) | strong)).astype(np.uint8)

    remove = valid_flat & (contradictions >= 2) & (confirmations == 0)
    guarded = metric.copy().reshape(-1)
    guarded[remove] = 0.

    # Reproject only the retained 16 sensor depths. A 1-pixel neighborhood
    # accounts for rasterization, while >=2 independently agreeing views
    # guard against snapping across foreground/background boundaries.
    projected = project_sensor_anchors(measured, camera, target_pose, stride=2)
    support = np.isfinite(projected)
    snap_count = 0
    if np.any(support):
        distance, indices = distance_transform_edt(~support, return_indices=True)
        anchor_z = projected[tuple(indices)].ravel()
        near = (distance.ravel() <= 1.5) & valid_flat & (~remove)
        close = np.abs(metric.ravel()-anchor_z) < np.maximum(.10, .08*anchor_z)
        snap = near & close & (confirmations >= 2)
        guarded[snap] = anchor_z[snap]
        snap_count = int(snap.sum())

    guarded = guarded.reshape(h, w).astype(np.float32)
    if not np.isfinite(guarded).all():
        raise RuntimeError("Nonfinite guarded depth")
    return guarded, {
        "valid_before": int(valid.sum()),
        "valid_after": int(np.count_nonzero(guarded)),
        "multi_view_confirmed_pixels": int(np.count_nonzero(confirmations >= 2)),
        "multi_view_contradicted_pixels": int(remove.sum()),
        "sensor_snapped_pixels": snap_count,
        "sensor_observed_pixels": int(np.count_nonzero(tested)),
    }


def run(work: Path, output: Path, receipt: dict):
    import open3d as o3d
    import trimesh

    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    dataset = o3d.data.SampleFountainRGBDImages(data_root=str(work))
    trajectory = o3d.io.read_pinhole_camera_trajectory(dataset.keyframe_poses_log_path)
    params = trajectory.parameters
    if not (len(dataset.color_paths) == len(dataset.depth_paths) == len(params) == 33):
        raise DataError("Fountain source RGB-D and poses changed")
    intr = params[0].intrinsic
    k = intr.intrinsic_matrix
    full = Camera(intr.width, intr.height, float(k[0, 0]), float(k[1, 1]),
                  float(k[0, 2]), float(k[1, 2]))
    camera = full.scaled(2)
    poses = [np.linalg.inv(np.asarray(p.extrinsic, np.float64)) for p in params]
    _, measured_ids = choose_fountain_indices(poses)
    targets = select_rgb_targets(measured_ids)
    holdouts = sorted(set(range(33)) - set(measured_ids) - set(targets))
    colors = []
    for path in dataset.color_paths:
        with Image.open(path) as img:
            rgb = np.asarray(img.convert("RGB"), np.uint8)
        if rgb.shape != (full.height, full.width, 3):
            raise DataError("Unexpected native RGB dimensions")
        colors.append(rgb)
    measured = []
    for i in measured_ids:
        with Image.open(dataset.depth_paths[i]) as img:
            raw = np.asarray(img)
        depth = np.ascontiguousarray(raw[::2, ::2].astype(np.float32)/1000.)
        depth[(depth < .2) | (depth > 4.)] = 0.
        measured.append(Frame(float(i), poses[i], depth,
            np.ascontiguousarray(colors[i][::2, ::2])).validate(camera))
    receipt.update(scene="Open3D real Fountain, 33 native 640x480 RGB",
        source="Open3D SampleFountainRGBDImages",
        measured_sensor_indices=list(measured_ids), rgb_only_indices=list(targets),
        holdout_indices=holdouts, supplied_scan_prior_used=False,
        target_sensor_depth_used_in_reconstruction=False,
        reference_poses="provided calibrated poses, not estimated",
        metric_scale="retained 16 measured depth frames",
        predicted_depth="Depth Anything V2 Small, CPU, pinned hash from #182",
        baseline="16 measured + 8 original CPU depth predictions",
        new_method="RGB-D multi-view visibility and anchor-surface snapping",
        historical_fail="#74 full-room FAIL; Desk 16.7779cm >15cm FAIL",
        predictions=[], meshes={}, views=[])
    predictor = CPUDepthAnything()
    original_frames, guarded_frames, predictions = [], [], {}
    for i in targets:
        relative = np.ascontiguousarray(predictor.infer(colors[i])[::2, ::2])
        anchors = project_sensor_anchors(measured, camera, poses[i])
        try:
            metric, calibration = metric_from_relative(relative, anchors)
        except RuntimeError as error:
            receipt["predictions"].append({"index":i,"used":False,"reason":str(error)})
            continue
        guard, details = multiview_guard(metric, colors[i][::2, ::2],
                                        poses[i], measured, camera)
        original_frames.append(Frame(float(i), poses[i], metric,
            np.ascontiguousarray(colors[i][::2, ::2])).validate(camera))
        guarded_frames.append(Frame(float(i), poses[i], guard,
            np.ascontiguousarray(colors[i][::2, ::2])).validate(camera))
        predictions[i] = (metric, guard)
        Image.fromarray(np.clip(metric*1000, 0, 65535).astype(np.uint16)).save(
            output / f"rgb-original-{i:02d}-millimetres.png")
        Image.fromarray(np.clip(guard*1000, 0, 65535).astype(np.uint16)).save(
            output / f"rgb-guarded-{i:02d}-millimetres.png")
        receipt["predictions"].append({"index":i, "used":True, **calibration, **details})
        print("RGB", i, "guard", details, flush=True)
    del predictor
    gc.collect()
    if len(original_frames) < 3:
        raise RuntimeError("Too few RGB predictions with sensor-supported metric scale")

    view_poses = [poses[i] for i in SOURCE_INDICES]
    view_poses += [free_view(poses[i], side) for i, side in FREE_INDICES]
    rendered = {}
    for name, frames in (
        ("measured16", measured),
        ("original16plus8", measured + original_frames),
        ("guarded16plus8", measured + guarded_frames),
    ):
        mesh, _ = fuse(o3d, frames, camera, pixel_stride=1, voxel=VOXEL_M)
        filename = f"fountain-{name}.glb"
        export_glb(mesh, output / filename)
        rendered[name] = [render(o3d, mesh, camera, pose) for pose in view_poses]
        receipt["meshes"][name] = {"glb":filename,
            "sensor_frames":16,"rgb_predictions":len(frames)-16}
        del mesh
        gc.collect()

    # Full-sensor reference is a separately labeled upper bound. No existing scan.
    # Target depth is not opened until ALL THREE reduced-input GLBs exist.
    all_frames = list(measured)
    for i in range(33):
        if i in measured_ids:
            continue
        with Image.open(dataset.depth_paths[i]) as img:
            raw = np.asarray(img)
        depth = np.ascontiguousarray(raw[::2, ::2].astype(np.float32)/1000.)
        depth[(depth < .2) | (depth > 4.)] = 0.
        all_frames.append(Frame(float(i), poses[i], depth,
            np.ascontiguousarray(colors[i][::2, ::2])).validate(camera))
    upper, _ = fuse(o3d, all_frames, camera, pixel_stride=1, voxel=VOXEL_M)
    export_glb(upper, output / "fountain-measured33-upper.glb")
    rendered["measured33"] = [render(o3d, upper, camera, pose) for pose in view_poses]
    receipt["meshes"]["measured33"] = {
        "glb":"fountain-measured33-upper.glb", "sensor_frames":33,
        "rgb_predictions":0, "upper_reference_only":True}
    del upper, all_frames
    gc.collect()

    # Independent post-hoc target-depth check, never fed into fusion.
    checks = []
    for i, (original, guard) in predictions.items():
        with Image.open(dataset.depth_paths[i]) as img:
            actual = np.asarray(img)[::2, ::2].astype(np.float32)/1000.
        pair = {"index":i}
        for name, depth in (("original",original),("guarded",guard)):
            good = ((actual>DEPTH_MIN) & (actual<DEPTH_MAX) & (depth>0))
            pair[name+"_median_abs_error_m"] = (
                float(np.median(np.abs(actual[good]-depth[good])))
                if good.any() else None)
            pair[name+"_valid_pixels"] = int(good.sum())
        checks.append(pair)
    receipt["posthoc_target_sensor_check"] = checks
    for spec in receipt["meshes"].values():
        path = output / spec["glb"]
        mesh = trimesh.load(str(path), force="mesh", process=False)
        if len(mesh.faces) < 1000 or mesh.visual.kind != "vertex":
            raise DataError("Invalid colored GLB: " + str(path))
        spec["triangles"] = len(mesh.faces)
        spec["sha256"] = file_sha256(path)

    boards = []
    for row, pose in enumerate(view_poses):
        if row < 3:
            i = SOURCE_INDICES[row]
            Image.fromarray(colors[i]).save(output / f"source-{i:02d}-native.png")
            photo = np.asarray(Image.fromarray(colors[i]).resize(
                (camera.width,camera.height), Image.Resampling.LANCZOS))
            kind = "recorded"
        else:
            i = FREE_INDICES[row-3][0]
            photo = np.full((camera.height,camera.width,3),20,np.uint8)
            kind = "free"
        name = f"{kind}-{i:02d}-multiview-guard.png"
        boards.append(panel(output / name,
            [photo,rendered["measured33"][row],rendered["measured16"][row],
             rendered["original16plus8"][row],rendered["guarded16plus8"][row]],
            ["REAL RGB" if kind=="recorded" else "FREE / NO PHOTO",
             "33 SENSOR","16 SENSOR","16+8 CPU RGB","16+8 RGB GUARDED"]))
        receipt["views"].append({"kind":kind,"index":i,"file":name})
    board = Image.new("RGB",(camera.width*5,(camera.height+27)*5))
    for row, panel_image in enumerate(boards):
        board.paste(panel_image,(0,row*(camera.height+27)))
    board.save(output/"fountain-original-vs-multiview-guard-five-views.png")
    receipt["status"] = "four_real_glbs_and_five_views_visual_review_pending"
    receipt["visual_verdict"] = "PENDING human inspection; complete room FAIL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    receipt = {"status":"running", "run_url":
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
