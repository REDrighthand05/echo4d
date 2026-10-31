"""Real Stanford Lounge: remove half the measured depth, restore 24 RGB-only views.

144 spatially distributed measured RGB-D frames are the upper reference (#181).
72 diverse sensor RGB-D anchors are retained; 24 other views have RGB and known
reference poses only. CPU Depth Anything V2 estimates relative depth, calibrated
solely with projections of the 72 retained measured frames. The withheld target
sensor-depth files are read only AFTER reduced-input GLBs are exported.
No supplied lounge.ply scan PRIOR is loaded or used.
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

from echo4d.model import Camera, DataError, Frame
from lounge_spatial_prior_real import VOXEL_M, read_frames, select_views
from lounge_rgb_gap_select import (choose_gap_targets, score_gap_views,
                                   conservative_hole_depth)
from lounge_rgb_depth_adapter import (
    CPUDepthAnything, DEPTH_MIN, DEPTH_MAX, MODEL_ID, MODEL_REV,
    WEIGHTS_SHA256, file_sha256, metric_from_relative, project_sensor_anchors,
)
from tum_xyz_open3d_320 import export_glb, free_view, fuse, panel, render

ANCHORS = 72
RGB_TARGETS = 24


def diverse_add(candidates, poses, count, seed=()):
    """Greedy spatial+orientation coverage, not chronological truncation."""
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) < count:
        raise ValueError("Not enough distinct candidates")
    chosen = list(dict.fromkeys(seed))
    xyz = np.asarray([poses[i][:3, 3] for i in candidates])
    forward = np.asarray([poses[i][:3, 2] for i in candidates])
    distance = np.full(len(candidates), np.inf)
    def update(index):
        p, direction = poses[index][:3, 3], poses[index][:3, 2]
        d = ((xyz - p)**2).sum(axis=1) + .25 * ((forward - direction)**2).sum(axis=1)
        np.minimum(distance, d, out=distance)
    for index in chosen:
        update(index)
    new = []
    while len(new) < count:
        used = set(chosen) | set(new)
        eligible = np.asarray([i not in used for i in candidates])
        if not eligible.any():
            raise ValueError("Spatial candidate pool exhausted")
        pick = candidates[int(np.argmax(np.where(eligible, distance, -1)))]
        new.append(pick)
        update(pick)
    return sorted(new)


def run(work, output, receipt, gap_selection=False):
    import open3d as o3d
    import trimesh

    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    dataset = o3d.data.LoungeRGBDImages(data_root=str(work))
    params = o3d.io.read_pinhole_camera_trajectory(
        dataset.trajectory_log_path).parameters
    _, selected, poses, pose_indices = select_views(params, dataset.color_paths)
    if len(selected) != 144:
        raise DataError("Expected the existing 144-frame spatial baseline")
    intrinsic = params[pose_indices[0]].intrinsic
    k = intrinsic.intrinsic_matrix
    full = Camera(intrinsic.width, intrinsic.height,
                  float(k[0, 0]), float(k[1, 1]),
                  float(k[0, 2]), float(k[1, 2]))
    if full.width % 2 or full.height % 2:
        raise DataError("Lounge calibration requires even image dimensions")
    camera = full.scaled(2)
    for i in selected:
        intr = params[pose_indices[i]].intrinsic
        if (intr.width != full.width or intr.height != full.height
                or not np.allclose(intr.intrinsic_matrix, k, atol=1e-3)):
            raise DataError("Per-frame camera intrinsics changed")

    source_ids = [selected[0], selected[len(selected)//2], selected[-1]]
    anchors = diverse_add(selected, poses, ANCHORS-3, seed=source_ids)
    anchors = sorted(set(anchors) | set(source_ids))
    if len(anchors) != ANCHORS:
        raise DataError("Expected exactly 72 diverse sensor anchors")
    remaining = sorted(set(selected) - set(anchors))
    targets = diverse_add(remaining, poses, RGB_TARGETS, seed=anchors)
    if set(anchors) & set(targets):
        raise DataError("Sensor and RGB-only indices overlap")
    print(f"Stanford Lounge: 144 upper, {len(anchors)} measured, "
          f"{len(targets)} RGB-only known-pose targets", flush=True)

    measured_by_id, _, _ = read_frames(dataset, anchors, poses, full, camera)
    measured = [measured_by_id[i] for i in anchors]
    receipt.update(
        scene="Stanford Lounge real 640x480 RGB-D",
        source="Open3D LoungeRGBDImages official archive",
        full_rgb_resolution=[full.width, full.height],
        fusion_resolution=[camera.width, camera.height],
        measured_sensor_indices=anchors, rgb_only_target_indices=targets,
        gap_selection_enabled=gap_selection,
        withheld_sensor_indices=sorted(set(selected)-set(anchors)),
        reference_pose="Supplied Stanford Lounge trajectory",
        metric_scale="72 retained measured depth frames, known metric poses",
        source_scan_prior_used=False, cpu_only=True,
        model=MODEL_ID, model_revision=MODEL_REV,
        model_weights_sha256=WEIGHTS_SHA256, model_license="Apache-2.0",
        predicted_depth=[], meshes={}, views=[],
        target_sensor_depth_used_in_reduced_mesh=False,
        historical_fail="#74 full room FAIL; Desk 16.7779cm >15cm FAIL")

    view_poses = [poses[i] for i in source_ids]
    view_poses += [free_view(poses[source_ids[0]], -1.),
                   free_view(poses[source_ids[-1]], 1.)]
    renders = {}
    sparse, _ = fuse(o3d, measured, camera, pixel_stride=1, voxel=VOXEL_M)
    sparse_path = output / "lounge-72-measured-only.glb"
    export_glb(sparse, sparse_path)
    renders["72"] = [render(o3d, sparse, camera, pose) for pose in view_poses]
    receipt["meshes"]["72"] = {"file": sparse_path.name, "sensor_frames":72,
                               "predicted_frames":0}

    gap_targets = []
    if gap_selection:
        # Broaden the candidate pool from the 144 depth-study frames to the
        # full real 2999-image trajectory. ONLY RGB poses and the 72-depth
        # Mesh are consulted here; target depth and supplied scan stay unread.
        excluded = set(anchors) | set(targets)
        candidate_ids = [
            i for i in range(0, len(dataset.color_paths), 7)
            if i not in excluded and np.isfinite(poses[i]).all()
            and abs(np.linalg.det(poses[i][:3, :3]) - 1.) < .05
            and np.linalg.norm(poses[i][:3, 3]) < 100.
        ]
        scores = score_gap_views(o3d, sparse, camera, poses, candidate_ids)
        gap_targets = choose_gap_targets(scores, poses, RGB_TARGETS)
        receipt["gap_rgb_only_target_indices"] = gap_targets
        receipt["gap_candidate_count"] = len(candidate_ids)
        receipt["gap_supported_candidate_count"] = len(scores)
        receipt["gap_selected_scores"] = [
            row for row in scores if row["index"] in set(gap_targets)]
        receipt["withheld_sensor_indices"] = sorted(
            (set(selected) | set(gap_targets)) - set(anchors))
        print(f"Selected {len(gap_targets)} gap-targeted RGB-only cameras "
              f"from {len(scores)} measured-visible candidates", flush=True)
    del sparse
    gc.collect()

    predictor = CPUDepthAnything()
    predictions, inferred_by_id, guarded_by_id = {}, {}, {}
    for i in sorted(set(targets) | set(gap_targets)):
        with Image.open(dataset.color_paths[i]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (full.height, full.width, 3):
            raise DataError(f"Native RGB dimensions invalid at frame {i}")
        relative = np.ascontiguousarray(predictor.infer(rgb)[::2, ::2])
        projected = project_sensor_anchors(measured, camera, poses[i])
        try:
            metric, stats = metric_from_relative(relative, projected)
        except RuntimeError as exc:
            receipt["predicted_depth"].append(
                {"index":i, "used":False, "reason":str(exc)})
            print(f"Skipped RGB-only frame {i}: {exc}", flush=True)
            continue
        inferred_by_id[i] = Frame(
            float(i), poses[i], metric,
            np.ascontiguousarray(rgb[::2, ::2])).validate(camera)
        if gap_selection and i in gap_targets:
            # Do not allow RGB-only predictions to re-fuse already measured
            # surfaces or extrapolate large unsupported planar sheets.
            guarded, guard_stats = conservative_hole_depth(metric, projected)
            guarded_by_id[i] = Frame(
                float(i), poses[i], guarded,
                np.ascontiguousarray(rgb[::2, ::2])).validate(camera)
            receipt.setdefault("guarded_gap_depth", []).append(
                {"index": i, **guard_stats})
        predictions[i] = metric
        Image.fromarray(np.clip(metric*1000, 0, 65535).astype(np.uint16)).save(
            output / f"RGB-estimated-depth-{i:04d}-mm.png")
        receipt["predicted_depth"].append({"index":i, "used":True, **stats})
        print(f"Predicted frame {i}: {stats['predicted_valid_fraction']:.1%} "
              "sensor-supported", flush=True)
    del predictor
    gc.collect()
    inferred = [inferred_by_id[i] for i in targets if i in inferred_by_id]
    gap_inferred = [inferred_by_id[i] for i in gap_targets
                    if i in inferred_by_id]
    if len(inferred) < 4:
        raise RuntimeError(f"Only {len(inferred)} of 24 baseline RGB views calibrated")
    if gap_selection and len(gap_inferred) < 4:
        raise RuntimeError(f"Only {len(gap_inferred)} of 24 gap RGB views calibrated")
    guarded_gap_inferred = [
        guarded_by_id[i] for i in gap_targets if i in guarded_by_id]

    hybrid, _ = fuse(o3d, measured + inferred, camera,
                     pixel_stride=1, voxel=VOXEL_M)
    hybrid_path = output / "lounge-72-measured-plus-24-rgb.glb"
    export_glb(hybrid, hybrid_path)
    renders["hybrid"] = [render(o3d, hybrid, camera, pose)
                         for pose in view_poses]
    receipt["meshes"]["hybrid"] = {
        "file":hybrid_path.name, "sensor_frames":72,
        "predicted_frames":len(inferred)}
    del hybrid
    gc.collect()
    if gap_selection:
        gap_mesh, _ = fuse(o3d, measured + gap_inferred, camera,
                           pixel_stride=1, voxel=VOXEL_M)
        gap_path = output / "lounge-72-measured-plus-24-gap-rgb.glb"
        export_glb(gap_mesh, gap_path)
        renders["gap"] = [render(o3d, gap_mesh, camera, pose)
                          for pose in view_poses]
        receipt["meshes"]["gap"] = {
            "file": gap_path.name, "sensor_frames":72,
            "predicted_frames":len(gap_inferred)}
        del gap_mesh
        gc.collect()

        guarded_mesh, _ = fuse(
            o3d, measured + guarded_gap_inferred, camera,
            pixel_stride=1, voxel=VOXEL_M)
        guarded_path = output / "lounge-72-plus-24-measured-supported-gap.glb"
        export_glb(guarded_mesh, guarded_path)
        renders["guarded_gap"] = [
            render(o3d, guarded_mesh, camera, pose) for pose in view_poses]
        receipt["meshes"]["guarded_gap"] = {
            "file": guarded_path.name, "sensor_frames":72,
            "predicted_frames":len(guarded_gap_inferred),
            "predicted_geometry_policy":
                "only unmeasured rays near consistent retained sensor depths"}
        del guarded_mesh
        gc.collect()

    # Only NOW may any withheld target depth be read, for the 144-depth
    # independent upper control and post-hoc depth error.
    all_by_id, _, _ = read_frames(dataset, selected, poses, full, camera)
    upper, _ = fuse(o3d, [all_by_id[i] for i in selected],
                    camera, pixel_stride=1, voxel=VOXEL_M)
    upper_path = output / "lounge-144-measured-upper.glb"
    export_glb(upper, upper_path)
    renders["144"] = [render(o3d, upper, camera, pose) for pose in view_poses]
    receipt["meshes"]["144"] = {"file":upper_path.name, "sensor_frames":144,
                                "predicted_frames":0, "upper_control_only":True}
    del upper
    gc.collect()
    receipt["posthoc_target_checks"] = []
    # The gap-selected RGB frames can be outside the 144-depth upper set.
    # Their real depth is opened only AFTER BOTH reduced-input GLBs exist.
    gap_outside_upper = sorted(set(predictions) - set(all_by_id))
    if gap_outside_upper:
        heldout_by_id, _, _ = read_frames(
            dataset, gap_outside_upper, poses, full, camera)
        all_by_id.update(heldout_by_id)
    for i, predicted in predictions.items():
        actual = all_by_id[i].depth
        valid = ((actual > DEPTH_MIN) & (actual < DEPTH_MAX) & (predicted > 0))
        if valid.any():
            receipt["posthoc_target_checks"].append({
                "index":i, "pixels":int(valid.sum()),
                "median_absolute_error_m":
                    float(np.median(np.abs(actual[valid]-predicted[valid])))})
    del all_by_id
    gc.collect()

    for spec in receipt["meshes"].values():
        path = output / spec["file"]
        loaded = trimesh.load(str(path), force="mesh", process=False)
        if len(loaded.faces) < 1000 or loaded.visual.kind != "vertex":
            raise DataError(f"Not a valid colored real GLB: {path}")
        spec["triangles"] = int(len(loaded.faces))
        spec["sha256"] = file_sha256(path)

    boards = []
    labels = ["REAL RGB / FREE", "144 SENSOR UPPER",
              "72 SENSOR", f"72 SENSOR + {len(inferred)} RGB"]
    if gap_selection:
        labels.append(f"72 SENSOR + {len(gap_inferred)} GAP RGB")
        labels.append("72 + 24 SUPPORTED GAPS")
    for row, pose in enumerate(view_poses):
        if row < 3:
            i = source_ids[row]
            with Image.open(dataset.color_paths[i]) as image:
                rgb = image.convert("RGB")
            rgb.save(output / f"REAL-lounge-{i:04d}-native.png")
            photo = np.asarray(rgb.resize((camera.width, camera.height),
                                          Image.Resampling.LANCZOS))
            kind = "recorded"
        else:
            i = source_ids[0] if row == 3 else source_ids[-1]
            photo = np.full((camera.height, camera.width, 3), 20, np.uint8)
            kind = "free"
        name = f"{kind}-{i:04d}-comparison.png"
        images = [photo, renders["144"][row],
                  renders["72"][row], renders["hybrid"][row]]
        if gap_selection:
            images.append(renders["gap"][row])
            images.append(renders["guarded_gap"][row])
        boards.append(panel(output / name, images, labels))
        receipt["views"].append({"kind":kind, "index":i, "file":name})
    board = Image.new("RGB", (camera.width*len(labels), (camera.height+27)*5))
    for row, image in enumerate(boards):
        board.paste(image, (0, row*(camera.height+27)))
    board.save(output / ("lounge-72-gap-rgb24-five-views.png" if gap_selection
                         else "lounge-144-vs-72-vs-72plus24rgb-five-views.png"))
    receipt["status"] = "real_meshes_and_five_views_review_pending"
    receipt["visual_verdict"] = "PENDING real five-view inspection; no room PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap-selection", action="store_true",
                        help="Compare original pose-spread RGB24 with 24 measured-gap RGB views")
    args = parser.parse_args()
    receipt = {
        "status":"running",
        "run_url":(f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                   f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None)}
    try:
        run(args.work, args.output, receipt, gap_selection=args.gap_selection)
        code = 0
    except Exception as exc:
        traceback.print_exc()
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status":receipt["status"],
                          "meshes":receipt.get("meshes"),
                          "error":receipt.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
