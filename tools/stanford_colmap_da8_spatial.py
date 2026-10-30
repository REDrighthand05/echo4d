"""Stanford Lounge: keep 8 sensor depths; expand 2 to 8 CPU DA2 RGB predictions.

The old 8+2 GLB is comparison ONLY, never used to construct the new mesh.
All predicted views use RGB-only COLMAP tracks for per-image metricization.
Four measured anchors set global SfM scale. Target sensor depths are read
only AFTER mesh export for held-out evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stanford_colmap_metric300 import original_pairs, read_model, depth_scale
from stanford_colmap_nested_depth import render_rays
from stanford_colmap_pose_cover10 import (
    UPSTREAM_RUN, sha256, select_pose_coverage, five_rays,
)
from stanford_colmap_stereo6 import CAMERA, camera_to_world, fuse_hybrid
from stanford_colmap_eight_measured_control import eight_only_names, old_mesh_to_open3d
from stanford_colmap_depthanything2 import (
    DepthAnythingCPU, calibrate_sparse_metric_depth, MODEL_ID,
    MODEL_REVISION, WEIGHTS_SHA256,
)

CONTROL_RUN = "https://github.com/REDrighthand05/echo4d/actions/runs/41000309095"
DA2_RUN = "https://github.com/REDrighthand05/echo4d/actions/runs/41011113006"
EXTRA_PREDICTIONS = 6


def eligible_views(views, fixed, scale):
    """Require sparse RGB geometry for metric fitting, never sensor depth."""
    result = []
    for name, w2c, matches in views:
        supported = 0
        for x, y, xyz in matches:
            z = float((w2c[:3, :3] @ xyz + w2c[:3, 3])[2] * scale)
            if 0.55 < z < 2.9 and 0 <= x < 640 and 0 <= y < 480:
                supported += 1
        if name in fixed or supported >= 45:
            result.append((name, w2c, matches))
    return result


def novelty(name, selected, poses):
    """Prioritize room coverage among the pose-frustum-selected candidates."""
    pose = poses[name]
    forward = pose[:3, 2]
    center = pose[:3, 3]
    distances = []
    for other in selected:
        p = poses[other]
        separation = float(np.linalg.norm(center - p[:3, 3]))
        angle = 1.0 - float(np.clip(forward @ p[:3, 2], -1, 1))
        distances.append(separation + 0.35 * angle)
    return min(distances)


def heldout_check(predicted, sensor_path):
    """Evaluation only, called after new GLB is written."""
    with Image.open(sensor_path) as im:
        actual = np.asarray(im, dtype=np.float32) / 1000.0
    if actual.shape != predicted.shape:
        raise ValueError("Held-out sensor depth shape mismatch")
    valid = ((predicted > 0) & np.isfinite(predicted) &
             (actual > 0.55) & (actual < 2.9))
    if not np.any(valid):
        return dict(sensor_overlap_pixels=0, within_15cm=None)
    errors = np.abs(predicted[valid] - actual[valid])
    return dict(sensor_overlap_pixels=int(valid.sum()),
                within_15cm=float(np.mean(errors < 0.15)))


def self_test():
    poses = {}
    for i in range(4):
        p = np.eye(4)
        p[0, 3] = float(i)
        poses[f"{i:06d}.png"] = p
    assert novelty("000003.png", ["000000.png"], poses) > novelty(
        "000001.png", ["000000.png"], poses)
    assert novelty("000002.png", ["000000.png", "000003.png"], poses) > 0
    print("PASS: pose novelty prioritizes unobserved space")


def run(upstream, control, da2, output):
    import open3d as o3d
    import trimesh

    upstream, control, da2, output = (
        Path(p).resolve() for p in (upstream, control, da2, output))
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing experiment")
    output.mkdir(parents=True)
    started = time.time()
    receipt = dict(
        status="RUNNING", scene="Stanford Lounge", source_run=UPSTREAM_RUN,
        control_run=CONTROL_RUN, da2_run=DA2_RUN,
        input_rgb=300, registered_rgb=287, measured_depth_maps=8,
        target_inferred_depth_maps=8, prior_mesh=False,
        ground_truth_poses=False, gpu=False,
        predicted_depth="CPU Depth Anything V2 Small, per-view RGB-SfM calibration",
        model_id=MODEL_ID, model_revision=MODEL_REVISION,
        model_weights_sha256=WEIGHTS_SHA256,
        metric_scale_source="four shared real measured depth anchors",
        predicted_target_sensor_depth_used_for="post-fusion evaluation ONLY",
        visual_acceptance="PENDING", full_room_acceptance="FAIL",
        prediction_failures=[], inferred=[],
    )

    def save():
        (output / "receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        parent = json.loads((upstream / "receipt.json").read_text())
        ledger = json.loads((upstream / "inputs.json").read_text())
        baseline = json.loads((control / "receipt.json").read_text())
        old_receipt = json.loads((da2 / "receipt.json").read_text())
        if (parent.get("selected_rgb"), parent.get("registered_frames"),
                parent.get("rgb_sampling")) != (300, 287, "early-overlap"):
            raise ValueError("Frozen independent RGB-COLMAP source changed")
        if len(ledger) != 300 or baseline.get("measured_depth_maps") != 8:
            raise ValueError("Frozen 300 RGB / eight sensor depth source changed")
        if (old_receipt.get("measured_depth_maps"),
                old_receipt.get("inferred_depth_maps"),
                old_receipt.get("prior_mesh")) != (8, 2, False):
            raise ValueError("Frozen DA2 comparison source changed")
        old_glb = da2 / "measured8_da2.glb"
        if sha256(old_glb) != old_receipt["da2_glb_sha256"]:
            raise ValueError("Old comparison GLB hash changed")
        views = read_model(upstream / parent["model"])
        if len(views) != 287:
            raise ValueError("Expected 287 independently registered RGB cameras")
        lookup = {int(Path(name).stem): name for name, _, _ in views}
        base30 = [lookup[int(i)] for i in parent["measured_depth_rgb_samples"]]
        base15 = base30[::2]
        base10 = [base15[int(i)] for i in np.linspace(0, 14, 10, dtype=int)]
        shared = [base10[i] for i in (0, 3, 6, 9)]
        source = original_pairs(o3d.data.LoungeRGBDImages())
        rgb_paths, depth_paths = {}, {}
        for row in ledger:
            sample = int(row["sample"])
            color, depth = source[int(row["original_index"])]
            if sha256(color) != row["rgb_sha256"]:
                raise ValueError(f"Original RGB hash changed: {sample}")
            name = f"{sample:06d}{color.suffix.lower()}"
            rgb_paths[name], depth_paths[name] = color, depth
        for name in shared:
            sample = int(Path(name).stem)
            if sha256(depth_paths[name]) != ledger[sample]["depth_sha256"]:
                raise ValueError(f"Real metric anchor changed: {sample}")
        scale, witnesses = depth_scale(
            [v for v in views if v[0] in set(shared)],
            {name: depth_paths[name] for name in shared})
        if not np.isclose(scale, baseline["metric_scale_m_per_colmap_unit"],
                          rtol=1e-7, atol=1e-10):
            raise ValueError("Frozen metric scale changed")
        ten = select_pose_coverage(views, shared, 10, scale)
        eight = eight_only_names(ten, baseline["stereo_replacement_samples"])
        if sorted(int(Path(n).stem) for n in eight) != baseline["measured_depth_samples"]:
            raise ValueError("Eight measured frames changed")
        if sorted(int(Path(n).stem) for n in eight) != old_receipt["measured_depth_samples"]:
            raise ValueError("Eight sensor inputs differ from DA2 baseline")
        for name in eight:
            sample = int(Path(name).stem)
            if sha256(depth_paths[name]) != ledger[sample]["depth_sha256"]:
                raise ValueError(f"Measured input hash changed: {sample}")
        old_targets = [f"{int(i):06d}.png" for i in old_receipt["inferred_depth_samples"]]
        if set(old_targets) & set(eight):
            raise ValueError("A predicted target is already a measured frame")
        fixed = sorted(set(eight) | set(old_targets))
        poses = {name: camera_to_world(w2c, scale) for name, w2c, _ in views}
        view_by_name = {name: (w2c, matches) for name, w2c, matches in views}
        eligible = eligible_views(views, set(fixed), scale)
        if len(eligible) < len(fixed) + EXTRA_PREDICTIONS:
            raise RuntimeError("Not enough RGB cameras with sparse 3D tracks")
        shortlist = select_pose_coverage(
            eligible, fixed, min(len(eligible), len(fixed) + 20), scale)
        candidates = sorted(set(shortlist) - set(fixed))
        receipt.update(
            metric_scale_m_per_colmap_unit=scale,
            metric_scale_witnesses=witnesses,
            measured_depth_samples=baseline["measured_depth_samples"],
            old_predicted_samples=[int(Path(n).stem) for n in old_targets],
            candidate_samples=[int(Path(n).stem) for n in candidates],
            candidate_policy="RGB-COLMAP frustum coverage then camera novelty",
            candidate_sparse_track_minimum=45)
        save()

        cache = {}
        def rgb(name):
            if name not in cache:
                with Image.open(rgb_paths[name]) as im:
                    cache[name] = np.asarray(im.convert("RGB"), dtype=np.uint8)
            return cache[name]

        predictor = DepthAnythingCPU()  # pinned, SHA-verified, CPU-only
        inferred, predictions = {}, {}
        for name in old_targets:
            w2c, matches = view_by_name[name]
            depth, info = calibrate_sparse_metric_depth(
                predictor.infer(rgb(name)), matches, w2c, scale)
            if info["valid_fraction"] < 0.002:
                raise RuntimeError(f"Frozen DA2 target lost sparse support: {name}")
            inferred[name] = (rgb(name), depth, CAMERA, poses[name])
            predictions[name] = depth
            Image.fromarray(np.clip(depth * 1000, 0, 65535).astype(np.uint16)).save(
                output / f"inferred-{Path(name).stem}-mm.png")
            receipt["inferred"].append(dict(target=name, reused_old_target=True, **info))
            save()

        # The same 8 measured frames remain. Only 6 additional, spatially
        # separated RGB frames may enter the new TSDF as learned depth.
        chosen = list(fixed)
        successes = 0
        while candidates and successes < EXTRA_PREDICTIONS:
            name = max(candidates, key=lambda n: (novelty(n, chosen, poses), n))
            candidates.remove(name)
            w2c, matches = view_by_name[name]
            try:
                depth, info = calibrate_sparse_metric_depth(
                    predictor.infer(rgb(name)), matches, w2c, scale)
                if info["valid_fraction"] < 0.002:
                    raise RuntimeError("Predicted depth lacks RGB sparse support")
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                receipt["prediction_failures"].append(
                    dict(target=name, error=f"{type(exc).__name__}: {exc}"))
                save()
                continue
            inferred[name] = (rgb(name), depth, CAMERA, poses[name])
            predictions[name] = depth
            Image.fromarray(np.clip(depth * 1000, 0, 65535).astype(np.uint16)).save(
                output / f"inferred-{Path(name).stem}-mm.png")
            receipt["inferred"].append(dict(target=name, reused_old_target=False, **info))
            chosen.append(name)
            successes += 1
            save()
        if successes != EXTRA_PREDICTIONS:
            raise RuntimeError(
                f"Only {successes}/{EXTRA_PREDICTIONS} additional CPU predictions calibrated")

        mesh = fuse_hybrid(o3d, trimesh, views, eight, inferred, scale,
                           rgb_paths, depth_paths, output / "measured8_da8.glb")
        receipt["new_glb_sha256"] = sha256(output / "measured8_da8.glb")
        receipt["new_predicted_samples"] = [
            int(Path(x["target"]).stem) for x in receipt["inferred"]
            if not x["reused_old_target"]]
        save()

        # Strictly post-fusion: held-out sensor depths were never accessed
        # for candidate choice, metricization, filtering or mesh generation.
        receipt["heldout_evaluation"] = {
            name: heldout_check(depth, depth_paths[name])
            for name, depth in predictions.items()}
        old = old_mesh_to_open3d(o3d, trimesh, old_glb)
        recorded, rays = five_rays(o3d, views, scale)
        panel = Image.new("RGB", (960, 5 * 270 + 30), (18, 18, 18))
        draw = ImageDraw.Draw(panel)
        for col, title in enumerate(
                ("Original RGB", "8 measured + 2 DA2", "8 measured + 8 DA2")):
            draw.text((col * 320 + 5, 8), title, fill="white")
        hits = {"old_da2": [], "new_da8": []}
        for i, ray in enumerate(rays):
            y = 30 + i * 270
            draw.text((5, y + 3),
                      f"recorded {Path(recorded[i][0]).stem}" if i < 3
                      else f"free orbit {i - 2}", fill="white")
            if i < 3:
                with Image.open(rgb_paths[recorded[i][0]]) as im:
                    native = im.convert("RGB")
                    native.save(output / f"source-native-{Path(recorded[i][0]).stem}.png")
                    panel.paste(native.resize((320, 240)), (0, y + 26))
            for col, (label, candidate) in enumerate(
                    (("old_da2", old), ("new_da8", mesh)), 1):
                image, hit = render_rays(o3d, candidate, ray)
                panel.paste(image, (col * 320, y + 26))
                hits[label].append(hit)
        image_name = "source-da2-vs-da8-3plus2.png"
        panel.save(output / image_name)
        receipt.update(
            status="REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED",
            image=image_name, inferred_depth_maps=len(inferred),
            old_da2_glb_sha256=sha256(old_glb),
            same_ray_hit_percent=hits,
            visual_acceptance="PENDING actual five-view human review; full room FAIL")
    except Exception as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = round(time.time() - started, 2)
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--da2", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not all((args.upstream, args.control, args.da2, args.output)):
            parser.error("--upstream --control --da2 --output required")
        run(args.upstream, args.control, args.da2, args.output)
