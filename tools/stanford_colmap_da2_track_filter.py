"""Real Stanford Lounge: reject Depth Anything surfaces contradicted by RGB-SfM tracks.

Reuses the *already inferred* two DA V2 metricized depth PNGs from #157.
Inputs stay 300 RGB / 287 RGB-COLMAP poses / 8 measured depth maps
(including four scale anchors) / 2 predicted depth maps. No prior Mesh.
The two target sensor-depth files are opened ONLY AFTER candidate fusion,
for held-out evaluation, never for filtering or metric calibration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from stanford_colmap_metric300 import original_pairs, read_model, depth_scale
from stanford_colmap_nested_depth import render_rays
from stanford_colmap_pose_cover10 import (
    UPSTREAM_RUN, sha256, select_pose_coverage, five_rays,
)
from stanford_colmap_stereo6 import CAMERA, camera_to_world, fuse_hybrid
from stanford_colmap_eight_measured_control import eight_only_names, old_mesh_to_open3d

DA2_RUN = "https://github.com/REDrighthand05/echo4d/actions/runs/41011113006"
CONTROL_RUN = "https://github.com/REDrighthand05/echo4d/actions/runs/41000309095"


def filter_with_rgb_tracks(depth, observations, w2c, scale):
    """Reject locally contradicted predicted surfaces using RGB-only SfM tracks.

    A measured depth map, held-out sensor depth, old GLB and GT pose are
    neither accepted nor consulted by this function. Conservative: only
    rejects a pixel near a strongly contradictory triangulated track when
    no closer geometrically agreeing track protects it.
    """
    if depth.ndim != 2 or depth.dtype != np.float32:
        raise ValueError("Expected HxW float32 metricized predicted depth")
    h, w = depth.shape
    bad = np.full((h, w), 255, np.uint8)
    good = np.full((h, w), 255, np.uint8)
    good_count = bad_count = 0
    for x, y, xyz in observations:
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        z = float((w2c[:3, :3] @ xyz + w2c[:3, 3])[2] * scale)
        predicted = float(depth[iy, ix])
        if not (0.55 < z < 2.9 and 0.55 < predicted < 2.9):
            continue
        error = abs(z - predicted)
        if error <= max(0.12, 0.10 * z):
            good[iy, ix] = 0
            good_count += 1
        elif error >= max(0.24, 0.20 * z):
            bad[iy, ix] = 0
            bad_count += 1

    if good_count < 20:
        raise RuntimeError(f"Insufficient agreeing RGB tracks: {good_count}")
    if bad_count == 0:
        return depth.copy(), dict(agreeing_tracks=good_count,
                                  contradicting_tracks=0, rejected_pixels=0,
                                  retained_pixels=int(np.count_nonzero(depth)))

    d_bad = cv2.distanceTransform(bad, cv2.DIST_L2, 3)
    d_good = cv2.distanceTransform(good, cv2.DIST_L2, 3)
    reject = ((depth > 0) & (d_bad <= 26.0) &
              (d_bad * 1.10 < d_good))
    filtered = depth.copy()
    filtered[reject] = 0
    return filtered, dict(
        agreeing_tracks=good_count, contradicting_tracks=bad_count,
        rejected_pixels=int(np.count_nonzero(reject)),
        retained_pixels=int(np.count_nonzero(filtered)),
        original_pixels=int(np.count_nonzero(depth)),
        filter="RGB-SfM projected track depth disagreement; 26px local support")


def evaluate_heldout(original, filtered, sensor_mm):
    """Held-out sensor depth is read only after both GLBs exist."""
    actual = np.asarray(sensor_mm, dtype=np.float32) / 1000.0
    if actual.shape != original.shape:
        raise ValueError("Held-out depth shape differs from predicted RGB")
    sensor_valid = np.isfinite(actual) & (actual > 0.55) & (actual < 2.9)
    result = {}
    for label, predicted in (("old_da2", original), ("filtered_da2", filtered)):
        valid = sensor_valid & np.isfinite(predicted) & (predicted > 0)
        count = int(np.count_nonzero(valid))
        errors = np.abs(predicted[valid] - actual[valid])
        result[label] = dict(
            supported_pixels=count,
            sensor_pixels=int(np.count_nonzero(sensor_valid)),
            median_absolute_error_m=float(np.median(errors)) if count else None,
            within_15cm_fraction=float(np.mean(errors < 0.15)) if count else None)
    return result


def self_test():
    h, w = 100, 140
    depth = np.full((h, w), 1.5, np.float32)
    observations = []
    for y in range(10, 90, 10):
        for x in range(10, 130, 10):
            # Left-hand patch contradicts prediction; right-hand patch agrees.
            z = 2.1 if x < 55 else 1.5
            observations.append((x, y, np.array([0., 0., z])))
    filtered, info = filter_with_rgb_tracks(
        depth, observations, np.eye(4), 1.0)
    assert info["rejected_pixels"] > 100, info
    assert np.mean(filtered[20:80, 15:45] == 0) > 0.5, info
    assert np.all(filtered[20:80, 90:125] > 0), info
    sensor = np.full((h, w), 1500, np.uint16)
    evaluation = evaluate_heldout(depth, filtered, sensor)
    assert evaluation["old_da2"]["median_absolute_error_m"] == 0.0
    assert evaluation["filtered_da2"]["supported_pixels"] < h*w
    print("PASS: RGB-SfM contradictory-patch removal and held-out-only evaluation")


def run(upstream, control, da2, output):
    import open3d as o3d
    import trimesh

    upstream, control, da2, output = (
        Path(p).resolve() for p in (upstream, control, da2, output))
    if output.exists():
        raise FileExistsError("Refuse to overwrite an existing real experiment")
    output.mkdir(parents=True)
    started = time.time()
    receipt = dict(
        status="RUNNING", scene="Stanford Lounge",
        upstream=UPSTREAM_RUN, da2_source=DA2_RUN, control_source=CONTROL_RUN,
        input_rgb=300, registered_rgb=287, measured_depth_maps=8,
        inferred_depth_maps=2, prior_mesh=False, ground_truth_poses=False,
        gpu=False, prediction="cached CPU Depth Anything V2 Small from #157",
        metric_scale_source="four original measured depth anchors",
        filter_source="RGB-only COLMAP tracks (no sensor-depth filtering)",
        heldout_depth_used_for="post-fusion evaluation ONLY",
        complete_room_acceptance="FAIL", visual_acceptance="PENDING")
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
            raise ValueError("Frozen 300 RGB / 287 pose source changed")
        if len(ledger) != 300 or baseline.get("measured_depth_maps") != 8:
            raise ValueError("Frozen original RGB or 8-measured control changed")
        if (old_receipt.get("measured_depth_maps"),
                old_receipt.get("inferred_depth_maps"),
                old_receipt.get("prior_mesh")) != (8, 2, False):
            raise ValueError("Unexpected cached Depth Anything experiment")
        old_glb = da2 / "measured8_da2.glb"
        if sha256(old_glb) != old_receipt["da2_glb_sha256"]:
            raise ValueError("Original Depth Anything GLB SHA mismatch")
        if sha256(control / "measured8_only.glb") != baseline["measured8_only_sha256"]:
            raise ValueError("Eight-measured control GLB SHA mismatch")

        views = read_model(upstream / parent["model"])
        if len(views) != 287:
            raise ValueError("COLMAP registration count changed")
        lookup = {int(Path(name).stem): name for name, _, _ in views}
        base30 = [lookup[int(i)] for i in parent["measured_depth_rgb_samples"]]
        base15 = base30[::2]
        base10 = [base15[int(i)] for i in np.linspace(0, 14, 10, dtype=int)]
        shared = [base10[i] for i in (0, 3, 6, 9)]
        source = original_pairs(o3d.data.LoungeRGBDImages())
        rgb_paths, depth_paths = {}, {}
        for row in ledger:
            i = int(row["sample"])
            rgb, sensor = source[int(row["original_index"])]
            if sha256(rgb) != row["rgb_sha256"]:
                raise ValueError(f"Original RGB checksum changed: {i}")
            name = f"{i:06d}{rgb.suffix.lower()}"
            rgb_paths[name], depth_paths[name] = rgb, sensor
        for name in shared:
            i = int(Path(name).stem)
            if sha256(depth_paths[name]) != ledger[i]["depth_sha256"]:
                raise ValueError(f"Metric anchor changed: {i}")
        scale, _ = depth_scale(
            [v for v in views if v[0] in set(shared)],
            {n: depth_paths[n] for n in shared})
        if not np.isclose(scale, baseline["metric_scale_m_per_colmap_unit"],
                          rtol=1e-7, atol=1e-10):
            raise ValueError("Frozen metric scale changed")
        ten = select_pose_coverage(views, shared, 10, scale)
        eight = eight_only_names(ten, baseline["stereo_replacement_samples"])
        if sorted(int(Path(n).stem) for n in eight) != baseline["measured_depth_samples"]:
            raise ValueError("Frozen eight measured samples changed")
        if sorted(int(Path(n).stem) for n in eight) != old_receipt["measured_depth_samples"]:
            raise ValueError("DA2 and eight-only measured inputs differ")
        targets = [f"{int(i):06d}.png" for i in baseline["stereo_replacement_samples"]]
        if sorted(targets) != sorted(
                f"{int(i):06d}.png" for i in old_receipt["inferred_depth_samples"]):
            raise ValueError("DA2 predicted targets changed")
        if set(targets) & set(eight):
            raise ValueError("Target held-out sensor frame entered measured input")

        poses = {name: camera_to_world(w2c, scale) for name, w2c, _ in views}
        view_by_name = {name: (w2c, matches) for name, w2c, matches in views}
        predicted = {}
        filtered = {}
        inferred = {}
        for name in targets:
            path = da2 / f"inferred-{Path(name).stem}-mm.png"
            with Image.open(path) as im:
                mm = np.asarray(im, dtype=np.uint16)
            depth = np.ascontiguousarray(mm.astype(np.float32) / 1000.)
            w2c, tracks = view_by_name[name]
            cleaned, info = filter_with_rgb_tracks(depth, tracks, w2c, scale)
            if not np.any(cleaned > 0):
                raise RuntimeError(f"All inferred pixels rejected for {name}")
            predicted[name], filtered[name] = depth, cleaned
            with Image.open(rgb_paths[name]) as im:
                color = np.asarray(im.convert("RGB"), dtype=np.uint8)
            inferred[name] = (color, cleaned, CAMERA, poses[name])
            Image.fromarray(np.clip(cleaned * 1000, 0, 65535).astype(np.uint16)).save(
                output / f"filtered-{Path(name).stem}-mm.png")
            receipt.setdefault("filter", []).append(dict(target=name, **info))
            save()

        # Geometry is produced BEFORE any held-out target sensor depth is opened.
        new_glb = output / "measured8_da2_track_filtered.glb"
        new_mesh = fuse_hybrid(
            o3d, trimesh, views, eight, inferred, scale,
            rgb_paths, depth_paths, new_glb)
        receipt["new_glb_sha256"] = sha256(new_glb)
        receipt["heldout_evaluation"] = {}
        for name in targets:
            # Evaluation only: this target sensor frame was absent from fusion,
            # metric calibration, track filtering and all other geometry steps.
            with Image.open(depth_paths[name]) as im:
                sensor_mm = im.copy()
            receipt["heldout_evaluation"][name] = evaluate_heldout(
                predicted[name], filtered[name], sensor_mm)
        old_mesh = old_mesh_to_open3d(o3d, trimesh, old_glb)
        recorded, rays = five_rays(o3d, views, scale)
        panel = Image.new("RGB", (960, 5*270+30), (18, 18, 18))
        draw = ImageDraw.Draw(panel)
        for col, title in enumerate(
                ("Original RGB", "8 measured + 2 DA2", "Same inputs + RGB-track filter")):
            draw.text((col*320+5, 8), title, fill="white")
        hits = {"old_da2": [], "track_filtered": []}
        for i, ray in enumerate(rays):
            y = 30 + i*270
            draw.text((5, y+3),
                      f"recorded {Path(recorded[i][0]).stem}" if i < 3
                      else f"free orbit {i-2}", fill="white")
            if i < 3:
                with Image.open(rgb_paths[recorded[i][0]]) as im:
                    panel.paste(im.convert("RGB").resize((320, 240)), (0, y+26))
            for col, (key, mesh) in enumerate(
                    (("old_da2", old_mesh), ("track_filtered", new_mesh)), 1):
                rendered, hit = render_rays(o3d, mesh, ray)
                panel.paste(rendered, (col*320, y+26))
                hits[key].append(hit)
        image_name = "source-da2-vs-track-filtered-3plus2.png"
        panel.save(output / image_name)
        receipt.update(
            status="REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED",
            image=image_name, old_da2_glb_sha256=sha256(old_glb),
            same_ray_hit_percent=hits,
            measured_depth_samples=baseline["measured_depth_samples"],
            inferred_depth_samples=old_receipt["inferred_depth_samples"],
            visual_acceptance="PENDING five-view human inspection; full room FAIL")
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
