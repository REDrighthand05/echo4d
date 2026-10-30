"""Replace six of ten measured Stanford Lounge depths with CPU RGB stereo.

Keeps the exact 300 RGBs, 287 independently COLMAP-estimated poses, and four
real metric depth anchors from #147. Baseline = ten measured maps; hybrid =
four measured maps plus six *inferred* stereo depths, no prior Mesh. Inferred
depth is never described as sensor truth. RGB-only single video is NOT a
prerequisite for this ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from echo4d.model import Camera
from echo4d.known_pose_stereo import infer_stereo
from stanford_colmap_metric300 import original_pairs, read_model, depth_scale
from stanford_colmap_nested_depth import render_rays
from stanford_colmap_pose_cover10 import (
    UPSTREAM_RUN, sha256, select_pose_coverage, fuse, five_rays,
    metric_extrinsic,
)

CAMERA = Camera(640, 480, 525.0, 525.0, 319.5, 239.5)
BRANCH_EXPERIMENT = "300 original RGB / 287 RGB-COLMAP poses / 4 measured + 6 stereo-inferred depth"


def camera_to_world(w2c, scale):
    return np.linalg.inv(metric_extrinsic(w2c, scale))


def candidate_pairs(target, views, scale, limit=14):
    """Allow either lateral direction; the rectified LEFT view owns inferred Z.

    A moving camera often has no later frame to its right. For a neighbour on
    its left, swap the stereo order and integrate predicted depth at the
    neighbour, rather than rejecting valid RGB parallax. No measured depth
    participates in pairing.
    """
    poses = {name: camera_to_world(w2c, scale) for name, w2c, _ in views}
    anchor = poses[target]
    candidates = []
    for name, other in poses.items():
        if name == target:
            continue
        delta = anchor[:3, :3].T @ (other[:3, 3] - anchor[:3, 3])
        baseline = float(np.linalg.norm(delta))
        if not (0.040 <= baseline <= 0.29 and abs(delta[0]) > 0.035):
            continue
        if abs(delta[1]) > 0.60 * abs(delta[0]) or abs(delta[2]) > 0.65 * abs(delta[0]):
            continue
        rotation = other[:3, :3].T @ anchor[:3, :3]
        angle = math.degrees(math.acos(float(np.clip((np.trace(rotation)-1)/2, -1, 1))))
        if angle > 30:
            continue
        left, right = (target, name) if delta[0] > 0 else (name, target)
        score = abs(baseline-0.12) + 0.45*abs(delta[1]) + 0.15*abs(delta[2]) + angle*0.002
        candidates.append((score, left, right, baseline, angle))
    return sorted(candidates)[:limit]


def fuse_hybrid(o3d, trimesh, views, measured, inferred, scale, colors, depths, dest):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.025, sdf_trunc=0.10,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    original_intr = o3d.camera.PinholeCameraIntrinsic(
        640, 480, 525.0, 525.0, 319.5, 239.5)
    poses = {name: w2c for name, w2c, _ in views}
    for name in measured:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(str(colors[name])),
            o3d.io.read_image(str(depths[name])),
            depth_scale=1000.0, depth_trunc=3.0,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, original_intr, metric_extrinsic(poses[name], scale))
    for name, (rgb, depth, cam, c2w) in inferred.items():
        if depth.dtype != np.float32 or not np.any(depth > 0):
            raise ValueError(f"Invalid RGB-inferred depth for {name}")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb, dtype=np.uint8)),
            o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)),
            depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(
            cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy)
        volume.integrate(rgbd, intr, np.linalg.inv(c2w))
    mesh = volume.extract_triangle_mesh()
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise RuntimeError("Hybrid did not generate a real colored Mesh")
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors8 = np.clip(np.asarray(mesh.vertex_colors)*255, 0, 255).astype(np.uint8)
    trimesh.Trimesh(vertices, triangles, vertex_colors=colors8, process=False).export(dest)
    return mesh


def self_test():
    views = []
    for i in range(31):
        pose = np.eye(4)
        pose[0, 3] = -i*0.012
        views.append((f"{i:06d}.png", pose, []))
    leftmost = candidate_pairs("000000.png", views, 1.0)
    rightmost = candidate_pairs("000030.png", views, 1.0)
    assert leftmost and rightmost
    assert all(left == "000000.png" for _, left, _, _, _ in leftmost)
    assert all(right == "000030.png" for _, _, right, _, _ in rightmost)
    assert candidate_pairs("000030.png", list(reversed(views)), 1.0) == rightmost
    assert all(0.04 <= b <= 0.29 for _, _, _, b, _ in rightmost)
    print("PASS: bidirectional CPU RGB stereo pairing without depth or mesh")


def run(upstream, output):
    import open3d as o3d
    import trimesh

    upstream, output = upstream.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Output must be a fresh directory")
    output.mkdir(parents=True)
    started = time.time()
    receipt = dict(
        status="RUNNING", scene="Stanford Lounge",
        experiment=BRANCH_EXPERIMENT,
        source_run=UPSTREAM_RUN, source_artifact_id=12939354226,
        prior_mesh=False, ground_truth_poses=False, gpu=False,
        pose_source="300 original RGB, independent COLMAP, 287 registered",
        predicted_depth_source="OpenCV CPU SGBM stereo via existing echo4d.known_pose_stereo",
        inferred_depth_is_measured=False,
        visual_acceptance="FAIL until actual 3+2 inspection",
        complete_room_acceptance="FAIL")
    def save():
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    try:
        parent = json.loads((upstream / "receipt.json").read_text())
        ledger = json.loads((upstream / "inputs.json").read_text())
        if (parent["selected_rgb"], parent["registered_frames"],
                parent["rgb_sampling"]) != (300, 287, "early-overlap"):
            raise ValueError("Frozen original RGB-COLMAP source changed")
        if len(ledger) != 300:
            raise ValueError("Frozen 300 RGB ledger missing")
        views = read_model(upstream / parent["model"])
        if len(views) != 287:
            raise ValueError("Expected 287 independently registered RGB cameras")
        lookup = {int(Path(name).stem): name for name, _, _ in views}
        base30 = [lookup[i] for i in parent["measured_depth_rgb_samples"]]
        base15 = base30[::2]
        base10 = [base15[int(i)] for i in np.linspace(0, 14, 10, dtype=int)]
        shared = [base10[i] for i in (0, 3, 6, 9)]
        sources = original_pairs(o3d.data.LoungeRGBDImages())
        rgb_paths, depth_paths = {}, {}
        for row in ledger:
            i = int(row["sample"])
            rgb, dep = sources[int(row["original_index"])]
            if sha256(rgb) != row["rgb_sha256"]:
                raise ValueError(f"Original RGB checksum changed at {i}")
            name = f"{i:06d}{rgb.suffix.lower()}"
            rgb_paths[name], depth_paths[name] = rgb, dep
        for name in shared:
            i = int(Path(name).stem)
            if sha256(depth_paths[name]) != ledger[i]["depth_sha256"]:
                raise ValueError(f"Metric scale anchor changed at {i}")
        scale, witnesses = depth_scale(
            [v for v in views if v[0] in set(shared)],
            {name: depth_paths[name] for name in shared})
        pose_names = select_pose_coverage(views, shared, 10, scale)
        replace = sorted(set(pose_names) - set(shared))
        if len(replace) != 6:
            raise ValueError("Exactly six measured maps must be replaced")
        receipt.update(
            selected_rgb=300, registered_rgb=287,
            baseline_measured_depth_maps=10, hybrid_measured_depth_maps=4,
            requested_inferred_depth_maps=6,
            common_measured_depth_samples=[int(Path(n).stem) for n in shared],
            replaced_measured_depth_samples=[int(Path(n).stem) for n in replace],
            metric_scale_m_per_colmap_unit=scale,
            metric_scale_source="only the four common measured depth maps",
            metric_scale_witnesses=witnesses)
        save()

        # The old ten-depth Mesh is re-fused, never read as a reconstruction prior.
        baseline = fuse(o3d, trimesh, views, pose_names, scale, rgb_paths,
                        depth_paths, output / "measured10.glb")
        inferred, trials = {}, []
        poses = {name: camera_to_world(w2c, scale) for name, w2c, _ in views}
        cache = {}
        def rgb(name):
            if name not in cache:
                with Image.open(rgb_paths[name]) as im:
                    cache[name] = np.asarray(im.convert("RGB"), dtype=np.uint8)
            return cache[name]
        for name in replace:
            best = None
            pairs = candidate_pairs(name, views, scale)
            for _, left, right, baseline_m, angle in pairs:
                try:
                    disparities = 64 if baseline_m < 0.15 else 128
                    rgb_l, dep, cam, pose, info = infer_stereo(
                        rgb(left), rgb(right), CAMERA, poses[left], poses[right],
                        num_disparities=disparities)
                    frac = info["valid_fraction"]
                    trials.append(dict(target=name, left=left, right=right,
                                       baseline_m=baseline_m, angle_deg=angle,
                                       num_disparities=disparities,
                                       valid_fraction=frac))
                    if best is None or frac > best[0]:
                        best = (frac, left, right, rgb_l, dep, cam, pose, info)
                    if frac >= 0.12:
                        break
                except (ValueError, RuntimeError) as exc:
                    trials.append(dict(target=name, left=left, right=right,
                                       error=f"{type(exc).__name__}: {exc}"))
            if best is None or best[0] < 0.003:
                raise RuntimeError(
                    f"No usable bilateral RGB stereo for {name}: "
                    f"{len(pairs)} pairs; best valid fraction "
                    f"{0 if best is None else best[0]:.5f}")
            frac, left, right, rgb_l, dep, cam, pose, info = best
            if left in inferred:
                raise RuntimeError(f"Stereo predicted left camera {left} twice")
            inferred[left] = (rgb_l, dep, cam, pose)
            receipt.setdefault("inferred_frames", []).append(
                dict(replaced_measured_sample=int(Path(name).stem),
                     inferred_left_sample=int(Path(left).stem),
                     paired_right_sample=int(Path(right).stem),
                     **info))
            save()

        hybrid = fuse_hybrid(o3d, trimesh, views, shared, inferred, scale,
                             rgb_paths, depth_paths, output / "measured4_stereo6.glb")
        # Second actual input budget: retain eight measured maps and replace
        # only the two RGB-stereo predictions with most consistent pixels.
        # Selection uses RGB matching quality only, never held-out depth.
        best_two = sorted(receipt["inferred_frames"],
                          key=lambda d: (-d["valid_fraction"],
                                         d["replaced_measured_sample"]))[:2]
        replaced_two = {f"{d['replaced_measured_sample']:06d}.png" for d in best_two}
        measured8 = sorted(set(pose_names) - replaced_two)
        inferred2 = {
            f"{d['inferred_left_sample']:06d}.png":
            inferred[f"{d['inferred_left_sample']:06d}.png"]
            for d in best_two
        }
        if len(measured8) != 8 or len(inferred2) != 2:
            raise RuntimeError("Eight measured plus two RGB-inferred budget violated")
        hybrid2 = fuse_hybrid(o3d, trimesh, views, measured8, inferred2, scale,
                              rgb_paths, depth_paths, output / "measured8_stereo2.glb")
        receipt.update(
            stereo_trials=trials, actual_inferred_depth_maps=len(inferred),
            intermediate_measured_depth_maps=8,
            intermediate_inferred_depth_maps=2,
            intermediate_replaced_measured_samples=sorted(
                int(Path(n).stem) for n in replaced_two),
            measured10_sha256=sha256(output / "measured10.glb"),
            measured4_stereo6_sha256=sha256(output / "measured4_stereo6.glb"),
            measured8_stereo2_sha256=sha256(output / "measured8_stereo2.glb"))
        recorded, rays = five_rays(o3d, views, scale)
        labels = [f"recorded RGB sample {Path(v[0]).stem}" for v in recorded]
        labels += ["free room orbit A", "free room orbit B"]
        receipt["same_ray_hit_percent"] = {}
        for candidate_name, candidate_mesh, column_label, image_name in (
            ("measured4_stereo6", hybrid, "4 measured + 6 RGB stereo",
             "source-measured10-vs-measured4-stereo6-3plus2.png"),
            ("measured8_stereo2", hybrid2, "8 measured + 2 RGB stereo",
             "source-measured10-vs-measured8-stereo2-3plus2.png"),
        ):
            panel = Image.new("RGB", (960, 5*270+30), (18,18,18))
            draw = ImageDraw.Draw(panel)
            for col, label in enumerate(("Original RGB", "10 measured", column_label)):
                draw.text((col*320+6, 8), label, fill="white")
            for i, ray in enumerate(rays):
                y = 30 + i*270
                draw.text((6,y+3), labels[i], fill="white")
                if i < 3:
                    with Image.open(rgb_paths[recorded[i][0]]) as photo:
                        panel.paste(photo.convert("RGB").resize((320,240)), (0,y+26))
                for col, (label, mesh) in enumerate(
                        (("measured10", baseline), (candidate_name, candidate_mesh)), 1):
                    rendered, coverage = render_rays(o3d, mesh, ray)
                    panel.paste(rendered, (col*320, y+26))
                    if label != "measured10" or candidate_name == "measured4_stereo6":
                        receipt["same_ray_hit_percent"].setdefault(label, []).append(coverage)
            panel.save(output / image_name)
        receipt.update(
            status="REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED",
            image="source-measured10-vs-measured4-stereo6-3plus2.png",
            intermediate_image="source-measured10-vs-measured8-stereo2-3plus2.png",
            recorded_samples=[int(Path(v[0]).stem) for v in recorded],
            visual_acceptance="PENDING human review; full room NOT established")

    except Exception as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}",
                       stereo_trials=locals().get("trials", []))
        raise
    finally:
        receipt["elapsed_seconds"] = round(time.time()-started, 2)
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.upstream is None or args.output is None:
            parser.error("--upstream and --output required")
        run(args.upstream, args.output)
