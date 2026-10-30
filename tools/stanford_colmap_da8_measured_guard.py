"""Rebuild Stanford Lounge 8 sensor + 8 CPU DA2 depths with cross-view occlusion guard.

The unfiltered #159 GLB is for comparison ONLY. No held-out sensor depth at
predicted target views enters selection, calibration, or fusion. Contradictions
are checked only against the eight *input* measured depths and the other seven
predicted maps. Unseen pixels remain unknown rather than being hallucinated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import stanford_colmap_da8_spatial as base
from stanford_colmap_metric300 import read_model
from stanford_colmap_nested_depth import render_rays
from stanford_colmap_pose_cover10 import five_rays, sha256
from stanford_colmap_stereo6 import CAMERA, camera_to_world
from stanford_colmap_eight_measured_control import old_mesh_to_open3d


def reject_front_ghosts(predictions, measured, *, fx, fy, cx, cy,
                        sample_step=2):
    """Return guarded float32 depths and per-target measured/predicted votes.

    A point in FRONT of an independently observed surface would have occluded
    that surface, so it is inconsistent. A point BEHIND a surface is merely
    unobserved and is never rejected by that observation. Votes at depth
    discontinuities are excluded. All reference maps are reconstruction
    inputs, not held-out ground truth.
    """
    references = []
    for name, (depth, c2w) in measured.items():
        references.append((name, depth, c2w, True))
    for name, (_, depth, _, c2w) in predictions.items():
        references.append((name, depth, c2w, False))
    guarded, diagnostics = {}, {}
    for target, (rgb, depth, cam, pose) in predictions.items():
        h, w = depth.shape
        ys, xs = np.mgrid[0:h:sample_step, 0:w:sample_step]
        z = depth[::sample_step, ::sample_step].astype(np.float64)
        valid = np.isfinite(z) & (z > 0.55) & (z < 2.9)
        pc = np.stack(((xs - cx) / fx * z, (ys - cy) / fy * z, z), axis=-1)
        world = pc.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
        sensor_votes = np.zeros(z.size, dtype=np.uint8)
        predicted_votes = np.zeros(z.size, dtype=np.uint8)
        sensor_strong = np.zeros(z.size, dtype=bool)
        for name, ref_depth, ref_pose, is_sensor in references:
            if name == target:
                continue
            inv = np.linalg.inv(ref_pose)
            projected = world @ inv[:3, :3].T + inv[:3, 3]
            zz = projected[:, 2]
            safe = np.isfinite(zz) & (zz > 0.55) & (zz < 2.9)
            # Never convert invalid/negative projections to pixel indices.
            zz_safe = np.where(safe, zz, 1.0)
            u = np.rint(fx * projected[:, 0] / zz_safe + cx).astype(np.int32)
            v = np.rint(fy * projected[:, 1] / zz_safe + cy).astype(np.int32)
            inside = safe & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
            indices = np.flatnonzero(inside)
            if not len(indices):
                continue
            uu, vv = u[indices], v[indices]
            observed = ref_depth[vv, uu]
            # Exclude silhouettes and depth jumps, where nearest-pixel
            # reprojection can falsely suggest a front-surface conflict.
            smooth = np.isfinite(observed) & (observed > 0.55) & (observed < 2.9)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbour = ref_depth[vv + dy, uu + dx]
                smooth &= np.isfinite(neighbour) & (
                    np.abs(neighbour - observed) < (0.08 if is_sensor else 0.12))
            delta = observed - zz[indices]
            conflict = smooth & (delta > (0.14 if is_sensor else 0.18))
            strong = smooth & (delta > 0.24)
            if is_sensor:
                sensor_votes[indices] += conflict.astype(np.uint8)
                sensor_strong[indices] |= strong
            else:
                predicted_votes[indices] += conflict.astype(np.uint8)
        # One unequivocal measured contradiction is enough; otherwise
        # require independent corroboration, not a single predicted map.
        remove = valid.ravel() & (
            sensor_strong |
            (sensor_votes >= 2) |
            ((sensor_votes >= 1) & (predicted_votes >= 1)) |
            (predicted_votes >= 3))
        coarse = remove.reshape(z.shape)
        full = np.repeat(np.repeat(coarse, sample_step, axis=0),
                         sample_step, axis=1)[:h, :w]
        corrected = depth.copy()
        corrected[full] = 0.0
        guarded[target] = (rgb, corrected.astype(np.float32), cam, pose)
        diagnostics[target] = dict(
            removed_pixels=int(np.count_nonzero(full & (depth > 0))),
            original_valid_pixels=int(np.count_nonzero(depth > 0)),
            removed_percent=round(100 * np.count_nonzero(full & (depth > 0)) /
                                  max(1, np.count_nonzero(depth > 0)), 3),
            source_measured_views=len(measured),
            source_predicted_views=len(predictions) - 1,
            heldout_target_sensor_used=False)
    return guarded, diagnostics


def self_test():
    h = w = 12
    pose = np.eye(4)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    def run(z):
        d = np.full((h, w), z, dtype=np.float32)
        inp = {"a": (rgb, d, None, pose)}
        measured = {"b": (np.full((h, w), 2.0, dtype=np.float32), pose)}
        return reject_front_ghosts(inp, measured, fx=8, fy=8,
                                   cx=5.5, cy=5.5)[0]["a"][1]
    assert np.count_nonzero(run(1.0) == 0) > 0, "front ghost not rejected"
    assert np.all(run(2.0) == 2.0), "consistent surface rejected"
    assert np.all(run(2.5) == 2.5), "occluded/behind surface rejected"
    print("PASS: measured front-surface occlusion guard and unknown preservation")


def run(upstream, control, da2, da8, output):
    import open3d as o3d
    import trimesh

    output, da8 = Path(output).resolve(), Path(da8).resolve()
    old_receipt = json.loads((da8 / "receipt.json").read_text())
    old_path = da8 / "measured8_da8.glb"
    if sha256(old_path) != old_receipt["new_glb_sha256"]:
        raise ValueError("Frozen #159 comparison GLB hash mismatch")
    original_fuse = base.fuse_hybrid
    evidence = {}

    def guarded_fuse(o3d_module, trimesh_module, views, measured_names, inferred,
                     scale, rgb_paths, depth_paths, dest):
        measured = {}
        poses = {name: w2c for name, w2c, _ in views}
        for name in measured_names:
            with Image.open(depth_paths[name]) as im:
                depth = np.asarray(im, dtype=np.float32) / 1000.0
            measured[name] = (depth, camera_to_world(poses[name], scale))
        corrected, stats = reject_front_ghosts(
            inferred, measured, fx=CAMERA.fx, fy=CAMERA.fy,
            cx=CAMERA.cx, cy=CAMERA.cy)
        evidence.update(stats)
        for name, (_, depth, _, _) in corrected.items():
            Image.fromarray(np.clip(depth * 1000, 0, 65535).astype(np.uint16)).save(
                output / f"guarded-{Path(name).stem}-mm.png")
        return original_fuse(o3d_module, trimesh_module, views, measured_names,
                             corrected, scale, rgb_paths, depth_paths, dest)

    # Existing 8+8 prediction/calibration/fusion path stays intact. Only the
    # predicted depth arrays are guarded immediately before the TSDF.
    base.fuse_hybrid = guarded_fuse
    try:
        base.run(upstream, control, da2, output)
    finally:
        base.fuse_hybrid = original_fuse
    generated = output / "measured8_da8.glb"
    renamed = output / "measured8_da8_guarded.glb"
    generated.rename(renamed)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["new_glb_sha256"] = sha256(renamed)
    receipt["unfiltered_da8_glb_sha256"] = sha256(old_path)
    receipt["guarded_glb"] = renamed.name
    receipt["cross_view_occlusion_guard"] = evidence
    receipt["sensor_target_holdouts_used_in_fusion"] = False
    receipt["visual_acceptance"] = "PENDING direct 3+2 review; complete room FAIL"

    parent = json.loads((Path(upstream) / "receipt.json").read_text())
    views = read_model(Path(upstream) / parent["model"])
    old = old_mesh_to_open3d(o3d, trimesh, old_path)
    new = old_mesh_to_open3d(o3d, trimesh, renamed)
    recorded, rays = five_rays(o3d, views, receipt["metric_scale_m_per_colmap_unit"])
    panel = Image.new("RGB", (960, 5 * 270 + 30), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    for col, title in enumerate(("Original RGB", "Unfiltered 8+8", "Guarded 8+8")):
        draw.text((col * 320 + 5, 8), title, fill="white")
    hits = {"unfiltered_da8": [], "guarded_da8": []}
    for i, ray in enumerate(rays):
        y = 30 + i * 270
        draw.text((5, y + 3), f"recorded {Path(recorded[i][0]).stem}" if i < 3
                  else f"free orbit {i-2}", fill="white")
        if i < 3:
            path = output / f"source-native-{Path(recorded[i][0]).stem}.png"
            with Image.open(path) as im:
                panel.paste(im.convert("RGB").resize((320, 240)), (0, y + 26))
        for col, (label, mesh) in enumerate(
                (("unfiltered_da8", old), ("guarded_da8", new)), 1):
            picture, hit = render_rays(o3d, mesh, ray)
            panel.paste(picture, (col * 320, y + 26))
            hits[label].append(hit)
    image_name = "source-da8-vs-guarded-3plus2.png"
    panel.save(output / image_name)
    receipt["guarded_comparison_image"] = image_name
    receipt["same_camera_ray_hits"] = hits
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"guarded_glb": str(renamed), "image": image_name,
                      "removed_pixels": {k: v["removed_pixels"]
                                         for k, v in evidence.items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--da2", type=Path)
    parser.add_argument("--da8", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not all((args.upstream, args.control, args.da2, args.da8, args.output)):
            parser.error("All four input artifacts and --output required")
        run(args.upstream, args.control, args.da2, args.da8, args.output)
