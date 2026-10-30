"""Choose ten measured RGB-D frames using only frozen RGB-COLMAP camera poses.

Compares the existing uniform-in-time ten-depth selection to a pose-frustum
coverage selection on the *same* 300 original RGB photos and 287 registered
COLMAP cameras (run 40970565394). Both ten-depth meshes are fused from scratch,
using the same metric scale calibrated by FOUR depth frames common to both
selections. Neither the 30-depth reference mesh nor any full-sequence pose,
predicted depth, or GPU is used. Pose frusta are approximate: not observed walls.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stanford_colmap_metric300 import original_pairs, read_model, depth_scale
from stanford_colmap_nested_depth import render_rays


UPSTREAM_RUN = "https://github.com/REDrighthand05/echo4d/actions/runs/40970565394"
DEPTH_METERS = (0.8, 1.5, 2.5)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_pose_coverage(views, fixed, budget, scale):
    """Greedy camera-frustum proxy coverage; requires NO measured depth or mesh.

    Proxy points are ray samples in front of each registered RGB camera at
    nominal distances. A candidate covers a proxy if it is inside that
    camera's RGB frustum and within the fusion's 3 m depth truncation.
    This is a pose-only *hypothesis* about visibility, not observed geometry.
    """
    ordered = sorted(views, key=lambda v: v[0])
    names = [v[0] for v in ordered]
    if len(set(names)) != len(names) or len(set(fixed)) != len(fixed):
        raise ValueError("Camera names and fixed anchors must be unique")
    if not set(fixed).issubset(names) or not len(fixed) < budget <= len(names):
        raise ValueError("Invalid fixed anchors or depth budget")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Metric scale must be positive and finite")

    rotation = np.stack([v[1][:3, :3] for v in ordered])
    translation = np.stack([v[1][:3, 3] for v in ordered])
    inv_rotation = np.transpose(rotation, (0, 2, 1))
    centers = -np.einsum("nij,nj->ni", inv_rotation, translation) * scale

    # 3x3 RGB rays, three physical distances, from every registered camera.
    rays = np.array([[x, y, 1.0]
                     for y in (-0.32, 0.0, 0.32)
                     for x in (-0.42, 0.0, 0.42)], dtype=np.float64)
    world_rays = np.einsum("nij,pj->npi", inv_rotation, rays)
    targets = (centers[:, None, None, :] +
               world_rays[:, :, None, :] *
               np.asarray(DEPTH_METERS)[None, None, :, None]).reshape(-1, 3)

    visible = np.empty((len(names), len(targets)), dtype=bool)
    for i in range(len(names)):
        xyz = (targets - centers[i]) @ rotation[i].T
        z = xyz[:, 2]
        visible[i] = ((z > 0.4) & (z < 3.0) &
                      (np.abs(xyz[:, 0]) < 0.60 * z) &
                      (np.abs(xyz[:, 1]) < 0.45 * z))

    # Do not let dozens of near-identical early-video poses dominate the room.
    cells = np.floor(centers / 0.45).astype(np.int64)
    _, cell_ids, cell_counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True)
    point_weights = np.repeat(1.0 / np.sqrt(cell_counts[cell_ids]), 27)
    chosen = [names.index(name) for name in fixed]
    hit_counts = visible[chosen].sum(axis=0).astype(np.float32)
    while len(chosen) < budget:
        gains = visible @ (point_weights / np.square(1.0 + hit_counts))
        gains[chosen] = -np.inf
        index = int(np.argmax(gains))
        if not np.isfinite(gains[index]):
            raise RuntimeError("No remaining camera for pose-only selection")
        chosen.append(index)
        hit_counts += visible[index]
    return sorted(names[i] for i in chosen)


def self_test():
    views = []
    for i in range(24):
        pose = np.eye(4)
        center = np.array([np.sin(i * 0.28), 0.05 * i, i * 0.09])
        pose[:3, 3] = -center
        views.append((f"{i:06d}.png", pose, []))
    fixed = [views[i][0] for i in (0, 7, 14, 21)]
    first = select_pose_coverage(views, fixed, 10, 1.0)
    second = select_pose_coverage(list(reversed(views)), fixed, 10, 1.0)
    assert first == second and len(first) == len(set(first)) == 10
    assert set(fixed).issubset(first)
    try:
        select_pose_coverage(views, fixed, 3, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid depth budget was accepted")
    print("PASS: deterministic pose-only ten-camera selection with fixed anchors")


def metric_extrinsic(w2c, scale):
    ext = w2c.copy()
    ext[:3, 3] *= scale
    return ext


def fuse(o3d, trimesh, views, names, scale, color_paths, depth_paths, destination):
    chosen = set(names)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.025, sdf_trunc=0.10,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intr = o3d.camera.PinholeCameraIntrinsic(
        640, 480, 525.0, 525.0, 319.5, 239.5)
    for name, w2c, _ in views:
        if name not in chosen:
            continue
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(str(color_paths[name])),
            o3d.io.read_image(str(depth_paths[name])),
            depth_scale=1000.0, depth_trunc=3.0,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intr, metric_extrinsic(w2c, scale))
    mesh = volume.extract_triangle_mesh()
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise RuntimeError("Real colored TSDF mesh was not generated")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.clip(np.asarray(mesh.vertex_colors) * 255, 0, 255).astype(np.uint8)
    trimesh.Trimesh(vertices, faces, vertex_colors=colors, process=False).export(destination)
    return mesh


def five_rays(o3d, views, scale):
    ordered = sorted(views, key=lambda v: v[0])
    recorded = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]])
    small = K.copy()
    small[:2] /= 2.0
    small[0, 2] = (K[0, 2] + 0.5) / 2.0 - 0.5
    small[1, 2] = (K[1, 2] + 0.5) / 2.0 - 0.5
    rays = []
    for _, w2c, _ in recorded:
        rays.append(o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(small.astype(np.float32)),
            o3d.core.Tensor(metric_extrinsic(w2c, scale).astype(np.float32)),
            320, 240))

    # Room-scale free orbit, using only RGB-COLMAP poses, never mesh bounds.
    # Estimate an observed-space focus from forward rays, and global upright
    # from camera orientations. Unlike a camera-local offset this avoids
    # rotated close-ups that hide whether the room itself is recognizable.
    camera_to_world = [np.linalg.inv(metric_extrinsic(w2c, scale))
                       for _, w2c, _ in ordered]
    centers = np.stack([p[:3, 3] for p in camera_to_world])
    forwards = np.stack([p[:3, 2] for p in camera_to_world])
    ups = np.stack([-p[:3, 1] for p in camera_to_world])
    target = np.median(centers + 1.5 * forwards, axis=0)
    up = ups.mean(axis=0)
    up /= np.linalg.norm(up)
    back = centers.mean(axis=0) - target
    back -= np.dot(back, up) * up
    if np.linalg.norm(back) < 0.1:
        back = -forwards.mean(axis=0)
        back -= np.dot(back, up) * up
    back /= np.linalg.norm(back)
    side = np.cross(up, back)
    side /= np.linalg.norm(side)
    spread = np.linalg.norm(np.percentile(centers, 90, axis=0) -
                            np.percentile(centers, 10, axis=0))
    reach = float(np.clip(spread * 0.65, 2.2, 3.5))
    for sign, height in ((1.0, 0.75), (-1.0, 0.55)):
        eye = target + reach * back + sign * 0.45 * reach * side + height * reach * up
        rays.append(o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            78.0, o3d.core.Tensor(target.astype(np.float32)),
            o3d.core.Tensor(eye.astype(np.float32)),
            o3d.core.Tensor(up.astype(np.float32)), 320, 240))
    return recorded, rays


def run(upstream, output):
    import open3d as o3d
    import trimesh

    upstream, output = upstream.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Output must be fresh")
    output.mkdir(parents=True)
    started = time.time()
    receipt = dict(
        status="RUNNING", scene="Stanford Lounge original measured RGB-D",
        upstream_run=UPSTREAM_RUN, upstream_artifact_id=12939354226,
        pose_source="frozen 300-RGB independent COLMAP (287 registered)",
        prior_mesh=False, prior_gt_poses=False, predicted_depth=False,
        selection_source="RGB-derived camera positions and orientations only",
        visual_acceptance="FAIL until five-view inspection",
        free_orbit_acceptance="FAIL until verified")
    def save():
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    try:
        parent = json.loads((upstream / "receipt.json").read_text())
        ledger = json.loads((upstream / "inputs.json").read_text())
        if (parent["registered_frames"], parent["selected_rgb"],
                parent["rgb_sampling"]) != (287, 300, "early-overlap"):
            raise ValueError("Unexpected frozen RGB-only COLMAP source")
        if len(ledger) != 300:
            raise ValueError("300 RGB ledger missing")
        views = read_model(upstream / parent["model"])
        if len(views) != 287:
            raise ValueError("Wrong COLMAP registered camera count")
        registered = sorted(name for name, _, _ in views)
        lookup = {int(Path(name).stem): name for name in registered}
        base30 = [lookup[i] for i in parent["measured_depth_rgb_samples"]]
        base15 = base30[::2]
        base10 = [base15[int(i)] for i in np.linspace(0, 14, 10, dtype=int)]
        if len(set(base10)) != 10:
            raise ValueError("Original ten measured anchors are not unique")

        # FOUR common real depth maps are the only metric-scale witnesses for
        # BOTH ten-map variants. No hidden 30-map or 10+10-map scale prior.
        shared = [base10[i] for i in (0, 3, 6, 9)]
        source = original_pairs(o3d.data.LoungeRGBDImages())
        rgb_paths, depth_paths = {}, {}
        for row in ledger:
            i = int(row["sample"])
            rgb, depth = source[int(row["original_index"])]
            if sha256(rgb) != row["rgb_sha256"]:
                raise ValueError(f"Changed original RGB sample {i}")
            name = f"{i:06d}{rgb.suffix.lower()}"
            rgb_paths[name], depth_paths[name] = rgb, depth
        for name in shared:
            i = int(Path(name).stem)
            if sha256(depth_paths[name]) != ledger[i]["depth_sha256"]:
                raise ValueError(f"Changed metric-anchor depth sample {i}")
        scale, witnesses = depth_scale(
            [v for v in views if v[0] in set(shared)],
            {name: depth_paths[name] for name in shared})
        optimized = select_pose_coverage(views, shared, 10, scale)
        if set(optimized) == set(base10):
            raise RuntimeError("Pose-only selection returned the unchanged ten-depth baseline")
        if not set(shared).issubset(optimized):
            raise RuntimeError("Optimized depth selection lost common scale anchors")
        for name in set(base10) | set(optimized):
            i = int(Path(name).stem)
            if sha256(depth_paths[name]) != ledger[i]["depth_sha256"]:
                raise ValueError(f"Changed measured depth sample {i}")

        receipt.update(
            selected_rgb=300, registered_rgb=287,
            measured_depth_per_mesh=10,
            metric_scale_m_per_colmap_unit=scale,
            metric_scale_witnesses=witnesses,
            metric_scale_source="four identical measured maps in both ten-depth sets",
            common_scale_depth_samples=[int(Path(n).stem) for n in shared],
            temporal_depth_samples=[int(Path(n).stem) for n in base10],
            pose_cover_depth_samples=[int(Path(n).stem) for n in optimized],
            selection_algorithm="greedy 3x3 RGB-frustum proxies at 0.8/1.5/2.5m, spatial density balanced")
        save()

        old = fuse(o3d, trimesh, views, base10, scale, rgb_paths, depth_paths,
                   output / "temporal10.glb")
        new = fuse(o3d, trimesh, views, optimized, scale, rgb_paths, depth_paths,
                   output / "posecover10.glb")
        receipt.update(
            temporal10_glb_sha256=sha256(output / "temporal10.glb"),
            posecover10_glb_sha256=sha256(output / "posecover10.glb"))
        save()

        recorded, rays = five_rays(o3d, views, scale)
        labels = [f"recorded source sample {Path(v[0]).stem}" for v in recorded]
        labels += ["free room orbit A (RGB pose only)", "free room orbit B (RGB pose only)"]
        panel = Image.new("RGB", (960, 5 * 270 + 30), (18, 18, 18))
        draw = ImageDraw.Draw(panel)
        for col, label in enumerate(("Original RGB", "temporal 10 depth", "pose-cover 10 depth")):
            draw.text((col * 320 + 7, 8), label, fill="white")
        receipt["same_ray_hit_percent"] = {}
        for i, ray in enumerate(rays):
            y = 30 + 270 * i
            draw.text((6, y + 3), labels[i], fill="white")
            if i < 3:
                with Image.open(rgb_paths[recorded[i][0]]) as photo:
                    panel.paste(photo.convert("RGB").resize((320, 240)), (0, y + 26))
            for col, (name, mesh) in enumerate((("temporal10", old), ("posecover10", new)), 1):
                rendered, coverage = render_rays(o3d, mesh, ray)
                panel.paste(rendered, (col * 320, y + 26))
                receipt["same_ray_hit_percent"].setdefault(name, []).append(coverage)
        image_name = "source-temporal10-vs-posecover10-3plus2.png"
        panel.save(output / image_name)
        receipt.update(
            status="REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED",
            image=image_name,
            recorded_samples=[int(Path(v[0]).stem) for v in recorded],
            free_camera_reference="room-scale RGB-COLMAP pose-only orbit; identical rays for both meshes",
            visual_acceptance="PENDING human comparison; full room NOT established")
    except Exception as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = round(time.time() - started)
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
