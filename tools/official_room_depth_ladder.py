"""Re-fuse a frozen Open3D Lounge RGB-D pose solution with fewer measured depth frames.

This is a FUSION-STAGE ablation: all camera poses were estimated upstream from the
full RGB-D sequence, so sparse-depth runs are not independent capture pipelines.
No prior mesh enters sparse TSDF integration. Requires Open3D 0.19, trimesh, PIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_RUN = 40865675012
SOURCE_ARTIFACT = 12896335330
UPSTREAM_SHA = "1e7b17438687a0b0c1e5a7187321ac7044afe275"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trajectory(path: Path) -> dict[int, np.ndarray]:
    lines = path.read_text().splitlines()
    if not lines or len(lines) % 5:
        raise ValueError("Invalid Open3D trajectory.log record count")
    poses = {}
    for k in range(0, len(lines), 5):
        header = [int(v) for v in lines[k].split()]
        if len(header) != 3 or header[0] != header[1] or header[2] != header[0] + 1:
            raise ValueError(f"Invalid Open3D pose header: {lines[k]}")
        idx = header[0]
        if idx in poses:
            raise ValueError(f"Duplicate pose index {idx}")
        values = [float(v) for row in lines[k + 1:k + 5] for v in row.split()]
        if len(values) != 16:
            raise ValueError(f"Incomplete pose matrix {idx}")
        pose = np.asarray(values, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError(f"Invalid homogeneous pose {idx}")
        poses[idx] = pose
    return poses


def native_pairs(color_dir: Path, depth_dir: Path, poses: dict[int, np.ndarray]):
    """Upstream trajectory headers are ordinal indices, NOT image filename IDs."""
    colors = {int(p.stem): p for p in color_dir.iterdir()
              if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
              and p.stem.isdecimal()}
    depths = {int(p.stem): p for p in depth_dir.iterdir()
              if p.is_file() and p.suffix.lower() == ".png" and p.stem.isdecimal()}
    ids = sorted(colors)
    if (not ids or colors.keys() != depths.keys() or
            len(ids) != len(poses) or sorted(poses) != list(range(len(ids))) or
            ids != list(range(ids[0], ids[0] + len(ids)))):
        raise ValueError(f"RGB/depth/pose identity mismatch: "
                         f"RGB={len(colors)}, depth={len(depths)}, pose={len(poses)}, "
                         f"first/last RGB IDs={ids[:2]}/{ids[-2:]}, "
                         f"pose indices={sorted(poses)[:2]}/{sorted(poses)[-2:]}")
    return [(ordinal, colors[frame_id], depths[frame_id], poses[ordinal])
            for ordinal, frame_id in enumerate(ids)]


def even_indices(length: int, count: int) -> list[int]:
    if not 2 <= count <= length:
        raise ValueError("Each measured-depth sample count must be between 2 and total frames")
    indices = np.linspace(0, length - 1, count, dtype=int).tolist()
    if len(set(indices)) != count:
        raise ValueError("Sampling produced duplicate frames")
    return indices


def integrate_subset(o3d, trimesh, pairs, count, output: Path, voxel: float):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    chosen = even_indices(len(pairs), count)
    for n, i in enumerate(chosen):
        frame_id, color_path, depth_path, pose = pairs[i]
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(str(color_path)), o3d.io.read_image(str(depth_path)),
            depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        if (n + 1) % 50 == 0:
            print(f"depth {count}: fused {n+1}/{count} measured frames", flush=True)
    mesh = volume.extract_triangle_mesh()
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise RuntimeError(f"No colored geometry with {count} measured frames")
    mesh.compute_vertex_normals()
    xyz = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.clip(np.asarray(mesh.vertex_colors) * 255, 0, 255).astype(np.uint8)
    trimesh.Trimesh(vertices=xyz, faces=faces, vertex_colors=colors,
                    process=False).export(output / f"measured-{count}.glb")
    return mesh, {"measured_depth_frames": count, "triangles": len(faces),
                  "vertices": len(xyz), "sample_pose_ordinals": [pairs[i][0] for i in chosen],
                  "sample_native_color_files": [pairs[i][1].name for i in chosen],
                  "glb_sha256": sha256(output / f"measured-{count}.glb")}


def render_fixed_views(o3d, mesh, pairs, bounds, output: Path):
    """Render identical recorded and world-space free rays for every Mesh."""
    from PIL import Image, ImageDraw
    faces = np.asarray(mesh.triangles)
    rgb = np.asarray(mesh.vertex_colors)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    k = np.array([[525., 0., 319.5], [0., 525., 239.5], [0., 0., 1.]])
    k[:2] /= 2
    k[0, 2] = (319.5 + .5) / 2 - .5
    k[1, 2] = (239.5 + .5) / 2 - .5
    center = (bounds[0] + bounds[1]) / 2
    span = bounds[1] - bounds[0]
    views = []
    for i in (0, len(pairs)//2, len(pairs)-1):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(k.astype(np.float32)),
            o3d.core.Tensor(np.linalg.inv(pairs[i][3]).astype(np.float32)),
            320, 240)
        views.append((f"recorded-{i:05d}.png", rays, pairs[i][1]))
    for j, offset in enumerate(([-.5, -.9, -.7], [.8, -1.2, .05]), 1):
        eye = center + span * np.asarray(offset)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            78., o3d.core.Tensor(center.astype(np.float32)),
            o3d.core.Tensor(eye.astype(np.float32)),
            o3d.core.Tensor(np.array([0., -1., 0.], np.float32)), 320, 240)
        views.append((f"free-{j}.png", rays, None))
    folder = output / "views"
    folder.mkdir()
    for name, rays, photo in views:
        hit = scene.cast_rays(rays)
        ids = hit["primitive_ids"].numpy().reshape(-1)
        uv = hit["primitive_uvs"].numpy().reshape(-1, 2)
        dist = hit["t_hit"].numpy().reshape(-1)
        valid = np.isfinite(dist) & (dist > .15) & (dist < 16.) & (ids < len(faces))
        pixels = np.zeros((320 * 240, 3), np.uint8)
        if np.any(valid):
            w = np.column_stack((1 - uv[valid].sum(axis=1), uv[valid]))
            pixels[valid] = np.clip(
                (rgb[faces[ids[valid].astype(int)]] * w[:, :, None]).sum(axis=1) * 255,
                0, 255).astype(np.uint8)
        panels = [Image.fromarray(pixels.reshape(240, 320, 3))]
        if photo is not None:
            with Image.open(photo) as image:
                panels.insert(0, image.convert("RGB").resize((320, 240)))
        canvas = Image.new("RGB", (320 * len(panels), 268), (24, 24, 24))
        ImageDraw.Draw(canvas).text((8, 8), name, fill="white")
        for j, panel in enumerate(panels):
            canvas.paste(panel, (320 * j, 28))
        canvas.save(folder / name)


def comparison_image(output: Path, labels: list[str]):
    from PIL import Image, ImageDraw
    folders = [output / label / "views" for label in labels]
    names = ["recorded-00000.png", "recorded-01500.png", "recorded-02999.png",
             "free-1.png", "free-2.png"]
    canvas = Image.new("RGB", (1280, 5 * 272 + 38), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(["REAL RGB / free view", "FULL RGB-D", "600 measured Z", "150 measured Z"]):
        draw.text((col * 320 + 8, 12), title, fill="white")
    for row, name in enumerate(names):
        y = 38 + row * 272
        for col in range(4):
            folder = folders[0 if col == 0 else col - 1]
            with Image.open(folder / name) as img:
                img = img.convert("RGB")
                if row < 3:
                    panel = img.crop((0 if col == 0 else img.width - 320, 28,
                                      320 if col == 0 else img.width, 268))
                else:
                    panel = img.crop((img.width - 320, 28, img.width, 268))
                if col == 0 and row >= 3:
                    panel = Image.new("RGB", (320, 240), (22, 22, 22))
                canvas.paste(panel, (col * 320, y + 24))
        draw.text((8, y + 3), name.replace(".png", ""), fill="white")
    target = output / "source-full-600-150-3plus2.png"
    canvas.save(target)
    return target


def run(official: Path, output: Path, counts=(600, 150)):
    import open3d as o3d
    import trimesh

    official, output = official.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Fresh output directory required")
    receipt = json.loads((official / "receipt.json").read_text())
    if receipt.get("upstream_commit") != UPSTREAM_SHA or receipt.get("real_frames") != 2999:
        raise ValueError("Unexpected official source provenance")
    poses = load_trajectory(official / "scene" / "trajectory.log")
    if len(poses) != 3000:
        raise ValueError(f"Expected original 3000 upstream poses, got {len(poses)}")
    original = o3d.data.LoungeRGBDImages()
    color_dir = Path(original.color_paths[0]).parent
    depth_dir = Path(original.depth_paths[0]).parent
    pairs = native_pairs(color_dir, depth_dir, poses)
    if len(pairs) != 3000:
        raise ValueError(f"Expected original 3000 frame identities, got {len(pairs)}")
    output.mkdir(parents=True)
    baseline = o3d.io.read_triangle_mesh(str(official / "scene" / "integrated.ply"))
    if not baseline.has_triangles() or not baseline.has_vertex_colors():
        raise RuntimeError("Official baseline has no colored triangles")
    print(f"Recovered {len(pairs)} actual RGB-D frames / {len(poses)} poses", flush=True)
    models = {"full": baseline}
    stats = {"full": {"measured_depth_frames": 3000,
                      "triangles": len(baseline.triangles),
                      "glb_sha256": sha256(official / "room-full.glb")}}
    for count in counts:
        if count >= len(pairs):
            raise ValueError("Sparse count must be less than original frame count")
        mesh, stat = integrate_subset(o3d, trimesh, pairs, count, output, 3.0 / 512.0)
        models[str(count)] = mesh
        stats[str(count)] = stat
    fixed_bounds = np.array([baseline.get_min_bound(), baseline.get_max_bound()])
    for key, mesh in models.items():
        target = output / key
        target.mkdir()
        render_fixed_views(o3d, mesh, pairs, fixed_bounds, target)
    labels = ["full", str(counts[0]), str(counts[1])]
    image_path = comparison_image(output, labels)
    report = {
        "status": "GENERATED_NOT_VISUALLY_ACCEPTED", "scene": "Stanford Lounge",
        "source_run": SOURCE_RUN, "source_artifact": SOURCE_ARTIFACT,
        "source_upstream_sha": UPSTREAM_SHA, "open3d_version": o3d.__version__,
        "input_real_rgb_frames": len(pairs),
        "original_input_ledger_count": receipt["real_frames"],
        "ledger_correction": "Open3D data.color_paths exposed 2999 files while the native "
                             "directories contain 3000. Trajectory indices are ordinals, "
                             "not numeric filename stems; all 3000 are paired by verified order.",
        "pose_source": "Estimated from ALL 3000 measured RGB-D frames by official upstream; "
                       "frozen prior in sparse-depth integration (not RGB-only).",
        "mesh_prior_in_sparse_runs": False,
        "measured_depth_units": "millimeters, converted by depth_scale=1000",
        "predicted_depth_frames": 0,
        "free_camera_reference": "Fixed from original full official Mesh bounds for all three models",
        "fusion_config": {"voxel_m": 3.0 / 512.0, "sdf_trunc_m": 0.04,
                          "depth_max_m": 3.0, "intrinsic": "PrimeSenseDefault"},
        "models": stats, "visual": image_path.name,
        "source_full_glb": f"https://github.com/REDrighthand05/echo4d/actions/runs/{SOURCE_RUN}/artifacts/{SOURCE_ARTIFACT}",
        "visual_acceptance": "PENDING_HUMAN_REVIEW; do not claim complete room",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "models": stats,
                      "visual": str(image_path)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, nargs=2, default=[600, 150])
    args = parser.parse_args()
    run(args.official, args.output, args.samples)
