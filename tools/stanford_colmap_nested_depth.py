"""Nested 30/15/10 real measured-depth ablation on ONE frozen 300-RGB COLMAP model.

Input is the completed independent early-overlap COLMAP artifact (run 40970565394),
NOT any earlier full-sequence trajectory or mesh. The 30-depth GLB in that artifact
is reference-only and is never read by this geometry program.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stanford_colmap_metric300 import original_pairs, read_model, depth_scale


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def render_rays(o3d, mesh, rays):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    hit = scene.cast_rays(rays)
    ids = hit["primitive_ids"].numpy().reshape(-1)
    uv = hit["primitive_uvs"].numpy().reshape(-1, 2)
    dist = hit["t_hit"].numpy().reshape(-1)
    ok = np.isfinite(dist) & (dist > 0.15) & (dist < 16.0) & (ids < len(faces))
    pixels = np.zeros((320 * 240, 3), np.uint8)
    if np.any(ok):
        weights = np.column_stack((1.0 - uv[ok].sum(axis=1), uv[ok]))
        values = (colors[faces[ids[ok].astype(int)]] * weights[:, :, None]).sum(axis=1)
        pixels[ok] = np.clip(values * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels.reshape(240, 320, 3)), round(100 * float(ok.mean()), 2)


def run(upstream, output):
    import open3d as o3d
    import trimesh

    upstream, output = upstream.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Output must be fresh")
    output.mkdir(parents=True)
    start = time.time()
    report = dict(
        status="RUNNING", scene="Stanford Lounge original RGB-D",
        pose_source="Frozen independent 300 RGB early-overlap COLMAP, run 40970565394",
        pose_artifact="12939354226", prior_mesh=False, prior_gt_poses=False,
        predicted_depth=False, visual_acceptance="FAIL until human five-view review")
    def save():
        (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        parent = json.loads((upstream / "receipt.json").read_text())
        ledger = json.loads((upstream / "inputs.json").read_text())
        if parent["registered_frames"] != 287 or parent["selected_rgb"] != 300:
            raise ValueError("Unexpected source COLMAP model")
        if len(ledger) != 300 or parent["rgb_sampling"] != "early-overlap":
            raise ValueError("Unexpected 300 RGB input ledger")
        source = original_pairs(o3d.data.LoungeRGBDImages())
        color_by_name, depth_by_name = {}, {}
        for row in ledger:
            index = row["original_index"]
            rgb, dep = source[index]
            if digest(rgb) != row["rgb_sha256"]:
                raise ValueError(f"RGB input changed: {index}")
            name = f"{row['sample']:06d}{rgb.suffix.lower()}"
            color_by_name[name], depth_by_name[name] = rgb, dep

        views = read_model(upstream / parent["model"])
        if len(views) != 287:
            raise ValueError("Source registered view count changed")
        registered = sorted(name for name, _, _ in views)
        base30 = [f"{i:06d}.png" for i in parent["measured_depth_rgb_samples"]]
        if len(base30) != 30 or not set(base30).issubset(registered):
            raise ValueError("Original 30 measured anchors changed")
        for name in base30:
            sample = int(Path(name).stem)
            if digest(depth_by_name[name]) != ledger[sample]["depth_sha256"]:
                raise ValueError(f"Measured depth changed: {sample}")
        base15 = base30[::2]
        base10 = [base15[int(i)] for i in np.linspace(0, 14, 10, dtype=int)]
        if len(set(base10)) != 10 or not set(base10).issubset(base15):
            raise ValueError("10/15/30 depth budgets are not strictly nested")
        budgets = {30: base30, 15: base15, 10: base10}

        # Ten real depth maps provide the ONLY metric calibration for all budgets.
        # This keeps ray origins and scale identical, without a hidden 30-map prior
        # in the 10-depth experiment.
        scale, witnesses = depth_scale(
            [v for v in views if v[0] in set(base10)],
            {name: depth_by_name[name] for name in base10})
        report.update(
            selected_rgb=300, registered_rgb=287,
            metric_scale_source="same 10 measured maps nested in 15 and 30",
            metric_scale_m_per_colmap_unit=scale, metric_scale_witnesses=witnesses,
            measured_depth_samples={
                str(b): [int(Path(name).stem) for name in names]
                for b, names in budgets.items()})
        save()

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 525., 525., 319.5, 239.5)
        meshes = {}
        for budget, names in budgets.items():
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=.025, sdf_trunc=.10,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            chosen = set(names)
            for name, w2c, _ in views:
                if name not in chosen:
                    continue
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.io.read_image(str(color_by_name[name])),
                    o3d.io.read_image(str(depth_by_name[name])),
                    depth_scale=1000., depth_trunc=3.,
                    convert_rgb_to_intensity=False)
                metric = w2c.copy()
                metric[:3, 3] *= scale
                volume.integrate(rgbd, intrinsic, metric)
            mesh = volume.extract_triangle_mesh()
            if not mesh.has_triangles() or not mesh.has_vertex_colors():
                raise RuntimeError(f"Missing colored real Mesh at {budget} depths")
            meshes[budget] = mesh
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            colors = np.clip(np.asarray(mesh.vertex_colors) * 255, 0, 255).astype(np.uint8)
            path = output / f"measured{budget}.glb"
            trimesh.Trimesh(vertices, faces, vertex_colors=colors,
                            process=False).export(path)
            report[f"measured{budget}_glb_sha256"] = digest(path)
            report[f"measured{budget}_triangles"] = len(faces)
            save()

        # Fixed three registered RGB viewpoints and two fixed world-space orbit
        # rays across ALL three meshes. No model-dependent camera re-framing.
        view_by_name = {name: w2c for name, w2c, _ in views}
        recorded = [registered[0], registered[len(registered) // 2], registered[-1]]
        K = np.array([[525., 0, 319.5], [0, 525., 239.5], [0, 0, 1.]])
        small = K.copy()
        small[:2] /= 2
        small[0, 2] = (K[0, 2] + .5) / 2 - .5
        small[1, 2] = (K[1, 2] + .5) / 2 - .5
        rays = []
        for name in recorded:
            ext = view_by_name[name].copy()
            ext[:3, 3] *= scale
            rays.append(o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                o3d.core.Tensor(small.astype(np.float32)),
                o3d.core.Tensor(ext.astype(np.float32)), 320, 240))
        mesh30 = meshes[30]
        center = (mesh30.get_min_bound() + mesh30.get_max_bound()) / 2
        span = mesh30.get_max_bound() - mesh30.get_min_bound()
        for offset in ([-.5, -.9, -.7], [.8, -1.2, .05]):
            eye = center + span * np.asarray(offset)
            rays.append(o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                78., o3d.core.Tensor(center.astype(np.float32)),
                o3d.core.Tensor(eye.astype(np.float32)),
                o3d.core.Tensor(np.array([0., -1., 0.], np.float32)),
                320, 240))

        labels = [f"recorded source sample {Path(n).stem}" for n in recorded]
        labels += ["free orbit A", "free orbit B"]
        panel = Image.new("RGB", (1280, 5 * 270 + 30), (18, 18, 18))
        draw = ImageDraw.Draw(panel)
        for col, label in enumerate(("Original RGB", "30 measured", "15 measured", "10 measured")):
            draw.text((col * 320 + 7, 8), label, fill="white")
        report["same_ray_hit_percent"] = {}
        for i, ray in enumerate(rays):
            y = 30 + 270 * i
            draw.text((6, y + 3), labels[i], fill="white")
            if i < 3:
                with Image.open(color_by_name[recorded[i]]) as photo:
                    panel.paste(photo.convert("RGB").resize((320, 240)), (0, y + 26))
            for col, budget in enumerate((30, 15, 10), start=1):
                rendered, coverage = render_rays(o3d, meshes[budget], ray)
                panel.paste(rendered, (col * 320, y + 26))
                report["same_ray_hit_percent"].setdefault(str(budget), []).append(coverage)
        panel.save(output / "source-depth30-15-10-3plus2.png")
        report.update(
            status="REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED",
            image="source-depth30-15-10-3plus2.png",
            recorded_samples=[int(Path(n).stem) for n in recorded],
            free_camera_reference="30-depth mesh bounds; same fixed rays for all budgets")
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = round(time.time() - start)
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.upstream, args.output)
