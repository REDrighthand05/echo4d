"""Ablation control: eight *measured* depths alone versus the same eight plus
two CPU RGB-stereo inferred maps, using exactly the original 300 RGB images.

The 8+2 GLB is read ONLY as a comparison artifact, never a fusion prior.
Re-fuse eight measured maps from original sensor images, fixed RGB-COLMAP
poses and the four identical measured scale anchors. No GPU or GT poses.
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
from stanford_colmap_stereo6 import fuse_hybrid
from stanford_colmap_pose_cover10 import (
    UPSTREAM_RUN, sha256, select_pose_coverage, five_rays,
)


def eight_only_names(ten_names, replaced_samples):
    """Remove precisely the two measured maps replaced by RGB stereo."""
    names = list(ten_names)
    removed = [int(x) for x in replaced_samples]
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError("Expected ten distinct pose-selected cameras")
    if len(removed) != 2 or len(set(removed)) != 2:
        raise ValueError("Expected two distinct stereo replacement samples")
    to_remove = {f"{x:06d}.png" for x in removed}
    if not to_remove.issubset(names):
        raise ValueError("Stereo replacement is not part of pose-cover ten")
    kept = sorted(set(names) - to_remove)
    if len(kept) != 8:
        raise AssertionError("Eight-only measured depth budget violated")
    return kept


def self_test():
    ten = [f"{i:06d}.png" for i in range(10)]
    assert eight_only_names(ten, [2, 7]) == [
        f"{i:06d}.png" for i in range(10) if i not in (2, 7)]
    for invalid in ([2, 2], [2, 20], [2], []):
        try:
            eight_only_names(ten, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid removal: {invalid}")
    print("PASS: exact eight-measured control, no stereo or hidden depth")


def old_mesh_to_open3d(o3d, trimesh, path):
    """Load the already-produced 8+2 GLB for visualization only."""
    old = trimesh.load(path, force="mesh", process=False)
    if not isinstance(old, trimesh.Trimesh) or len(old.faces) == 0:
        raise RuntimeError("Existing 8+2 comparison GLB has no triangles")
    rgba = np.asarray(old.visual.vertex_colors)
    if rgba.shape[0] != len(old.vertices) or rgba.shape[1] < 3:
        raise RuntimeError("Existing 8+2 comparison GLB lacks vertex colors")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(old.vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(old.faces, dtype=np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        rgba[:, :3].astype(np.float64) / 255.0)
    return mesh


def run(upstream, stereo_artifact, output):
    import open3d as o3d
    import trimesh

    upstream = upstream.resolve()
    stereo_artifact = stereo_artifact.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Refuse to overwrite an existing experiment")
    output.mkdir(parents=True)
    start = time.time()
    receipt = dict(
        status="RUNNING",
        scene="Stanford Lounge",
        source_run=UPSTREAM_RUN,
        stereo_comparison_run="https://github.com/REDrighthand05/echo4d/actions/runs/40994901644",
        input_rgb=300, registered_rgb=287,
        estimated_poses=True, ground_truth_poses=False,
        measured_depth_maps=8, predicted_depth_maps=0,
        prior_mesh=False, gpu=False,
        metric_scale_source="four shared real sensor depth maps",
        comparison_mesh_is_prior=False,
        full_room_acceptance="FAIL",
        visual_acceptance="PENDING actual five-view comparison",
    )

    def save():
        (output / "receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        parent = json.loads((upstream / "receipt.json").read_text())
        ledger = json.loads((upstream / "inputs.json").read_text())
        prior = json.loads((stereo_artifact / "receipt.json").read_text())
        if (parent.get("selected_rgb"), parent.get("registered_frames"),
                parent.get("rgb_sampling")) != (300, 287, "early-overlap"):
            raise ValueError("Frozen RGB-COLMAP upstream does not match")
        if len(ledger) != 300 or prior.get("status") != "REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED":
            raise ValueError("Frozen RGB input ledger or stereo receipt missing")
        if (prior.get("selected_rgb"), prior.get("registered_rgb"),
                prior.get("intermediate_measured_depth_maps"),
                prior.get("intermediate_inferred_depth_maps")) != (300, 287, 8, 2):
            raise ValueError("Prior 8+2 experiment is not the expected input budget")
        views = read_model(upstream / parent["model"])
        if len(views) != 287:
            raise ValueError("Frozen COLMAP model does not have 287 poses")
        lookup = {int(Path(n).stem): n for n, _, _ in views}
        base30 = [lookup[int(i)] for i in parent["measured_depth_rgb_samples"]]
        base15 = base30[::2]
        base10 = [base15[int(i)] for i in np.linspace(0, 14, 10, dtype=int)]
        shared = [base10[i] for i in (0, 3, 6, 9)]

        # Download the original RGB-D dataset, not any reconstructed geometry.
        source = original_pairs(o3d.data.LoungeRGBDImages())
        rgb_paths, depth_paths = {}, {}
        for row in ledger:
            i = int(row["sample"])
            rgb, depth = source[int(row["original_index"])]
            if sha256(rgb) != row["rgb_sha256"]:
                raise ValueError(f"Changed original RGB at sample {i}")
            name = f"{i:06d}{rgb.suffix.lower()}"
            rgb_paths[name], depth_paths[name] = rgb, depth
        for name in shared:
            i = int(Path(name).stem)
            if sha256(depth_paths[name]) != ledger[i]["depth_sha256"]:
                raise ValueError(f"Changed shared measured scale anchor {i}")
        scale, witnesses = depth_scale(
            [v for v in views if v[0] in set(shared)],
            {n: depth_paths[n] for n in shared})
        if not np.isclose(scale, prior["metric_scale_m_per_colmap_unit"],
                          rtol=1e-7, atol=1e-10):
            raise ValueError("Measured scale differs from frozen stereo comparison")
        pose_names = select_pose_coverage(views, shared, 10, scale)
        eight = eight_only_names(
            pose_names, prior["intermediate_replaced_measured_samples"])
        if not set(shared).issubset(eight):
            raise ValueError("A shared metric scale anchor was removed")
        if sorted(int(Path(n).stem) for n in shared) != sorted(
                prior["common_measured_depth_samples"]):
            raise ValueError("Scale anchors differ from the previous experiment")
        for name in eight:
            i = int(Path(name).stem)
            if sha256(depth_paths[name]) != ledger[i]["depth_sha256"]:
                raise ValueError(f"Changed measured depth sample {i}")

        prior_glb = stereo_artifact / "measured8_stereo2.glb"
        if sha256(prior_glb) != prior["measured8_stereo2_sha256"]:
            raise ValueError("Previous 8+2 GLB checksum mismatch")
        receipt.update(
            measured_depth_samples=[int(Path(n).stem) for n in eight],
            stereo_replacement_samples=prior["intermediate_replaced_measured_samples"],
            shared_scale_samples=[int(Path(n).stem) for n in shared],
            metric_scale_m_per_colmap_unit=scale,
            metric_scale_witnesses=witnesses,
            comparison_glb_sha256=sha256(prior_glb))
        save()

        measured8 = fuse_hybrid(
            o3d, trimesh, views, eight, {}, scale, rgb_paths, depth_paths,
            output / "measured8_only.glb")
        stereo8plus2 = old_mesh_to_open3d(o3d, trimesh, prior_glb)
        recorded, rays = five_rays(o3d, views, scale)
        panel = Image.new("RGB", (960, 5 * 270 + 30), (18, 18, 18))
        draw = ImageDraw.Draw(panel)
        for col, label in enumerate(
                ("Original RGB", "8 measured ONLY", "8 measured + 2 RGB stereo")):
            draw.text((col * 320 + 6, 8), label, fill="white")
        coverage = {"measured8_only": [], "measured8_stereo2": []}
        for i, ray in enumerate(rays):
            y = 30 + i * 270
            label = (f"original RGB camera {Path(recorded[i][0]).stem}"
                     if i < 3 else f"free orbit {i - 2}")
            draw.text((6, y + 3), label, fill="white")
            if i < 3:
                with Image.open(rgb_paths[recorded[i][0]]) as photo:
                    panel.paste(photo.convert("RGB").resize((320, 240)),
                                (0, y + 26))
            for col, (key, mesh) in enumerate(
                    (("measured8_only", measured8),
                     ("measured8_stereo2", stereo8plus2)), 1):
                rendered, hit = render_rays(o3d, mesh, ray)
                panel.paste(rendered, (col * 320, y + 26))
                coverage[key].append(hit)
        png = "source-eight-only-vs-eight-plus-stereo-3plus2.png"
        panel.save(output / png)
        receipt.update(
            status="REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED",
            measured8_only_sha256=sha256(output / "measured8_only.glb"),
            comparison_glb_sha256=sha256(prior_glb),
            image=png,
            same_ray_hit_percent=coverage,
            recorded_samples=[int(Path(v[0]).stem) for v in recorded],
            visual_acceptance="PENDING human inspection; not a full-room PASS")
    except Exception as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = round(time.time() - start, 2)
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--stereo-artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not all((args.upstream, args.stereo_artifact, args.output)):
            parser.error("--upstream, --stereo-artifact and --output required")
        run(args.upstream, args.stereo_artifact, args.output)
