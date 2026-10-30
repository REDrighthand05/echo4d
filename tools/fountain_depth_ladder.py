"""Real Fountain information reduction: 33 -> 16 -> 8 measured RGB-D frames.

Reuse #179's Open3D Fountain importer, known keyframe poses, metric sensor
fusion, and five-camera renderer. The supplied scan is a separate, explicitly
labeled upper-reference PRIOR: never included in any reduced TSDF reconstruction.
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

from echo4d.model import DataError
from fountain_measured_prior_fusion import read_sample
from tum_xyz_open3d_320 import export_glb, free_view, fuse, panel, render


SOURCE_INDICES = (0, 16, 32)
FREE_INDICES = ((10, -1.), (23, 1.))
VOXEL_M = .015


def choose_fountain_indices(poses):
    """Choose nested 8/16 frame sets using known poses, never RGB/depth values.

    Eight time-spaced anchors preserve the trajectory's start/end and rotation.
    Eight further frames are chosen by farthest-point camera center + optical
    axis separation, not by sensor depth or visual reconstruction scores.
    """
    p = np.asarray(poses, dtype=np.float64)
    if p.shape != (33, 4, 4) or not np.isfinite(p).all():
        raise ValueError("Expected 33 finite known camera-to-world poses")
    if not np.allclose(p[:, 3, :], [0., 0., 0., 1.], atol=1e-5):
        raise ValueError("Invalid camera-to-world homogeneous row")
    centers = p[:, :3, 3]
    optical = p[:, :3, 2]
    norms = np.linalg.norm(optical, axis=1)
    if np.any(norms < .5):
        raise ValueError("Invalid camera optical axis")
    extent = np.linalg.norm(np.ptp(centers, axis=0))
    scale = max(.35, extent / 3.)
    feature = np.column_stack((centers / scale, optical / norms[:, None] * .8,
                               np.linspace(0., 1., len(p)) * .45))
    anchors = np.rint(np.linspace(0, 32, 8)).astype(int).tolist()
    chosen = anchors.copy()
    d2 = np.full(33, np.inf)
    for i in chosen:
        d2 = np.minimum(d2, np.sum((feature - feature[i]) ** 2, axis=1))
    d2[chosen] = -1.
    while len(chosen) < 16:
        i = int(np.argmax(d2))
        if d2[i] < 0:
            raise ValueError("Not enough distinct camera frames")
        chosen.append(i)
        d2 = np.minimum(d2, np.sum((feature - feature[i]) ** 2, axis=1))
        d2[chosen] = -1.
    return tuple(sorted(anchors)), tuple(sorted(chosen))


def run(work: Path, output: Path, receipt: dict):
    import open3d as o3d
    import trimesh

    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    # Official Open3D asset: original RGB/depth and poses, plus an existing scan.
    dataset = o3d.data.SampleFountainRGBDImages(data_root=str(work))
    frames, originals, camera, source_size = read_sample(o3d, dataset)
    eight, sixteen = choose_fountain_indices([f.camera_to_world for f in frames])
    if len(frames) != 33 or not set(eight).issubset(sixteen):
        raise DataError("Fountain source or nested cohorts changed")

    # Scan is an *existing* geometry reference, not an input to any TSDF below.
    prior = o3d.io.read_triangle_mesh(dataset.reconstruction_path)
    if len(prior.triangles) < 1000:
        raise DataError("Missing official pre-existing scan PRIOR")
    if len(prior.triangles) > 250_000:
        prior = prior.simplify_quadric_decimation(250_000)
    rgbds = []
    for rgb, depth in zip(dataset.color_paths, dataset.depth_paths):
        rgbds.append(o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(rgb), o3d.io.read_image(depth),
            depth_scale=1000., depth_trunc=4., convert_rgb_to_intensity=False))
    trajectory = o3d.io.read_pinhole_camera_trajectory(dataset.keyframe_poses_log_path)
    prior, _ = o3d.pipelines.color_map.run_rigid_optimizer(
        prior, rgbds, trajectory,
        o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=0))
    if not prior.has_vertex_colors():
        raise DataError("Official scan prior has no projected RGB")

    # Fixed five viewpoints, identical for all three actual reconstructed Meshes.
    view_poses = [frames[i].camera_to_world for i in SOURCE_INDICES]
    view_poses += [free_view(frames[i].camera_to_world, side)
                   for i, side in FREE_INDICES]
    rendered = {}
    receipt.update({
        "scene": "Open3D real Fountain RGB-D keyframe sequence",
        "source": "Open3D SampleFountainRGBDImages official release",
        "source_resolution": list(source_size),
        "fusion_resolution": [camera.width, camera.height],
        "known_poses": "Open3D provided keyframe camera poses; not RGB-estimated",
        "metric_scale": "measured sensor depth in metres; no inferred scale",
        "input_reduction": "33 to 16 to 8 real sensor depth frames; RGB source unchanged",
        "cohorts": {"measured33": list(range(33)),
                    "measured16": list(sixteen), "measured8": list(eight)},
        "scan_prior_in_reduced_meshes": False,
        "scan_prior_reference_only": "FULL existing Open3D scan, color projected from all 33 RGB frames",
        "predicted_depth": False,
        "gpu_inference": False,
        "voxel_m": VOXEL_M,
        "meshes": {},
        "views": [],
        "visual_verdict": "PENDING actual five-view inspection; full-room free orbit FAIL",
        "historical_fail": "#74 FAIL; Desk 16.7779cm >15cm FAIL",
    })
    for name, indices in receipt["cohorts"].items():
        mesh, _ = fuse(o3d, [frames[i] for i in indices], camera,
                       pixel_stride=1, voxel=VOXEL_M)
        filename = f"fountain-{name}.glb"
        export_glb(mesh, output / filename)
        loaded = trimesh.load(str(output / filename), force="mesh", process=False)
        if len(loaded.faces) < 1000 or loaded.visual.kind != "vertex":
            raise DataError(name + " exported GLB is not a colored Mesh")
        rendered[name] = [render(o3d, mesh, camera, pose) for pose in view_poses]
        receipt["meshes"][name] = {"glb": filename,
                                   "measured_depth_frames": len(indices),
                                   "triangles": len(mesh.triangles),
                                   "scan_prior": False}
        print(name, "measured triangles", len(mesh.triangles), flush=True)
        del mesh, loaded
        gc.collect()

    export_glb(prior, output / "fountain-FULL-SUPPLIED-PRIOR.glb")
    loaded = trimesh.load(str(output / "fountain-FULL-SUPPLIED-PRIOR.glb"),
                          force="mesh", process=False)
    if len(loaded.faces) < 1000 or loaded.visual.kind != "vertex":
        raise DataError("Existing scan PRIOR GLB is invalid")
    rendered["FULL_PRIOR"] = [render(o3d, prior, camera, pose)
                               for pose in view_poses]
    receipt["meshes"]["FULL_PRIOR"] = {
        "glb": "fountain-FULL-SUPPLIED-PRIOR.glb",
        "triangles": len(prior.triangles), "scan_prior": True,
        "measured_depth_frames_used_for_geometry": 0,
        "color_reference_rgb_frames": 33}

    boards = []
    for row, pose in enumerate(view_poses):
        if row < 3:
            i = SOURCE_INDICES[row]
            Image.fromarray(originals[i]).save(output / f"source-{i:02d}-native.png")
            photo = np.asarray(Image.fromarray(originals[i]).resize(
                (camera.width, camera.height), Image.Resampling.LANCZOS))
            kind = "recorded"
        else:
            i = FREE_INDICES[row - 3][0]
            photo = np.full((camera.height, camera.width, 3), 20, np.uint8)
            kind = "free"
        filename = f"{kind}-{i:02d}-depth-ladder.png"
        images = [photo] + [rendered[name][row] for name in
                            ("measured33", "measured16", "measured8", "FULL_PRIOR")]
        labels = [("REAL RGB" if kind == "recorded" else "FREE / NO PHOTO"),
                  "33 SENSOR", "16 SENSOR", "8 SENSOR", "FULL PRIOR / EXISTING"]
        boards.append(panel(output / filename, images, labels))
        receipt["views"].append({"kind": kind, "frame": i, "file": filename})
    board = Image.new("RGB", (camera.width * 5, (camera.height + 27) * 5))
    for row, image in enumerate(boards):
        board.paste(image, (0, row * (camera.height + 27)))
    board.save(output / "fountain-depth33-vs-16-vs-8-vs-PRIOR-five-views.png")
    receipt["status"] = "real_glbs_and_five_views_exported_visual_review_pending"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = {"status": "running", "run_url": (
        f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
        f"{os.environ['GITHUB_RUN_ID']}"
        if os.getenv("GITHUB_RUN_ID") else None)}
    try:
        run(args.work, args.output, receipt)
        code = 0
    except Exception as exc:
        traceback.print_exc()
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": receipt["status"],
                          "meshes": receipt.get("meshes"),
                          "error": receipt.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
