"""Real TUM Freiburg1 XYZ: measured-depth Open3D 320x240 edge-guard A/B.
Known TUM reference poses; no supplied mesh, no RGB-only reconstruction.
48 fused frames; 12 excluded (indices divisible by five). Existing FAIL gates unchanged.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw

from echo4d.model import Camera, DataError, load_recording
from tum_public_roundtrip import ROOT, URL, download, extract, invoke

FRAME_COUNT = 60
TRAIN_INDICES = tuple(i for i in range(FRAME_COUNT) if i % 5)
RECORDED_INDICES = (7, 29, 47)
MAX_DEPTH = 4.0


def half_resolution(depth: np.ndarray, color: np.ndarray):
    """Preserve original even-indexed pixel centres and calibrated pinhole rays."""
    if depth.ndim != 2 or color.shape != (*depth.shape, 3):
        raise ValueError("RGB-D dimensions differ")
    return (np.ascontiguousarray(depth[::2, ::2]),
            np.ascontiguousarray(color[::2, ::2]))


def camera_matrix(camera: Camera) -> np.ndarray:
    return np.array([[camera.fx, 0., camera.cx],
                     [0., camera.fy, camera.cy],
                     [0., 0., 1.]], dtype=np.float64)


def free_view(camera_to_world: np.ndarray, side: float) -> np.ndarray:
    """Translate 22 cm sideways and rotate 12 degrees about camera local Y."""
    if side not in (-1., 1.):
        raise ValueError("Side must be -1 or +1")
    angle = side * np.deg2rad(12.)
    c, s = np.cos(angle), np.sin(angle)
    yaw = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    pose = np.array(camera_to_world, dtype=np.float64, copy=True)
    pose[:3, 3] += side * .22 * pose[:3, 0]
    pose[:3, :3] = pose[:3, :3] @ yaw
    return pose


def suppress_flying_depth(depth: np.ndarray) -> tuple[np.ndarray, int]:
    """Reject intermediate depth between two opposite, valid measurements.

    Keep near foreground and far background endpoints of real silhouettes.
    Only reject pixels whose left/right OR up/down neighbors straddle them
    by at least 5cm each with a 16cm total jump. Never use invalid zeros as
    evidence. This removes potential Kinect flying pixels, not entire edges.
    """
    if depth.ndim != 2:
        raise ValueError("Expected 2D metric depth")
    d = np.asarray(depth, dtype=np.float32)
    p = np.pad(d, 1, mode="constant", constant_values=0.)
    left, right = p[1:-1, :-2], p[1:-1, 2:]
    up, down = p[:-2, 1:-1], p[2:, 1:-1]

    def straddles(a, b):
        near, far = np.minimum(a, b), np.maximum(a, b)
        return ((a > 0) & (b > 0) & (far - near >= .16) &
                (d - near >= .05) & (far - d >= .05))

    rejected = (d > 0) & (straddles(left, right) | straddles(up, down))
    cleaned = d.copy()
    cleaned[rejected] = 0.
    return np.ascontiguousarray(cleaned), int(np.count_nonzero(rejected))


def fuse(o3d, frames, camera: Camera, *, pixel_stride: int, voxel: float,
         edge_guard: bool = False):
    if pixel_stride not in (1, 2) or voxel <= 0:
        raise ValueError("Invalid fusion configuration")
    cam = camera if pixel_stride == 1 else camera.scaled(2)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * 3.5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    removed = 0
    for i, frame in enumerate(frames):
        if pixel_stride == 1:
            depth, color = frame.depth, frame.color
        else:
            depth, color = half_resolution(frame.depth, frame.color)
        if color is None or depth.shape != (cam.height, cam.width):
            raise DataError("Missing or uncalibrated RGB-D input")
        depth = np.ascontiguousarray(depth.astype(np.float32))
        depth[(depth < .2) | (depth > MAX_DEPTH)] = 0
        if edge_guard:
            depth, rejected = suppress_flying_depth(depth)
            removed += rejected
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(depth), depth_scale=1.,
            depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(frame.camera_to_world))
        if (i + 1) % 12 == 0:
            print(f"Integrated {i+1}/{len(frames)} measured RGB-D frames "
                  f"at {cam.width}x{cam.height}", flush=True)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) < 100:
        raise RuntimeError("TSDF produced insufficient triangles")
    return mesh, removed


def export_glb(mesh, path: Path):
    import trimesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if len(colors) != len(vertices):
        raise RuntimeError("Extracted mesh is missing vertex RGB")
    colors = np.round(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    scene = trimesh.Trimesh(vertices=vertices, faces=triangles,
                            vertex_colors=colors, process=False)
    scene.export(str(path))


def render(o3d, mesh, camera: Camera, pose: np.ndarray) -> np.ndarray:
    """Real ray-traced mesh color; no fake camera RGB pasted into the geometry."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(camera_matrix(camera).astype(np.float32)),
        o3d.core.Tensor(np.linalg.inv(pose).astype(np.float32)),
        camera.width, camera.height)
    result = scene.cast_rays(rays)
    ids = result["primitive_ids"].numpy().reshape(-1)
    uv = result["primitive_uvs"].numpy().reshape(-1, 2)
    distances = result["t_hit"].numpy().reshape(-1)
    valid = np.isfinite(distances) & (ids < len(mesh.triangles))
    output = np.full((camera.width * camera.height, 3), 18, np.uint8)
    if np.any(valid):
        triangles = np.asarray(mesh.triangles)[ids[valid]]
        colors = np.asarray(mesh.vertex_colors)[triangles]
        u, v = uv[valid, 0, None], uv[valid, 1, None]
        interpolated = ((1 - u - v) * colors[:, 0] +
                        u * colors[:, 1] + v * colors[:, 2])
        output[valid] = np.round(np.clip(interpolated, 0, 1) * 255).astype(np.uint8)
    del scene
    return output.reshape(camera.height, camera.width, 3)


def panel(path: Path, images: list[np.ndarray], captions: list[str]):
    if len(images) != len(captions):
        raise ValueError("Panel columns mismatch")
    height, width = images[0].shape[:2]
    canvas = Image.new("RGB", (width * len(images), height + 27), (20, 20, 20))
    pen = ImageDraw.Draw(canvas)
    for i, (array, caption) in enumerate(zip(images, captions)):
        if array.shape != images[0].shape:
            raise ValueError("Panel dimensions mismatch")
        canvas.paste(Image.fromarray(array), (width * i, 27))
        pen.text((width * i + 7, 7), caption, fill=(245, 245, 245))
    canvas.save(path)
    return canvas


def experiment(args, receipt):
    import open3d as o3d
    args.work.mkdir(parents=True, exist_ok=False)
    archive = args.work / "source.tgz"
    if args.archive:
        from echo4d.model import file_sha256
        from tum_public_roundtrip import PINNED_ARCHIVE_SHA256, PINNED_ARCHIVE_BYTES
        if (args.archive.stat().st_size != PINNED_ARCHIVE_BYTES or
                file_sha256(args.archive) != PINNED_ARCHIVE_SHA256):
            raise DataError("Offline TUM archive does not match pinned official bytes")
        shutil.copyfile(args.archive, archive)
    else:
        download(archive, receipt)
    extract(archive, args.work / "dataset")
    imported = args.work / "imported-320"
    invoke("import-TUM-320x240-60", [
        sys.executable, "-m", "echo4d", "import-tum",
        str(args.work / "dataset" / ROOT), "--output", str(imported),
        "--intrinsics", "525", "525", "319.5", "239.5",
        "--stride", "12", "--limit", "60", "--downsample", "2",
        "--world-up", "z"], receipt, timeout=900)
    record = load_recording(imported / "recording.json")
    if len(record) != FRAME_COUNT or (record.camera.width, record.camera.height) != (320, 240):
        raise DataError("Expected 60 calibrated 320x240 frames")
    full_entries = record.entries
    record.entries = [full_entries[i] for i in TRAIN_INDICES]
    frames = list(frame for _, frame, _ in record.iter_frames())
    if len(frames) != 48:
        raise DataError("Expected 48 real depth frames and 12 untouched heldouts")
    # Identical 48 measured frames, calibrated rays, 1.8cm voxels and poses.
    # The edge guard is the ONLY changed geometry input in this A/B.
    old, _ = fuse(o3d, frames, record.camera, pixel_stride=1, voxel=.018)
    new, rejected = fuse(o3d, frames, record.camera, pixel_stride=1,
                         voxel=.018, edge_guard=True)
    print(f"Edge guard removed {rejected} intermediate depth samples", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    export_glb(old, args.output / "measured-320-unguarded.glb")
    export_glb(new, args.output / "measured-320-edge-guard.glb")
    print("Exported actual colored meshes", flush=True)
    views = []
    canvases = []
    original_root = args.work / "dataset" / ROOT
    association = json.loads((imported / "association.json").read_text())
    for i in RECORDED_INDICES:
        if i not in TRAIN_INDICES:
            raise DataError("Recorded camera must be among fused training frames")
        spec = full_entries[i]
        pose = np.asarray(spec["cameraToWorld"]).reshape(4, 4, order="F")
        with Image.open(original_root / association["frames"][i]["rgb_path"]) as img:
            photo_native = img.convert("RGB").copy()
        photo_native.save(args.output / f"real-original-{i:02d}-640x480.png")
        photo = np.asarray(photo_native.resize((320, 240), Image.Resampling.LANCZOS))
        old_rgb = render(o3d, old, record.camera, pose)
        new_rgb = render(o3d, new, record.camera, pose)
        name = f"recorded-{i:02d}-comparison.png"
        canvases.append(panel(args.output / name,
                              [photo, old_rgb, new_rgb],
                              ["REAL RGB 640 source", "320 unguarded", "320 edge guarded"]))
        views.append({"kind": "recorded", "index": i, "file": name})
        gc.collect()
    for side, index in ((-1., 19), (1., 39)):
        pose = free_view(np.asarray(full_entries[index]["cameraToWorld"]).reshape(
            4, 4, order="F"), side)
        old_rgb = render(o3d, old, record.camera, pose)
        new_rgb = render(o3d, new, record.camera, pose)
        blank = np.full_like(new_rgb, 20)
        name = f"free-{'left' if side < 0 else 'right'}-comparison.png"
        canvases.append(panel(args.output / name,
                              [blank, old_rgb, new_rgb],
                              ["FREE CAMERA / NO RGB", "320 unguarded", "320 edge guarded"]))
        views.append({"kind": "free", "reference_index": index, "file": name,
                      "camera_shift_m": .22, "camera_yaw_deg": 12 * side})
        gc.collect()
    board = Image.new("RGB", (960, 267 * len(canvases)))
    for i, canvas in enumerate(canvases):
        board.paste(canvas, (0, 267 * i))
    board.save(args.output / "five-view-measured-320-edge-guard-ab.png")
    receipt["meshes"] = {
        "320x240_unguarded": {"triangles": len(old.triangles),
                              "vertices": len(old.vertices), "glb": "measured-320-unguarded.glb"},
        "320x240_edge_guard": {"triangles": len(new.triangles),
                               "vertices": len(new.vertices), "glb": "measured-320-edge-guard.glb"}}
    receipt["edge_guard_removed_samples"] = rejected
    receipt["edge_guard_policy"] = "opposite-neighbor depth straddle, preserve near/far"
    receipt["views"] = views
    receipt["geometry_source"] = "TUM measured RGB-D + provided reference poses only; no scan prior"
    receipt["visual_verdict"] = "PENDING human review; counts do not imply quality"
    receipt["free_orbit"] = "FAIL until human validated"
    receipt["issue_74"] = "FAIL unchanged"
    receipt["old_desk"] = "FAIL 16.7779cm > 15cm unchanged"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    args = p.parse_args()
    receipt = {"scene": "TUM Freiburg1 XYZ real sensor scene",
               "source": URL, "status": "running",
               "inputs": "640x480 original RGB-D imported at 320x240; known TUM poses",
               "train_indices": list(TRAIN_INDICES),
               "holdout_indices": list(range(0, FRAME_COUNT, 5)),
               "commands": [], "run_url": (
                   f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                   f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None)}
    try:
        experiment(args, receipt)
        receipt["status"] = "mesh_and_five_views_exported_visual_review_pending"
        code = 0
    except Exception as exc:
        code = 1
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
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
