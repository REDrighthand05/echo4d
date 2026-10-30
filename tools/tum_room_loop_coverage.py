"""Real TUM Freiburg1 ROOM RGB-D Mesh: 291 measured frames with cross-view guard.

Both meshes use measured RGB/depth, provided reference poses, calibrated 320x240,
20mm TSDF. No existing scan prior, inferred depth, or geometry hallucination.
Five views require independent human inspection; #74 and old Desk remain FAIL.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import traceback
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from echo4d.model import DataError, load_recording
from echo4d.tum import nearest_timestamp, read_rows
from tum_desk_known_pose_open3d import FR1_INTRINSICS
from tum_public_roundtrip import invoke
from tum_xyz_open3d_320 import export_glb, free_view, fuse, panel, render

ROOT = "rgbd_dataset_freiburg1_room"
URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz"
HOSTS = {"cvg.cit.tum.de", "webshare.cvg.cit.tum.de", "vision.in.tum.de"}
STRIDE, MAX_FRAMES = 4, 440
VOXEL_M, MAX_FACES, MAX_ARCHIVE = .020, 2_500_000, 2 * 1024**3


def choose_views(count: int, train: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if count < 110 or len(train) < 90:
        raise DataError("Room loop needs at least 110 synchronized RGB-D views")
    def nearest(fraction: float) -> int:
        target = round((count - 1) * fraction)
        return min(train, key=lambda index: (abs(index - target), index))
    recorded = tuple(nearest(x) for x in (.10, .50, .85))
    free = tuple(nearest(x) for x in (.36, .70))
    if len(set(recorded + free)) != 5:
        raise DataError("Viewpoints not distinct")
    return recorded, free


def sparse_reference_indices(train: tuple[int, ...]) -> tuple[int, ...]:
    """Every other dense observation reproduces the prior stride-8 source set.

    Holdout is assigned on dense indices first. Since 2*i % 7 == 0 iff
    i % 7 == 0, the sparse subset matches the previous 145-frame run.
    """
    return tuple(i for i in train if i % 2 == 0)



# Reject only measured foreground contradicted by two other viewpoints.
def depth_neighbors(i, poses):
    here = poses[i][:3, 3]
    choices = [j for j in range(max(0, i-22), min(len(poses), i+23))
               if j != i and .07 <= np.linalg.norm(poses[j][:3, 3]-here) <= 1.]
    selected = []
    for side in (sorted((j for j in choices if j < i), reverse=True),
                 sorted(j for j in choices if j > i)):
        if side:
            selected += [side[k] for k in sorted(set(np.linspace(
                0, len(side)-1, min(3, len(side)), dtype=int)))]
    return sorted(selected)


def depth_local_min(depth):
    """3x3 nearest valid measured surface; zeros mean unknown."""
    d = np.where((depth > .2) & (depth < 4.), depth, np.inf).astype(np.float32)
    p = np.pad(d, 1, constant_values=np.inf)
    h, w = d.shape
    return np.minimum.reduce([p[y:y+h, x:x+w] for y in range(3) for x in range(3)])


def depth_conflict_mask(depth, pose, others, camera):
    """Keep occluded, unobserved, and corroborated surfaces; remove 2x conflicts."""
    h, w = depth.shape
    mask = np.zeros((h, w), bool)
    if len(others) < 2:
        return mask
    yy, xx = np.indices((h, w), dtype=np.float32)
    valid = np.flatnonzero(((depth >= .2) & (depth <= 4.)).ravel())
    if not len(valid):
        return mask
    z = depth.ravel()[valid]
    xyz = np.stack(((xx.ravel()[valid]-camera.cx)*z/camera.fx,
                    (yy.ravel()[valid]-camera.cy)*z/camera.fy, z))
    world = pose[:3, :3] @ xyz + pose[:3, 3, None]
    farther = np.zeros(len(valid), np.uint8)
    agree = np.zeros(len(valid), np.uint8)
    for other_pose, other_depth in others:
        if other_depth.shape != (h, w):
            raise DataError("Measured neighbor depth dimensions differ")
        local = other_pose[:3, :3].T @ (world-other_pose[:3, 3, None])
        zz = local[2]
        safe = np.where(zz > .2, zz, 1.)
        u = np.rint(camera.fx*local[0]/safe+camera.cx).astype(np.int32)
        v = np.rint(camera.fy*local[1]/safe+camera.cy).astype(np.int32)
        ok = np.flatnonzero((zz > .2) & (zz <= 4.) &
                            (u >= 0) & (u < w) & (v >= 0) & (v < h))
        if not len(ok):
            continue
        measured = other_depth[v[ok], u[ok]]
        known = np.isfinite(measured)
        farther[ok] += (known & (measured > zz[ok]+.12)).astype(np.uint8)
        agree[ok] += (known & (np.abs(measured-zz[ok]) <= .055)).astype(np.uint8)
    mask.ravel()[valid] = (farther >= 2) & (agree == 0)
    return mask


def guard_depth_frames(frames, camera):
    """Simultaneous filtering against original real depth, never cascading."""
    from dataclasses import replace
    poses = [np.asarray(f.camera_to_world, dtype=np.float64) for f in frames]
    mins = [depth_local_min(f.depth) for f in frames]
    guarded, removed, neighbors = [], [], []
    for i, f in enumerate(frames):
        indices = depth_neighbors(i, poses)
        mask = depth_conflict_mask(f.depth, poses[i],
                                  [(poses[j], mins[j]) for j in indices], camera)
        new_depth = np.array(f.depth, copy=True)
        new_depth[mask] = 0.
        guarded.append(replace(f, depth=new_depth))
        removed.append(int(mask.sum()))
        neighbors.append(len(indices))
        if (i+1) % 40 == 0:
            print(f"Checked {i+1}/{len(frames)} measured views", flush=True)
    return guarded, removed, neighbors


def safe_member(member: tarfile.TarInfo) -> tuple[str, ...]:
    path = PurePosixPath(member.name)
    if (path.is_absolute() or ".." in path.parts or not path.parts or
            path.parts[0] != ROOT or not (member.isfile() or member.isdir()) or
            member.size < 0 or member.size > 16 * 1024**2):
        raise DataError("Unsafe or oversized TUM member: " + member.name[:120])
    return path.parts


def download(path: Path, result: dict, offline: Path | None):
    if offline is not None:
        if not offline.is_file() or offline.stat().st_size > MAX_ARCHIVE:
            raise DataError("Offline archive absent/oversized")
        shutil.copyfile(offline, path)
    else:
        request = Request(URL, headers={"User-Agent": "echo4d-room-coverage/1.0"})
        with urlopen(request, timeout=120) as response:
            url = urlsplit(response.geturl())
            if url.scheme != "https" or url.hostname not in HOSTS:
                raise DataError("Dataset redirected outside official TUM HTTPS")
            size = 0
            with path.open("xb") as out:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_ARCHIVE:
                        raise DataError("TUM room archive exceeds resource limit")
                    out.write(chunk)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result["archive"] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest(),
                         "official_source": URL, "offline_supplied": offline is not None}


def extract_sampled(archive: Path, root: Path) -> dict:
    """Read indices first, then extract ONLY real RGB/depth PNGs selected by time."""
    root.mkdir(parents=True, exist_ok=False)
    indexes: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            parts = safe_member(member)
            if member.isfile() and len(parts) == 2 and parts[1] in (
                    "depth.txt", "rgb.txt", "groundtruth.txt"):
                if parts[1] in indexes or member.size > 8 * 1024**2:
                    raise DataError("Duplicate/oversized TUM index")
                payload = source.extractfile(member)
                if payload is None:
                    raise DataError("Cannot read TUM index")
                indexes[parts[1]] = payload.read()
    if set(indexes) != {"depth.txt", "rgb.txt", "groundtruth.txt"}:
        raise DataError("Missing RGB/depth/groundtruth indices")
    for name, data in indexes.items():
        (root / name).write_bytes(data)
    depths = read_rows(root / "depth.txt", 2)
    colors = read_rows(root / "rgb.txt", 2)
    rgb_times = np.array([float(row[0]) for row in colors])
    chosen_depth = []
    chosen_color = {}
    for row in depths[::STRIDE][:MAX_FRAMES]:
        match = nearest_timestamp(rgb_times, float(row[0]), .03)
        if match is None:
            continue
        chosen_depth.append(row)
        chosen_color[colors[match][1]] = colors[match]
    if len(chosen_depth) < 110:
        raise DataError("Room has insufficient synchronized depth observations")
    required = {row[1] for row in chosen_depth} | set(chosen_color)
    if len(required) != len(chosen_depth) + len(chosen_color):
        raise DataError("RGB/depth archive path collision")
    extracted = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            parts = safe_member(member)
            rel = "/".join(parts[1:])
            if not member.isfile() or rel not in required:
                continue
            if rel in extracted:
                raise DataError("Duplicate measured image in archive")
            dest = root.joinpath(*parts[1:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            payload = source.extractfile(member)
            if payload is None:
                raise DataError("Cannot read TUM measured image")
            with payload, dest.open("xb") as out:
                shutil.copyfileobj(payload, out)
            extracted.add(rel)
    if extracted != required:
        raise DataError(f"Missing {len(required - extracted)} selected measured images")
    (root / "depth.txt").write_text("".join(" ".join(row) + "\n" for row in chosen_depth))
    sorted_rgb = sorted(chosen_color.values(), key=lambda row: float(row[0]))
    (root / "rgb.txt").write_text("".join(" ".join(row) + "\n" for row in sorted_rgb))
    return {"selected_depth_frames": len(chosen_depth),
            "selected_rgb_frames": len(sorted_rgb), "extracted_images": len(extracted),
            "original_rgb_depth": "640x480 measured sensor PNG", "stride": STRIDE}


def run(args, result):
    import open3d as o3d
    args.work.mkdir(parents=True, exist_ok=False)
    archive = args.work / "room-source.tgz"
    download(archive, result, args.archive)
    root = args.work / "dataset" / ROOT
    result["extraction"] = extract_sampled(archive, root)
    imported = args.work / "imported-room-320"
    invoke("import-TUM-room-measured-RGBD-and-reference-poses", [
        sys.executable, "-m", "echo4d", "import-tum", str(root),
        "--output", str(imported), "--intrinsics", *map(str, FR1_INTRINSICS),
        "--stride", "1", "--limit", str(MAX_FRAMES), "--downsample", "2",
        "--world-up", "z"], result, timeout=1200)
    recording = load_recording(imported / "recording.json")
    camera = recording.camera
    if (camera.width, camera.height) != (320, 240):
        raise DataError("Expected calibrated 320x240 room recording")
    entries = list(recording.entries)
    train = tuple(i for i in range(len(entries)) if i % 7 != 0)
    holdout = tuple(i for i in range(len(entries)) if i % 7 == 0)
    sparse = sparse_reference_indices(train)
    recorded, free = choose_views(len(entries), sparse)
    recording.entries = [entries[i] for i in train]
    frames = [frame for _, frame, _ in recording.iter_frames()]
    if len(frames) != len(train) or len(sparse) < 90:
        raise DataError("Measured frames lost before fusion")
    args.output.mkdir(parents=True, exist_ok=True)
    result.update({"scene": "TUM freiburg1_room office full loop",
                   "geometry_source": "measured RGB-D + TUM reference poses; NO scan prior",
                   "camera": camera.to_dict(), "training": list(train),
                   "holdout": list(holdout), "recorded_views": list(recorded),
                   "free_views": list(free), "sparse_frames": len(sparse), "sparse_indices": list(sparse),
                   "dense_frames": len(frames), "voxel_m": VOXEL_M,
                   "visual_verdict": "PENDING human inspection; no free-orbit acceptance"})
    poses = {i: np.asarray(entries[i]["cameraToWorld"], dtype=np.float64).reshape(
        (4, 4), order="F") for i in set(recorded + free)}
    rendered = {}
    result["meshes"] = {}
    frame_by_index = dict(zip(train, frames))
    guarded, removed, neighbors = guard_depth_frames(frames, camera)
    result["crossview_guard"] = {"removed_pixels": sum(removed),
                                 "frames_changed": sum(x > 0 for x in removed),
                                 "per_frame_removed": removed, "neighbor_counts": neighbors,
                                 "rule": "two measured views >12cm farther, no agreement within 5.5cm"}
    for label, cohort in (("dense", frames), ("guarded", guarded)):
        mesh, _ = fuse(o3d, cohort, camera, pixel_stride=1, voxel=VOXEL_M)
        if len(mesh.triangles) > MAX_FACES:
            mesh = mesh.simplify_quadric_decimation(MAX_FACES)
            mesh.compute_vertex_normals()
        if not 1000 < len(mesh.triangles) <= MAX_FACES:
            raise DataError(f"{label}: invalid mesh size")
        filename = f"room-measured-{label}-320.glb"
        export_glb(mesh, args.output / filename)
        result["meshes"][label] = {"glb": filename, "triangles": len(mesh.triangles),
                                   "frames": len(cohort), "prior_mesh": False}
        rendered[label] = [render(o3d, mesh, camera, poses[i]) for i in recorded]
        rendered[label] += [render(o3d, mesh, camera, free_view(poses[i], side))
                            for side, i in zip((-1., 1.), free)]
        print(f"Real TUM room {label} Mesh: {len(mesh.triangles)} triangles", flush=True)
        del mesh
        gc.collect()
    association = json.loads((imported / "association.json").read_text())
    panels = []
    result["views"] = []
    for n, index in enumerate(recorded):
        with Image.open(root / association["frames"][index]["rgb_path"]) as img:
            photo = img.convert("RGB")
            photo.save(args.output / f"real-room-{index:03d}-640x480.png")
            original = np.asarray(photo.resize((320, 240), Image.Resampling.LANCZOS))
        filename = f"recorded-{index:03d}-dense-vs-guarded.png"
        panels.append(panel(args.output / filename, [original, rendered["dense"][n],
                            rendered["guarded"][n]],
                            ["REAL RGB", f"BASE / {len(frames)}", f"GUARDED / {len(frames)}"]))
        result["views"].append({"kind": "recorded", "index": index, "file": filename})
    for j, index in enumerate(free):
        n = len(recorded) + j
        filename = f"free-{j+1}-dense-vs-guarded.png"
        blank = np.full_like(rendered["dense"][n], 20)
        panels.append(panel(args.output / filename,
                            [blank, rendered["dense"][n], rendered["guarded"][n]],
                            ["FREE / no photo", f"BASE / {len(frames)}", f"GUARDED / {len(frames)}"]))
        result["views"].append({"kind": "free", "index": index, "file": filename,
                                "shift_m": .22, "yaw_deg": 12 * (-1 if j == 0 else 1)})
    board = Image.new("RGB", (960, 267 * len(panels)))
    for n, item in enumerate(panels):
        board.paste(item, (0, 267*n))
    board.save(args.output / "room-five-view-dense-vs-guarded.png")
    result["issue_74"] = "FAIL unchanged"
    result["historical_desk"] = "FAIL 16.7779cm > 15cm unchanged"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--archive", type=Path)
    args = p.parse_args()
    result = {"status": "running", "source": URL, "commands": [],
              "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                          f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None)}
    try:
        run(args, result)
        result["status"] = "real_room_meshes_and_views_exported_visual_review_pending"
        code = 0
    except Exception as exc:
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": result["status"], "meshes": result.get("meshes"),
                          "error": result.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
