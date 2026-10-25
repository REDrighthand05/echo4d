"""TUM RGB-D adapter: explicit calibration, timestamp association, no extrapolation."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from .model import Camera, DataError, file_sha256, safe_path
from .output import output_transaction, write_json


def read_rows(path: Path, columns: int) -> list[list[str]]:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise DataError(f"Timestamp index too large: {path.name}")
    result, previous = [], -math.inf
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.partition("#")[0].strip().split()
        if not fields:
            continue
        if len(fields) != columns:
            raise DataError(f"{path.name}:{number}: expected {columns} columns")
        t = float(fields[0])
        if not math.isfinite(t) or t <= previous:
            raise DataError(f"{path.name}:{number}: timestamps must strictly increase")
        previous = t
        result.append(fields)
    if not result:
        raise DataError(f"Empty index: {path.name}")
    return result


def interpolate_pose(times: np.ndarray, translations: np.ndarray, rotations: Rotation,
                     t: float, max_gap: float) -> np.ndarray | None:
    high = int(np.searchsorted(times, t))
    if high < len(times) and abs(times[high] - t) < 1e-9:
        position, rotation = translations[high], rotations[high].as_matrix()
    elif high == 0 or high == len(times) or times[high] - times[high-1] > max_gap:
        return None
    else:
        low = high - 1
        alpha = (t-times[low]) / (times[high]-times[low])
        position = translations[low]*(1-alpha) + translations[high]*alpha
        rotation = Slerp(times[[low, high]], rotations[[low, high]])([t]).as_matrix()[0]
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, position
    return matrix


def nearest_timestamp(times: np.ndarray, t: float, max_gap: float) -> int | None:
    at = int(np.searchsorted(times, t))
    candidates = [i for i in (at-1, at) if 0 <= i < len(times)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: (abs(times[i]-t), i))
    return best if abs(times[best]-t) <= max_gap else None


def import_tum(args: argparse.Namespace) -> dict:
    if any(type(v) is not int or v < 1 for v in (args.stride, args.limit, args.downsample)):
        raise DataError("stride, limit, downsample must be positive integers")
    if any(not math.isfinite(v) or v <= 0 for v in (args.depth_scale, args.max_pose_gap, args.max_color_gap)):
        raise DataError("Depth scale and time gaps must be positive and finite")
    root = args.dataset.resolve()
    camera = Camera(*args.size, *args.intrinsics)
    small = camera.scaled(args.downsample)
    depth_index, rgb_index, pose_file = [safe_path(root, p) for p in ("depth.txt", "rgb.txt", args.poses)]
    depths, colors, rows = read_rows(depth_index, 2), read_rows(rgb_index, 2), read_rows(pose_file, 8)
    poses = np.array(rows, dtype=np.float64)
    if not np.isfinite(poses).all() or np.any(np.abs(np.linalg.norm(poses[:, 4:8], axis=1)-1) > .01):
        raise DataError("Poses must be finite with unit quaternions (tolerance .01)")
    rotations = Rotation.from_quat(poses[:, 4:8])
    times, translations = poses[:, 0], poses[:, 1:4]
    rgb_times = np.array([float(row[0]) for row in colors])
    world = np.eye(4)
    if args.world_up == "z":
        world[:3, :3] = [[1,0,0], [0,0,1], [0,-1,0]]
    associations, missing_pose, missing_color = [], 0, 0
    with output_transaction(args.output) as output:
        (output / "depth").mkdir()
        (output / "color").mkdir()
        entries, start_time = [], None
        for timestamp, relative_depth in depths[::args.stride]:
            if len(entries) >= args.limit:
                break
            t = float(timestamp)
            pose = interpolate_pose(times, translations, rotations, t, args.max_pose_gap)
            if pose is None:
                missing_pose += 1
                continue
            color_index = nearest_timestamp(rgb_times, t, args.max_color_gap)
            if color_index is None:
                missing_color += 1
                continue
            depth_path, color_path = safe_path(root, relative_depth), safe_path(root, colors[color_index][1])
            depth_hash, color_hash = file_sha256(depth_path), file_sha256(color_path)
            with Image.open(depth_path) as image:
                if image.size != tuple(args.size):
                    raise DataError("Depth resolution does not match --size")
                raw = np.array(image)
            if raw.ndim != 2 or raw.dtype.kind not in "ui":
                raise DataError("TUM depth must be a single-channel integer PNG")
            with Image.open(color_path) as image:
                if image.size != tuple(args.size):
                    raise DataError("Registered color must match depth resolution")
                rgb = np.array(image.convert("RGB"))
            if file_sha256(depth_path) != depth_hash or file_sha256(color_path) != color_hash:
                raise DataError("Input changed during conversion")
            depth = raw[::args.downsample, ::args.downsample].astype(np.float32) / args.depth_scale
            if not np.isfinite(depth).all() or np.any(depth > 100):
                raise DataError("Depth scale produced invalid or >100 metre values")
            i = len(entries)
            np.save(output / "depth" / f"{i:05d}.npy", depth, allow_pickle=False)
            Image.fromarray(rgb[::args.downsample, ::args.downsample]).save(output / "color" / f"{i:05d}.png")
            if start_time is None:
                start_time = t
            entries.append({"t": t-start_time, "depth": f"depth/{i:05d}.npy", "color": f"color/{i:05d}.png",
                            "cameraToWorld": (world @ pose).flatten(order="F").tolist()})
            associations.append({"index": i, "depth_timestamp": t, "rgb_timestamp": float(rgb_times[color_index]),
                "depth_path": relative_depth, "depth_sha256": depth_hash,
                "rgb_path": colors[color_index][1], "rgb_sha256": color_hash})
        if not entries:
            raise DataError("No synchronized samples; inspect timestamps and maximum time gaps")
        write_json(output / "recording.json", {"version": "echo-dataset-1", "source": "recorded", "units": "meters",
            "name": root.name, "duration": max(entries[-1]["t"], .001), "depth_scale": 1,
            "intrinsics": small.to_dict(), "frames": entries,
            "notes": "Known trajectory; SLERP rotation and linear translation. No SLAM performed."})
        write_json(output / "association.json", {"version": "echo-tum-association-1", "frames": associations,
            "pose_file": args.poses, "pose_sha256": file_sha256(pose_file),
            "depth_index_sha256": file_sha256(depth_index), "rgb_index_sha256": file_sha256(rgb_index),
            "world_up": args.world_up, "max_pose_gap_s": args.max_pose_gap, "max_color_gap_s": args.max_color_gap,
            "skipped_missing_pose": missing_pose, "skipped_missing_color": missing_color,
            "source_format": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"})
    return {"output": str(args.output.resolve()), "frames": len(entries),
            "skipped_missing_pose": missing_pose, "skipped_missing_color": missing_color}
