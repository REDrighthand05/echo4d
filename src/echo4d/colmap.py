"""Adapt undistorted COLMAP text poses and separately registered metric Z-depth.

This is an adapter, NOT a video pose/depth estimator. COLMAP's monocular
gauge needs an independently measured world scale.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path, PurePosixPath
import shutil

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .model import Camera, DataError, load_recording, safe_path, file_sha256
from .output import output_transaction, write_json


def _lines(path: Path, limit_bytes: int = 32 * 1024 * 1024) -> list[str]:
    if not path.is_file() or path.stat().st_size > limit_bytes:
        raise DataError(f"Missing or oversized COLMAP file: {path}")
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeError as error:
        raise DataError(f"COLMAP file is not UTF-8: {path}") from error


def _frame_name(value: str) -> str:
    name = PurePosixPath(value)
    if (not value or "\\" in value or name.is_absolute() or
            any(part in ("", ".", "..") for part in value.split("/")) or
            name.suffix.lower() not in (".png", ".jpg", ".jpeg")):
        raise DataError(f"Unsafe or unsupported COLMAP image name: {value}")
    return str(name)


def _cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _lines(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise DataError("Invalid COLMAP cameras.txt entry")
        try:
            camera_id, kind = int(fields[0]), fields[1]
            width, height = int(fields[2]), int(fields[3])
            params = [float(p) for p in fields[4:]]
            if kind == "PINHOLE" and len(params) == 4:
                fx, fy, cx, cy = params
            elif kind == "SIMPLE_PINHOLE" and len(params) == 3:
                f, cx, cy = params
                fx = fy = f
            else:
                raise DataError("Only undistorted PINHOLE / SIMPLE_PINHOLE COLMAP cameras are supported")
            if camera_id <= 0 or camera_id in cameras:
                raise DataError("Duplicate or invalid COLMAP camera ID")
            cameras[camera_id] = Camera(width, height, fx, fy, cx, cy)
        except DataError:
            raise
        except (ValueError, TypeError, OverflowError) as error:
            raise DataError(f"Invalid COLMAP camera entry: {line[:120]}") from error
    if not cameras:
        raise DataError("No usable COLMAP cameras")
    return cameras


def _validate_points2d(line: str, line_number: int) -> None:
    """Check the paired row before consuming it; an empty observation row is valid."""
    fields = line.split()
    if len(fields) % 3:
        raise DataError(f"Invalid COLMAP POINTS2D row at line {line_number}: expected x y point3D_id triples")
    for index in range(0, len(fields), 3):
        try:
            x, y = float(fields[index]), float(fields[index + 1])
            point_id = int(fields[index + 2])
        except ValueError as error:
            raise DataError(f"Invalid COLMAP POINTS2D values at line {line_number}") from error
        if not math.isfinite(x) or not math.isfinite(y) or point_id < -1:
            raise DataError(f"Invalid COLMAP POINTS2D values at line {line_number}")


def _images(path: Path) -> list[tuple[str, int, np.ndarray]]:
    # A pose line is followed by one POINTS2D line, possibly empty.
    lines = _lines(path)
    result: list[tuple[str, int, np.ndarray]] = []
    names, ids = set(), set()
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor].strip()
        cursor += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise DataError("Invalid COLMAP image pose entry")
        if cursor >= len(lines):
            raise DataError("COLMAP image pose is missing its POINTS2D line")
        _validate_points2d(lines[cursor], cursor + 1)
        cursor += 1
        try:
            image_id = int(fields[0])
            q = np.array([float(v) for v in fields[1:5]], dtype=np.float64)
            translation = np.array([float(v) for v in fields[5:8]], dtype=np.float64)
            camera_id = int(fields[8])
            name = _frame_name(fields[9])
            if (image_id <= 0 or image_id in ids or name in names or
                    not np.isfinite(q).all() or not np.isfinite(translation).all() or
                    abs(np.linalg.norm(q) - 1) > .01):
                raise DataError(f"Invalid/duplicate COLMAP image pose: {name}")
            r_wc = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = r_wc.T
            pose[:3, 3] = -(r_wc.T @ translation)
        except DataError:
            raise
        except (ValueError, TypeError) as error:
            raise DataError(f"Invalid COLMAP pose line: {line[:120]}") from error
        ids.add(image_id)
        names.add(name)
        result.append((name, camera_id, pose))
        if len(result) > 100000:
            raise DataError("Too many COLMAP frames")
    if not result:
        raise DataError("No COLMAP poses")
    return result


def _timestamps(path: Path) -> dict[str, float]:
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise DataError("Missing or oversized image timestamp CSV")
    result: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["name", "t"]:
            raise DataError("Timestamp CSV header must be name,t (seconds)")
        for row in reader:
            try:
                name = _frame_name(row["name"])
                t = float(row["t"])
            except (ValueError, TypeError, KeyError) as error:
                raise DataError("Invalid timestamp CSV record") from error
            if name in result or not math.isfinite(t) or t < 0:
                raise DataError("Duplicate or invalid image timestamp")
            result[name] = t
    if not result:
        raise DataError("Timestamp CSV is empty")
    return result


def import_colmap(args: argparse.Namespace) -> dict:
    """Publish an echo-dataset-1 recording from existing pose and depth inputs."""
    scale = args.meters_per_colmap_unit
    if not math.isfinite(scale) or scale <= 0 or scale > 10000:
        raise DataError("meters-per-colmap-unit must be a finite positive measured scale")
    if not isinstance(args.scale_source, str) or not args.scale_source.strip():
        raise DataError("--scale-source must identify the external real-world scale measurement")
    model_dir, image_dir, depth_dir = (Path(p).resolve() for p in
                                        (args.model, args.images, args.depth))
    cameras = _cameras(model_dir / "cameras.txt")
    images = _images(model_dir / "images.txt")
    times = _timestamps(Path(args.timestamps).resolve())
    names = {entry[0] for entry in images}
    if names != set(times):
        raise DataError("Timestamp CSV must contain exactly the registered COLMAP images")
    used = {entry[1] for entry in images}
    if len(used) != 1 or not used.issubset(cameras):
        raise DataError("echo-dataset-1 requires a single known COLMAP camera model")
    camera = cameras[next(iter(used))]
    frames = sorted(images, key=lambda entry: times[entry[0]])
    if any(times[frames[i][0]] >= times[frames[i + 1][0]] for i in range(len(frames) - 1)):
        raise DataError("All frame timestamps must be unique")
    t0 = times[frames[0][0]]

    # Validate first; output_transaction publishes all or none.
    files = []
    depth_names = set()
    for name, _, pose in frames:
        color_file = safe_path(image_dir, name)
        depth_name = str(PurePosixPath(name).with_suffix(".npy"))
        if depth_name in depth_names:
            raise DataError("Multiple images resolve to the same depth filename")
        depth_names.add(depth_name)
        depth_file = safe_path(depth_dir, depth_name)
        with Image.open(color_file) as image:
            if image.size != (camera.width, camera.height):
                raise DataError(f"Image size differs from COLMAP camera: {name}")
        depth = np.load(depth_file, allow_pickle=False, mmap_mode="r")
        if not isinstance(depth, np.ndarray):
            if isinstance(depth, np.lib.npyio.NpzFile):
                depth.close()
            raise DataError(f"Depth must be a single numeric NPY array, not an archive: {depth_name}")
        if depth.shape != (camera.height, camera.width) or depth.dtype.kind != "f":
            raise DataError(f"Depth must be floating-point metric Z, camera-sized NPY: {depth_name}")
        if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(depth > 100):
            raise DataError(f"Invalid metric depth values: {depth_name}")
        pose[:3, 3] *= scale
        if not np.isfinite(pose).all() or np.max(np.abs(pose[:3, 3])) > 10000:
            raise DataError("Scaled COLMAP pose is outside echo4d's coordinate range")
        files.append((name, depth_name, color_file, depth_file, pose))

    with output_transaction(Path(args.output)) as output:
        entries = []
        for name, depth_name, color_file, depth_file, pose in files:
            out_color = output / "color" / name
            out_depth = output / "depth" / depth_name
            out_color.parent.mkdir(parents=True, exist_ok=True)
            out_depth.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(color_file, out_color)
            shutil.copyfile(depth_file, out_depth)
            entries.append({"t": times[name] - t0,
                            "cameraToWorld": pose.flatten(order="F").tolist(),
                            "color": "color/" + name,
                            "depth": "depth/" + depth_name})
        duration = max(entries[-1]["t"], .001)
        write_json(output / "recording.json", {
            "version": "echo-dataset-1", "name": "colmap-registered-depth",
            "source": "recorded", "units": "meters", "depth_scale": 1,
            "duration": duration, "intrinsics": camera.to_dict(), "frames": entries,
            "notes": "Imported poses and externally supplied metric depth. No pose or depth estimation performed."})
        write_json(output / "import-provenance.json", {
            "version": "echo-colmap-import-1", "source_format": "COLMAP text",
            "scale_meters_per_colmap_unit": scale, "scale_source": args.scale_source.strip(),
            "pose_source": "COLMAP images.txt (world-to-camera inverted)",
            "depth_source": "externally supplied registered metric Z depth",
            "model_sha256": {p: file_sha256(model_dir / p) for p in ("cameras.txt", "images.txt")},
            "timestamps_sha256": file_sha256(Path(args.timestamps).resolve()),
            "frame_count": len(entries)})
        recording = load_recording(output / "recording.json")
        for _ in recording.iter_frames():  # verify converted bytes, not only metadata
            pass
    return {"output": str(Path(args.output).resolve()), "frames": len(entries),
            "scale_meters_per_colmap_unit": scale, "scale_source": args.scale_source.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Convert undistorted COLMAP poses + independently measured scale + "
        "registered metric Z-depth; does NOT estimate any of these."))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--timestamps", required=True, type=Path)
    parser.add_argument("--meters-per-colmap-unit", required=True, type=float)
    parser.add_argument("--scale-source", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        import json
        print(json.dumps(import_colmap(args), ensure_ascii=False))
        return 0
    except (DataError, OSError, ValueError, MemoryError) as error:
        import json
        import sys
        print(json.dumps({"error": str(error), "type": type(error).__name__}, ensure_ascii=False),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
