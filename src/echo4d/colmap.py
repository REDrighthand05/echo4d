"""COLMAP TEXT + external metric depth adapter for echo-dataset-1."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .model import Camera, DataError, file_sha256, rigid_matrix, safe_path
from .output import output_transaction, write_json


def _read_text_rows(path: Path) -> list[list[str]]:
    if path.stat().st_size > 128 * 1024 * 1024:
        raise DataError(f"File too large: {path.name}")
    rows: list[list[str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        clean = line.partition("#")[0].strip()
        if not clean:
            continue
        rows.append(clean.split())
    if not rows:
        raise DataError(f"Empty file: {path.name}")
    return rows


def _read_mapping(path: Path, value_name: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for number, fields in enumerate(_read_text_rows(path), 1):
        if len(fields) != 2:
            raise DataError(f"{path.name}:{number}: expected 2 columns: image_name {value_name}")
        key, value = fields
        if key in mapping:
            raise DataError(f"{path.name}:{number}: duplicate image name: {key}")
        mapping[key] = value
    return mapping


def _read_timestamps(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for number, fields in enumerate(_read_text_rows(path), 1):
        if len(fields) != 2:
            raise DataError(f"{path.name}:{number}: expected 2 columns: image_name timestamp")
        name, value = fields
        try:
            t = float(value)
        except ValueError as error:
            raise DataError(f"{path.name}:{number}: invalid timestamp: {value}") from error
        if not math.isfinite(t) or t < 0:
            raise DataError(f"{path.name}:{number}: timestamp must be finite and nonnegative")
        if name in result:
            raise DataError(f"{path.name}:{number}: duplicate image name: {name}")
        result[name] = t
    return result


def _parse_camera(colmap_text: Path) -> tuple[Camera, int, str]:
    cameras_path = safe_path(colmap_text, "cameras.txt")
    rows = _read_text_rows(cameras_path)
    if len(rows) != 1:
        raise DataError("COLMAP import currently requires exactly one camera entry")
    fields = rows[0]
    if len(fields) < 5:
        raise DataError("cameras.txt row is incomplete")
    camera_id_raw, model, width, height, *params = fields
    try:
        camera_id = int(camera_id_raw)
    except ValueError as error:
        raise DataError("Camera id must be an integer") from error
    try:
        width_i, height_i = int(width), int(height)
    except ValueError as error:
        raise DataError("Camera width/height must be integers") from error
    if model == "SIMPLE_PINHOLE":
        if len(params) != 3:
            raise DataError("SIMPLE_PINHOLE expects 3 parameters")
        fx = fy = float(params[0])
        cx, cy = float(params[1]), float(params[2])
    elif model == "PINHOLE":
        if len(params) != 4:
            raise DataError("PINHOLE expects 4 parameters")
        fx, fy, cx, cy = map(float, params)
    else:
        raise DataError(
            f"Unsupported COLMAP camera model {model}; undistort to PINHOLE/SIMPLE_PINHOLE first"
        )
    return Camera(width_i, height_i, fx, fy, cx, cy), camera_id, model


def _parse_images(colmap_text: Path) -> list[dict]:
    images_path = safe_path(colmap_text, "images.txt")
    lines = images_path.read_text(encoding="utf-8").splitlines()
    entries: list[dict] = []
    camera_ids: set[int] = set()
    names: set[str] = set()
    for number, line in enumerate(lines, 1):
        fields = line.partition("#")[0].strip().split()
        if not fields:
            continue
        if len(fields) < 10:
            continue
        name = " ".join(fields[9:]).strip()
        if not any(ch.isalpha() for ch in name):
            continue
        try:
            image_id = int(fields[0])
            qw, qx, qy, qz, tx, ty, tz = map(float, fields[1:8])
            camera_id = int(fields[8])
        except ValueError as error:
            raise DataError(f"images.txt:{number}: invalid numeric values in image header") from error
        quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or abs(norm - 1) > 0.01:
            raise DataError(f"images.txt:{number}: quaternion must be unit length (tolerance .01)")
        rotation_wc = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        translation_wc = np.array([tx, ty, tz], dtype=np.float64)
        rotation_cw = rotation_wc.T
        camera_center = -rotation_cw @ translation_wc
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation_cw
        matrix[:3, 3] = camera_center
        rigid_matrix(matrix.flatten(order="F"), "cameraToWorld converted from COLMAP")
        if name in names:
            raise DataError(f"Duplicate image name in images.txt: {name}")
        names.add(name)
        entries.append({
            "image_id": image_id,
            "camera_id": camera_id,
            "name": name,
            "camera_to_world": matrix,
        })
        camera_ids.add(camera_id)
    if not entries:
        raise DataError("No COLMAP image poses found in images.txt")
    if len(camera_ids) != 1:
        raise DataError("COLMAP import currently requires one shared camera id")
    entries.sort(key=lambda item: item["image_id"])
    return entries


def import_colmap(args: argparse.Namespace) -> dict:
    if not args.scale_evidence or not args.scale_evidence.strip():
        raise DataError("--scale-evidence is required and cannot be empty")
    if not math.isfinite(args.depth_scale) or args.depth_scale <= 0:
        raise DataError("--depth-scale must be positive and finite")
    root = args.dataset.resolve()
    if not root.is_dir():
        raise DataError("dataset must be an existing directory")
    colmap_text = args.colmap_text.resolve()
    if not colmap_text.is_dir():
        raise DataError("--colmap-text must be an existing directory")

    camera, camera_id, camera_model = _parse_camera(colmap_text)
    poses = _parse_images(colmap_text)
    if any(pose["camera_id"] != camera_id for pose in poses):
        raise DataError("images.txt camera ids do not match cameras.txt")
    depths = _read_mapping(safe_path(root, args.depth_index), "depth_path")
    timestamps = _read_timestamps(safe_path(root, args.timestamps))

    records: list[dict] = []
    for pose in poses:
        name = pose["name"]
        if name not in timestamps:
            raise DataError(f"Missing timestamp for COLMAP image: {name}")
        if name not in depths:
            raise DataError(f"Missing depth for COLMAP image: {name}")
        records.append({
            "name": name,
            "timestamp": timestamps[name],
            "depth_rel": depths[name],
            "camera_to_world": pose["camera_to_world"],
        })

    records.sort(key=lambda row: (row["timestamp"], row["name"]))
    previous = -math.inf
    for row in records:
        if row["timestamp"] <= previous:
            raise DataError("Timestamps must be strictly increasing after association")
        previous = row["timestamp"]

    with output_transaction(args.output) as output:
        (output / "depth").mkdir()
        (output / "color").mkdir()
        frames, provenance = [], []
        start = records[0]["timestamp"]
        for i, row in enumerate(records):
            depth_path = safe_path(root, row["depth_rel"])
            color_path = safe_path(root, row["name"])
            depth_hash, color_hash = file_sha256(depth_path), file_sha256(color_path)

            if depth_path.suffix.lower() == ".npy":
                raw = np.load(depth_path, allow_pickle=False)
            else:
                with Image.open(depth_path) as image:
                    raw = np.array(image)
            if raw.ndim != 2 or raw.shape != (camera.height, camera.width) or raw.dtype.kind not in "fiu":
                raise DataError(f"Depth for {row['name']} must be single-channel numeric with shape {(camera.height, camera.width)}")
            depth = raw.astype(np.float32) / args.depth_scale
            if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(depth > 100):
                raise DataError(f"Depth for {row['name']} produced invalid values outside [0,100] meters")

            with Image.open(color_path) as image:
                if image.size != (camera.width, camera.height):
                    raise DataError(f"Color image size mismatch for {row['name']}: {image.size}")
                rgb = np.array(image.convert("RGB"))

            if file_sha256(depth_path) != depth_hash or file_sha256(color_path) != color_hash:
                raise DataError("Input changed during conversion")

            np.save(output / "depth" / f"{i:05d}.npy", depth, allow_pickle=False)
            Image.fromarray(rgb).save(output / "color" / f"{i:05d}.png")
            frames.append({
                "t": row["timestamp"] - start,
                "depth": f"depth/{i:05d}.npy",
                "color": f"color/{i:05d}.png",
                "cameraToWorld": row["camera_to_world"].flatten(order="F").tolist(),
            })
            provenance.append({
                "index": i,
                "timestamp": row["timestamp"],
                "image": row["name"],
                "depth_path": row["depth_rel"],
                "color_sha256": color_hash,
                "depth_sha256": depth_hash,
            })

        write_json(output / "recording.json", {
            "version": "echo-dataset-1",
            "name": root.name,
            "source": "recorded",
            "units": "meters",
            "duration": max(frames[-1]["t"], 0.001),
            "depth_scale": 1,
            "intrinsics": camera.to_dict(),
            "frames": frames,
            "notes": "COLMAP world->camera converted to camera->world; external metric depth required.",
        })
        write_json(output / "import-provenance.json", {
            "version": "echo-colmap-import-1",
            "colmap_text": str(colmap_text),
            "camera_model": camera_model,
            "scale_evidence": args.scale_evidence.strip(),
            "frames": provenance,
            "cameras_sha256": file_sha256(safe_path(colmap_text, "cameras.txt")),
            "images_sha256": file_sha256(safe_path(colmap_text, "images.txt")),
            "timestamps_sha256": file_sha256(safe_path(root, args.timestamps)),
            "depth_index_sha256": file_sha256(safe_path(root, args.depth_index)),
            "source_format": "https://colmap.github.io/format.html",
        })

    return {"output": str(args.output.resolve()), "frames": len(frames)}
