"""Explicit indexed/raw-depth preparation for the single strict COLMAP importer.

This module does not infer depth, alignment, or the physical scale. It converts
known registered camera-Z measurements to metres, then delegates all poses,
timestamps, and echo-dataset-1 publication to echo4d.colmap.import_colmap.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path, PurePosixPath
import shutil
import tempfile

import numpy as np
from PIL import Image

from .colmap import _cameras, _frame_name, _images, import_colmap
from .model import DataError, file_sha256, safe_path
from .output import output_transaction, write_json


def _read_index(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise DataError("Missing or oversized depth index CSV")
    rows: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["name", "depth_path"]:
            raise DataError("Depth index CSV header must be name,depth_path")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise DataError("Malformed depth index CSV row")
            name = _frame_name(row["name"])
            path_value = row["depth_path"]
            rel = PurePosixPath(path_value)
            if (not path_value or "\\" in path_value or rel.is_absolute()
                    or any(part in ("", ".", "..") for part in path_value.split("/"))
                    or rel.suffix.lower() not in (".png", ".npy")):
                raise DataError(f"Unsafe or unsupported indexed depth path: {path_value}")
            if name in rows:
                raise DataError(f"Duplicate image in depth index: {name}")
            rows[name] = path_value
    if not rows:
        raise DataError("Depth index CSV is empty")
    if len(set(rows.values())) != len(rows):
        raise DataError("One indexed depth file cannot be assigned to multiple images")
    return rows


def _depth_as_metres(path: Path, camera_shape: tuple[int, int],
                     units_per_metre: float) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".png":
        with Image.open(path) as image:
            if image.size != (camera_shape[1], camera_shape[0]):
                raise DataError(f"Indexed depth shape does not match COLMAP camera: {path.name}")
            raw = np.asarray(image)
        if raw.ndim != 2 or raw.dtype.kind not in "ui":
            raise DataError(f"PNG indexed depth must be single-channel integer camera-Z: {path.name}")
    else:
        raw = np.load(path, allow_pickle=False, mmap_mode="r")
        if not isinstance(raw, np.ndarray):
            if isinstance(raw, np.lib.npyio.NpzFile):
                raw.close()
            raise DataError(f"Indexed depth must be an NPY array, not NPZ: {path.name}")
        if raw.dtype.kind not in "fiu":
            raise DataError(f"Indexed NPY depth must be real numeric: {path.name}")
    if raw.shape != camera_shape:
        raise DataError(f"Indexed depth shape does not match COLMAP camera: {path.name}")
    dtype = str(raw.dtype)
    converted = raw.astype(np.float64) / units_per_metre
    if (not np.isfinite(converted).all() or np.any(converted < 0)
            or np.any(converted > 100)):
        raise DataError(f"Indexed depth has invalid values after conversion to metres: {path.name}")
    metric = converted.astype(np.float32)
    if not np.isfinite(metric).all():
        raise DataError(f"Indexed depth overflows metric float32: {path.name}")
    return metric, dtype


def import_colmap_indexed_depth(args: argparse.Namespace) -> dict:
    units = args.depth_units_per_meter
    if not math.isfinite(units) or units <= 0 or units > 1e12:
        raise DataError("--depth-units-per-meter must be finite, positive and <= 1e12")
    if not isinstance(args.depth_unit, str) or not args.depth_unit.strip():
        raise DataError("--depth-unit must name the independently known raw sensor unit")
    destination = Path(args.output).resolve()
    if destination.exists():
        raise DataError(f"Output already exists; choose a new directory: {destination}")

    model_dir = Path(args.model).resolve()
    depth_root = Path(args.depth_root).resolve()
    index_path = Path(args.depth_index).resolve()
    index = _read_index(index_path)
    images = _images(model_dir / "images.txt")
    cameras = _cameras(model_dir / "cameras.txt")
    used_ids = {camera_id for _, camera_id, _ in images}
    if len(used_ids) != 1 or not used_ids.issubset(cameras):
        raise DataError("Indexed depth requires one known COLMAP camera")
    camera = cameras[next(iter(used_ids))]
    registered = {name for name, _, _ in images}
    if registered != set(index):
        missing = sorted(registered - set(index))
        extra = sorted(set(index) - registered)
        raise DataError(f"Depth index must match registered images exactly; missing={missing[:3]}, extra={extra[:3]}")

    # The temporary converted data and strict importer output are never
    # publicly exposed. The outer transaction publishes both files and
    # original raw-unit provenance atomically, or nothing at all.
    evidence = []
    with tempfile.TemporaryDirectory(prefix="echo4d-indexed-") as scratch:
        scratch = Path(scratch)
        metric_root = scratch / "metric-depth"
        for name in sorted(registered):
            raw_relative = index[name]
            source = safe_path(depth_root, raw_relative)
            original_hash = file_sha256(source)
            metric, dtype = _depth_as_metres(
                source, (camera.height, camera.width), units)
            if file_sha256(source) != original_hash:
                raise DataError(f"Indexed depth changed during conversion: {name}")
            relative_npy = str(PurePosixPath(name).with_suffix(".npy"))
            converted_path = metric_root / relative_npy
            if converted_path.exists():
                raise DataError("COLMAP image names collide after mapping to NPY depth")
            converted_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(converted_path, metric, allow_pickle=False)
            evidence.append({
                "image": name, "raw_depth_path": raw_relative,
                "raw_depth_sha256": original_hash,
                "raw_depth_dtype": dtype,
                "converted_metric_depth_path": relative_npy,
                "converted_metric_sha256": file_sha256(converted_path),
            })

        strict_output = scratch / "strict-import"
        imported = import_colmap(argparse.Namespace(
            model=args.model, images=args.images, depth=metric_root,
            timestamps=args.timestamps,
            meters_per_colmap_unit=args.meters_per_colmap_unit,
            scale_source=args.scale_source, output=strict_output,
        ))
        with output_transaction(destination) as output:
            shutil.copytree(strict_output, output, dirs_exist_ok=True)
            write_json(output / "indexed-depth-provenance.json", {
                "version": "echo-indexed-depth-preparation-1",
                "depth_index_sha256": file_sha256(index_path),
                "raw_depth_unit": args.depth_unit.strip(),
                "raw_depth_units_per_meter": units,
                "conversion": "metric_camera_z_m = raw_camera_z / raw_depth_units_per_meter",
                "camera_scale": "handled only by strict import-colmap; not applied to depth",
                "limitations": [
                    "Index and unit metadata are supplied by the caller, not inferred",
                    "Depth must already be registered camera-Z, not ray length or relative AI depth",
                ],
                "frames": evidence,
            })
    return {
        "output": str(destination), "frames": imported["frames"],
        "scale_meters_per_colmap_unit": imported["scale_meters_per_colmap_unit"],
        "raw_depth_units_per_meter": units,
    }
