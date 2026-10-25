"""Optional indexed raw-depth preparation for the ONE COLMAP pose importer.

This module does not parse COLMAP poses into output transforms or publish its
own recording format: all indexed inputs call colmap.import_colmap after the
explicit raw-camera-Z -> metric-camera-Z conversion.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import tempfile

import numpy as np
from PIL import Image

from .colmap import _frame_name, _images, import_colmap
from .model import DataError, file_sha256, safe_path
from .output import output_transaction, write_json


def _depth_mapping(index_path: Path) -> dict[str, str]:
    if not index_path.is_file() or index_path.stat().st_size > 16 * 1024 * 1024:
        raise DataError("Missing or oversized depth-index file")
    try:
        rows = index_path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeError as error:
        raise DataError("Depth index must be UTF-8") from error
    mapping: dict[str, str] = {}
    paths: set[str] = set()
    for line_number, line in enumerate(rows, 1):
        row = line.partition("#")[0].strip()
        if not row:
            continue
        fields = row.split()
        if len(fields) != 2:
            raise DataError(f"depth-index line {line_number}: expected image_name relative_depth_path")
        name, relative = fields
        name = _frame_name(name)
        path = PurePosixPath(relative)
        if (relative.startswith("/") or "\\" in relative or
                any(part in ("", ".", "..") for part in relative.split("/")) or
                path.suffix.lower() not in (".png", ".npy")):
            raise DataError(f"Unsafe or unsupported indexed depth path: {relative}")
        if name in mapping or relative in paths:
            raise DataError(f"Duplicate image or depth path in depth-index: {name}")
        mapping[name] = relative
        paths.add(relative)
    if not mapping:
        raise DataError("Depth index is empty")
    return mapping


def _metric_array(path: Path, divisor: float) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=False, mmap_mode="r")
        if not isinstance(raw, np.ndarray):
            if isinstance(raw, np.lib.npyio.NpzFile):
                raw.close()
            raise DataError("Indexed depth must be numeric NPY, not an NPZ archive")
    else:
        with Image.open(path) as image:
            if image.width * image.height > 4_000_000:
                raise DataError("Indexed depth exceeds 4 million pixels")
            raw = np.asarray(image)
    if (raw.ndim != 2 or raw.size > 4_000_000 or
            raw.dtype.kind not in ("f", "i", "u")):
        raise DataError("Indexed depth must be a 2D real numeric array")
    if raw.dtype.kind == "f" and divisor != 1.0:
        raise DataError("Floating-point indexed depth is already in metres; use --depth-scale 1")
    # Use float64 for conversion so large integer values cannot overflow before validation.
    metric = raw.astype(np.float64) / divisor
    if not np.isfinite(metric).all() or (metric < 0).any() or (metric > 100).any():
        raise DataError("Indexed camera-Z depth must convert to finite [0,100] metres")
    return metric.astype(np.float32), str(raw.dtype)


def import_colmap_indexed(args: argparse.Namespace) -> dict:
    """Stage meter-valued NPYs and invoke the unchanged strict COLMAP core."""
    divisor = args.depth_scale
    if divisor is None or not math.isfinite(divisor) or not 0 < divisor <= 1e9:
        raise DataError("Indexed input requires positive finite --depth-scale (raw units per metre)")
    destination = Path(args.output).resolve()
    if destination.exists():
        raise DataError("Output already exists; refusing to overwrite")
    index_path = Path(args.depth_index).resolve()
    index_hash = file_sha256(index_path) if index_path.is_file() else None
    mapping = _depth_mapping(index_path)
    poses = _images(Path(args.model).resolve() / "images.txt")
    names = {name for name, _, _ in poses}
    if names != set(mapping):
        raise DataError("Depth index must contain exactly the registered COLMAP image names")

    root = Path(args.depth).resolve()
    evidence = []
    with tempfile.TemporaryDirectory(prefix="echo-colmap-depth-") as temporary:
        staged = Path(temporary) / "metric-depth"
        staged.mkdir()
        for name in sorted(names):
            raw_rel = mapping[name]
            raw_path = safe_path(root, raw_rel)
            before = file_sha256(raw_path)
            metric, dtype = _metric_array(raw_path, divisor)
            if file_sha256(raw_path) != before:
                raise DataError(f"Raw depth changed during conversion: {raw_rel}")
            output_name = str(PurePosixPath(name).with_suffix(".npy"))
            target = staged / output_name
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, metric, allow_pickle=False)
            evidence.append({"image": name, "raw_depth": raw_rel,
                             "raw_depth_sha256": before, "raw_dtype": dtype})
        if file_sha256(index_path) != index_hash:
            raise DataError("Depth index changed during conversion")
        with output_transaction(destination) as output:
            child_args = argparse.Namespace(**vars(args))
            child_args.depth = staged
            child_args.output = output / "_core"
            result = import_colmap(child_args)  # sole camera parser, pose scale and manifest builder
            for item in child_args.output.iterdir():
                item.rename(output / item.name)
            child_args.output.rmdir()
            provenance_path = output / "import-provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["indexed_depth"] = {
                "index_sha256": index_hash,
                "raw_units_per_metre": divisor,
                "conversion": "raw camera-Z depth / raw_units_per_metre; no COLMAP pose-scale applied",
                "frames": [
                    {**row, "metric_depth_sha256": file_sha256(
                        output / "depth" / str(PurePosixPath(row["image"]).with_suffix(".npy")))}
                    for row in evidence
                ],
            }
            write_json(provenance_path, provenance)
            if file_sha256(index_path) != index_hash:
                raise DataError("Depth index changed before publication")
            for row in evidence:
                if file_sha256(safe_path(root, row["raw_depth"])) != row["raw_depth_sha256"]:
                    raise DataError("Raw indexed depth changed before publication")
    return {**result, "output": str(destination), "indexed_depth": True}
