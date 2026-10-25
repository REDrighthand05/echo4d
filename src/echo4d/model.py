"""Validated streaming input. Camera axes: right, down, forward."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image


class DataError(ValueError):
    """The input violates a data or resource contract."""


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DataError(f"{name} must be a finite number")
    return float(value)


def safe_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise DataError("Frame files must use nonempty relative paths")
    root = root.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise DataError(f"Missing frame file or path outside dataset: {value}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rigid_matrix(values: Any, name: str = "cameraToWorld") -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DataError(f"{name} must contain 16 numeric values") from error
    if matrix.shape != (16,) or not np.isfinite(matrix).all():
        raise DataError(f"{name} must contain 16 finite column-major values")
    matrix = matrix.reshape(4, 4, order="F").copy()
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3, rtol=0)
            or abs(np.linalg.det(rotation) - 1) > 2e-3):
        raise DataError(f"{name} must be a right-handed rigid transform")
    if np.max(np.abs(matrix[:3, 3])) > 10000:
        raise DataError("Use local coordinates within 10 km of the origin")
    matrix.flags.writeable = False
    return matrix


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        for key in ("width", "height"):
            value = getattr(self, key)
            if type(value) is not int or not 2 <= value <= 4096:
                raise DataError(f"Camera {key} must be an integer in [2,4096]")
        if self.width * self.height > 4_000_000:
            raise DataError("Depth image exceeds 4 million pixels")
        for key in ("fx", "fy", "cx", "cy"):
            finite_number(getattr(self, key), key)
        if not (1e-3 <= self.fx <= 1e7 and 1e-3 <= self.fy <= 1e7):
            raise DataError("Focal lengths must be in [0.001,10000000] pixels")
        if max(abs(self.cx), abs(self.cy)) > 1e7:
            raise DataError("Principal point magnitude exceeds 10000000 pixels")

    @classmethod
    def from_dict(cls, data: Any) -> Camera:
        if not isinstance(data, dict):
            raise DataError("intrinsics must be an object")
        try:
            return cls(**{key: data[key] for key in ("width", "height", "fx", "fy", "cx", "cy")})
        except KeyError as error:
            raise DataError(f"Missing camera parameter: {error.args[0]}") from error

    def scaled(self, factor: int) -> Camera:
        if type(factor) is not int or factor < 1:
            raise DataError("Downsample factor must be a positive integer")
        # [::factor] keeps pixel centres 0, factor, ...; no half-pixel offset.
        return Camera((self.width + factor - 1) // factor, (self.height + factor - 1) // factor,
                      self.fx / factor, self.fy / factor, self.cx / factor, self.cy / factor)

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in ("width", "height", "fx", "fy", "cx", "cy")}


@dataclass(frozen=True)
class Frame:
    t: float
    camera_to_world: np.ndarray
    depth: np.ndarray
    color: np.ndarray | None = None
    excluded: np.ndarray | None = None

    def validate(self, camera: Camera) -> Frame:
        if not math.isfinite(self.t) or self.t < 0:
            raise DataError("Frame time must be finite and nonnegative")
        if self.camera_to_world.shape != (4, 4):
            raise DataError("Frame transform must be 4x4")
        rigid_matrix(self.camera_to_world.flatten(order="F"))
        shape = (camera.height, camera.width)
        if self.depth.shape != shape or self.depth.dtype.kind not in "fiu":
            raise DataError(f"Depth image must have real numeric shape {shape}")
        if not np.isfinite(self.depth).all() or np.any(self.depth < 0) or np.any(self.depth > 100):
            raise DataError("Depth must be finite in [0,100] metres; zero means missing")
        if self.color is not None and (self.color.shape != (*shape, 3) or self.color.dtype != np.uint8):
            raise DataError("Color must be registered uint8 RGB matching depth resolution")
        if self.excluded is not None and (self.excluded.shape != shape or self.excluded.dtype != bool):
            raise DataError("Exclusion mask must be boolean and match depth resolution")
        return self


def validate_entities(entities: Any, duration: float) -> None:
    if not isinstance(entities, list) or len(entities) > 8:
        raise DataError("entities must contain at most 8 tracks")
    identifiers = set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise DataError("Each entity must be an object")
        identifier = entity.get("id")
        names, bones, samples = (entity.get(key) for key in ("jointNames", "bones", "samples"))
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 80 or identifier in identifiers:
            raise DataError("Entity ids must be unique nonempty strings up to 80 characters")
        identifiers.add(identifier)
        if (not isinstance(names, list) or not 2 <= len(names) <= 40
                or any(not isinstance(n, str) or not 1 <= len(n) <= 80 for n in names)
                or len(set(names)) != len(names)):
            raise DataError("jointNames must contain 2-40 unique names")
        if (not isinstance(bones, list) or len(bones) > 80
                or any(not isinstance(b, list) or len(b) != 2
                       or any(type(j) is not int or not 0 <= j < len(names) for j in b) for b in bones)):
            raise DataError("Invalid bone indices")
        if not isinstance(samples, list) or len(samples) > 10000:
            raise DataError("Each track supports at most 10000 samples")
        previous = -1.0
        for sample in samples:
            if not isinstance(sample, dict):
                raise DataError("Skeleton samples must be objects")
            t = finite_number(sample.get("t"), "skeleton time")
            joints = sample.get("joints")
            if t < 0 or t <= previous or t > duration:
                raise DataError("Skeleton times must strictly increase within recording duration")
            if not isinstance(joints, list) or len(joints) != len(names):
                raise DataError("Each sample must supply every joint")
            for point in joints:
                if not isinstance(point, list) or len(point) != 3:
                    raise DataError("Joint positions need three world coordinates")
                if any(abs(finite_number(value, "joint coordinate")) > 10000 for value in point):
                    raise DataError("Joint positions must remain within 10 km")
            previous = t


@dataclass
class Recording:
    camera: Camera
    entries: list[dict]
    duration: float
    source: str
    name: str
    root: Path
    manifest_sha256: str
    depth_scale: float = 1.0
    inline: bool = False
    entities: list | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def iter_frames(self, stride: int = 1, limit: int | None = None) -> Iterator[tuple[int, Frame, dict]]:
        if type(stride) is not int or stride < 1 or (limit is not None and (type(limit) is not int or limit < 1)):
            raise DataError("stride and limit must be positive integers")
        selected = range(0, len(self.entries), stride)
        if limit is not None:
            selected = selected[:limit]
        for index in selected:
            spec = self.entries[index]
            camera = self.camera
            shape = (camera.height, camera.width)
            evidence = {"index": index, "t": spec["t"], "files": {}}
            if self.inline:
                depth = np.asarray(spec["depth"], dtype=np.float32).reshape(shape)
                color = np.asarray(spec["colors"], dtype=np.uint8).reshape(*shape, 3) if "colors" in spec else None
                excluded = np.asarray(spec["excluded"], dtype=bool).reshape(shape) if "excluded" in spec else None
            else:
                def read_image(key: str, mode: str) -> np.ndarray:
                    path = safe_path(self.root, spec[key])
                    before = file_sha256(path)
                    evidence["files"][key] = {"path": spec[key], "sha256": before}
                    if path.suffix.lower() == ".npy":
                        array = np.load(path, allow_pickle=False, mmap_mode="r")
                        if array.shape != ((*shape, 3) if mode == "RGB" else shape):
                            raise DataError(f"Unexpected {key} shape: {array.shape}")
                        array = np.array(array)
                    else:
                        with Image.open(path) as image:
                            if image.size != (camera.width, camera.height):
                                raise DataError(f"Unexpected {key} resolution: {image.size}")
                            array = np.array(image.convert("RGB") if mode == "RGB" else image)
                    if file_sha256(path) != before:
                        raise DataError(f"Input changed during reading: {spec[key]}")
                    return array
                raw = read_image("depth", "depth")
                if raw.dtype.kind not in "fiu":
                    raise DataError("Depth arrays must have a real numeric dtype")
                depth = raw.astype(np.float32) / self.depth_scale
                color = read_image("color", "RGB") if "color" in spec else None
                excluded = read_image("mask", "mask") > 0 if "mask" in spec else None
            frame = Frame(spec["t"], rigid_matrix(spec["cameraToWorld"]), depth, color, excluded).validate(camera)
            yield index, frame, evidence


def load_recording(path: str | Path) -> Recording:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size > 40 * 1024 * 1024:
        raise DataError("Recording manifest must exist and be at most 40 MiB")
    raw = path.read_bytes()
    def invalid_constant(value: str):
        raise DataError(f"Invalid JSON constant {value}")
    try:
        data = json.loads(raw, parse_constant=invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise DataError(f"Invalid JSON recording: {error}") from error
    if not isinstance(data, dict) or data.get("version") not in ("echo-1", "echo-dataset-1"):
        raise DataError("Expected echo-1 or echo-dataset-1 recording")
    if data.get("units") != "meters" or data.get("source") not in ("recorded", "synthetic"):
        raise DataError("units must be meters; source must be recorded or synthetic")
    camera = Camera.from_dict(data.get("intrinsics"))
    entries = data.get("frames")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 100000:
        raise DataError("A dataset must contain 1-100000 frames")
    inline = data["version"] == "echo-1"
    size = camera.width * camera.height
    if inline and (len(entries) > 300 or len(entries) * size > 1_500_000):
        raise DataError("Inline recordings are limited to 300 frames and 1.5 million depth pixels")
    previous = -1.0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise DataError(f"Frame {index} must be an object")
        t = finite_number(entry.get("t"), f"frame {index} time")
        if t < 0 or t <= previous:
            raise DataError("Frame times must be nonnegative and strictly increasing")
        previous = t
        rigid_matrix(entry.get("cameraToWorld"), f"frame {index} cameraToWorld")
        if inline:
            try:
                depth = np.asarray(entry.get("depth"), dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise DataError(f"Invalid depth array in frame {index}") from error
            if depth.shape != (size,) or not np.isfinite(depth).all() or np.any(depth < 0) or np.any(depth > 100):
                raise DataError(f"Invalid depth values or length in frame {index}")
            if "colors" in entry:
                color = np.asarray(entry["colors"], dtype=np.float64)
                if (color.shape != (size * 3,) or not np.isfinite(color).all() or np.any(color < 0)
                        or np.any(color > 255) or np.any(color != np.floor(color))):
                    raise DataError(f"Invalid RGB array in frame {index}")
            if "excluded" in entry:
                mask = entry["excluded"]
                if not isinstance(mask, list) or len(mask) != size or any(type(v) is not bool for v in mask):
                    raise DataError(f"Invalid boolean exclusion mask in frame {index}")
        else:
            for key in ("depth", "color", "mask"):
                if key == "depth" or key in entry:
                    safe_path(path.parent, entry.get(key))
    duration = finite_number(data.get("duration"), "duration")
    if duration <= 0 or duration < previous or duration > 86400:
        raise DataError("duration must cover all samples and be in (0,86400] seconds")
    scale = finite_number(data.get("depth_scale", 1), "depth_scale")
    if scale <= 0 or scale > 1e9 or (inline and scale != 1):
        raise DataError("depth_scale must be positive; inline depths must already be metres")
    entities = data.get("entities", [])
    validate_entities(entities, duration)
    return Recording(camera, entries, duration, data["source"], str(data.get("name", path.stem))[:200],
                     path.parent, hashlib.sha256(raw).hexdigest(), scale, inline, entities)
