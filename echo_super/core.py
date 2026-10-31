"""Uniform object — the PyObject-like layer.

Everything that flows through the super-function is wrapped in an
EchoObject: one header shape regardless of origin.

    kind   : what it is            ("video", "frames", "depth", "points", ...)
    data   : the payload           (path, array, dict, ... — untouched)
    t      : timestamp or (t0, t1) it belongs to, seconds, or None
    frame  : coordinate frame name ("world", "camera:12", ...) or None
    source : who produced it       ("user", a kernel name, a sensor, ...)
    meta   : free small dict (shape, units, notes ...)

Kernels never receive raw payloads from strangers; they receive
EchoObjects and return EchoObjects. That is the whole contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_EXT_KIND = {
    ".mp4": "video", ".mov": "video", ".mkv": "video", ".webm": "video",
    ".jpg": "photo", ".jpeg": "photo", ".png": "photo",
    ".ply": "points", ".bin": "points",
    ".glb": "scene", ".html": "scene",
    ".npy": "array", ".npz": "array",
    ".json": "record", ".csv": "record",
}


@dataclass
class EchoObject:
    kind: str
    data: Any
    t: Any = None
    frame: str | None = None
    source: str = "user"
    meta: dict = field(default_factory=dict)

    def header(self) -> dict:
        """Serializable view of the wrapper (payload excluded)."""
        return {"kind": self.kind, "t": self.t, "frame": self.frame,
                "source": self.source, "meta": self.meta}

    def derive(self, kind: str, data: Any, source: str, **meta) -> "EchoObject":
        """Child object inheriting time/frame context from this one."""
        merged = dict(self.meta); merged.update(meta)
        return EchoObject(kind=kind, data=data, t=self.t,
                          frame=self.frame, source=source, meta=merged)


def infer_kind(value: Any) -> str:
    if isinstance(value, EchoObject):
        return value.kind
    if isinstance(value, (str, Path)):
        name = str(value).lower()
        if "imu" in os.path.basename(name):
            return "imu"
        return _EXT_KIND.get(Path(name).suffix, "file")
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(v, (str, Path)) for v in value):
        return "photos"
    if isinstance(value, dict) and "kind" in value:
        return str(value["kind"])
    return type(value).__name__


def wrap(value: Any, kind: str | None = None, **kw) -> EchoObject:
    """Normalize anything the user throws at the super-function.

    Accepts: EchoObject (passthrough), a (kind, data) tuple, a path,
    a list of paths, or any payload with an explicit kind=.
    """
    if isinstance(value, EchoObject):
        return value
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        return EchoObject(kind=value[0], data=value[1], **kw)
    return EchoObject(kind=kind or infer_kind(value), data=value, **kw)
