"""Deterministic mesh export and completed-directory publication."""
from __future__ import annotations
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import numpy as np
from .fusion import Mesh
from .model import Camera, DataError, Recording


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")


@contextmanager
def output_transaction(destination: str | Path) -> Iterator[Path]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise DataError(f"Output already exists; choose a new directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        yield temporary
        if destination.exists():
            raise DataError("Output appeared during processing; refusing to overwrite")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def write_ply(path: Path, mesh: Mesh) -> None:
    mesh.validate()
    header = ("ply\nformat binary_little_endian 1.0\ncomment ECHO TSDF; metres; known camera poses\n"
              f"element vertex {len(mesh.vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(mesh.faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    dtype = [(key, "<f4") for key in ("x", "y", "z", "nx", "ny", "nz")]
    dtype += [(key, "u1") for key in ("red", "green", "blue")]
    vertices = np.empty(len(mesh.vertices), dtype=dtype)
    for i, key in enumerate(("x", "y", "z")):
        vertices[key] = mesh.vertices[:, i]
    for i, key in enumerate(("nx", "ny", "nz")):
        vertices[key] = mesh.normals[:, i]
    for i, key in enumerate(("red", "green", "blue")):
        vertices[key] = mesh.colors[:, i]
    faces = np.empty(len(mesh.faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    faces["count"], faces["indices"] = 3, mesh.faces
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(vertices.tobytes())
        stream.write(faces.tobytes())
        stream.flush()
        os.fsync(stream.fileno())


def mesh_summary(mesh: Mesh) -> dict:
    triangles = mesh.vertices[mesh.faces].astype(np.float64)
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) / 2
    edges = np.sort(np.concatenate([mesh.faces[:, [0,1]], mesh.faces[:, [1,2]], mesh.faces[:, [2,0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {"vertices": len(mesh.vertices), "triangles": len(mesh.faces),
            "bounds_m": [mesh.vertices.min(axis=0).tolist(), mesh.vertices.max(axis=0).tolist()],
            "surface_area_m2": float(areas.sum()), "zero_area_faces": int(np.count_nonzero(areas < 1e-12)),
            "boundary_edges": int(np.count_nonzero(counts == 1)), "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
            "min_nearest_voxel_observations": int(mesh.observations.min()),
            "max_nearest_voxel_observations": int(mesh.observations.max()),
            "note": "Open surfaces are expected at observation boundaries. Support counts are not calibrated confidence."}


def preview_recording(recording: Recording, max_frames: int = 24, max_width: int = 80) -> dict:
    if max_frames < 1 or max_width < 2:
        raise DataError("Preview limits must be positive")
    stride = max(1, int(np.ceil(len(recording) / max_frames)))
    sx = max(1, int(np.ceil(recording.camera.width / max_width)))
    sy = max(1, int(np.ceil(recording.camera.height / 60)))
    c = recording.camera
    camera = Camera((c.width+sx-1)//sx, (c.height+sy-1)//sy, c.fx/sx, c.fy/sy, c.cx/sx, c.cy/sy)
    frames = []
    for _, frame, _ in recording.iter_frames(stride=stride, limit=max_frames):
        depth = frame.depth[::sy, ::sx].copy()
        if frame.excluded is not None:
            depth[frame.excluded[::sy, ::sx]] = 0
        entry = {"t": frame.t, "cameraToWorld": frame.camera_to_world.flatten(order="F").tolist(),
                 "depth": np.round(depth, 5).ravel().tolist()}
        if frame.color is not None:
            entry["colors"] = frame.color[::sy, ::sx].ravel().tolist()
        frames.append(entry)
    return {"version": "echo-1", "name": recording.name, "source": recording.source, "units": "meters",
            "duration": recording.duration, "intrinsics": camera.to_dict(), "frames": frames,
            "entities": recording.entities or [],
            "preview_provenance": {
                "loaded_recording_resolution_px": {"width": c.width, "height": c.height},
                "exported_preview_resolution_px": {"width": camera.width, "height": camera.height},
                "pixel_stride_xy": [sx, sy], "frame_stride": stride,
                "original_capture_resolution_px": None,
                "is_original_capture": False,
                "warning": "This is a sampled inspection preview, not the original RGB capture. "
                           "The loaded recording may already have been downsampled upstream."
            },
            "notes": "Downsampled inspection preview; source capture resolution is not inferred."}


def write_scene(path: Path, recording: Recording, mesh: Mesh, report: dict) -> None:
    if len(mesh.vertices) > 180000 or len(mesh.faces) > 300000:
        raise DataError("Browser export exceeds 180k vertices / 300k faces; use --no-scene")
    times = np.array([entry["t"] for entry in recording.entries])
    write_json(path, {"version": "echo-scene-1", "algorithm": "projective-tsdf-v1",
        "recording": preview_recording(recording),
        "mesh": {"vertices": np.round(mesh.vertices, 5).tolist(), "faces": mesh.faces.tolist(),
                 "colors": mesh.colors.tolist(), "observations": mesh.observations.tolist(),
                 "first": times[mesh.first_frame].tolist(), "last": times[mesh.last_frame].tolist()},
        "provenance": {"manifest_sha256": recording.manifest_sha256, "config": report["config"],
                       "note": "Vertex evidence is sampled from the nearest supported TSDF voxel."}})
    if path.stat().st_size > 40 * 1024 * 1024:
        raise DataError("Browser scene exceeds 40 MiB; use --no-scene")
