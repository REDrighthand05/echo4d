"""M2 alternative: observed per-frame projective depth triangles, not TSDF.

This does not fuse overlapping frames, fill unseen surfaces, estimate poses,
or claim watertight geometry. Disconnected patches are expected.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .fusion import Mesh
from .model import DataError, Recording, file_sha256, load_recording, safe_path
from .output import mesh_summary, output_transaction, write_json, write_ply


def projective_mesh(
    recording: Recording,
    *,
    pixel_stride: int = 4,
    frame_stride: int = 1,
    min_depth: float = 0.15,
    max_depth: float = 6.0,
    max_depth_jump: float = 0.12,
    max_edge_length: float = 0.30,
    max_vertices: int = 300_000,
    max_faces: int = 500_000,
) -> tuple[Mesh, dict[str, str]]:
    """Triangulate only observed depth pixels in each source camera.

    No triangles span frames. Returns an ECHO Mesh and consumed source hashes.
    """
    for key, value in (("pixel_stride", pixel_stride), ("frame_stride", frame_stride),
                       ("max_vertices", max_vertices), ("max_faces", max_faces)):
        if type(value) is not int or value < 1:
            raise DataError(f"{key} must be a positive integer")
    for key, value in (("min_depth", min_depth), ("max_depth", max_depth),
                       ("max_depth_jump", max_depth_jump),
                       ("max_edge_length", max_edge_length)):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise DataError(f"{key} must be a positive finite number")
    if min_depth >= max_depth or max_depth > 100:
        raise DataError("Invalid depth clipping range")
    if pixel_stride > 16:
        raise DataError("pixel_stride exceeds bounded sampling limit of 16")
    camera = recording.camera
    height = len(range(0, camera.height, pixel_stride))
    width = len(range(0, camera.width, pixel_stride))
    count = len(range(0, len(recording), frame_stride))
    # Fail before loading frames when the theoretical allocation exceeds budget.
    if count * height * width > max_vertices or count * max(0, height - 1) * max(0, width - 1) * 2 > max_faces:
        raise DataError("Projective mesh upper bound exceeds vertex/face budget")
    if width < 2 or height < 2:
        raise DataError("Sampling removes all possible triangles")

    xx, yy = np.meshgrid(np.arange(0, camera.width, pixel_stride, dtype=np.float64),
                         np.arange(0, camera.height, pixel_stride, dtype=np.float64))
    grid = np.arange(height * width, dtype=np.int32).reshape(height, width)
    a, b, c, d = (grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:])
    # Image Y is down, so winding points normals toward the camera.
    candidate = np.concatenate((
        np.stack((a, c, b), axis=-1).reshape(-1, 3),
        np.stack((b, c, d), axis=-1).reshape(-1, 3),
    ), axis=0)

    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    sources: dict[str, str] = {}
    total_vertices = total_faces = 0
    for frame_index, frame, evidence in recording.iter_frames(stride=frame_stride):
        for item in evidence["files"].values():
            name, digest = item["path"], item["sha256"]
            if name in sources and sources[name] != digest:
                raise DataError("Source file changed between frame reads")
            sources[name] = digest
        depth = frame.depth[::pixel_stride, ::pixel_stride].astype(np.float64, copy=False)
        # Conservative block support: a sampled vertex is invalid if ANY
        # original pixel in its tile is missing, masked or discontinuous.
        # Otherwise stride>1 could bridge an unsampled hole and invent surface.
        source_valid = (frame.depth >= min_depth) & (frame.depth <= max_depth)
        if frame.excluded is not None:
            source_valid &= ~frame.excluded
        valid = np.ones(depth.shape, dtype=bool)
        tile_min = depth.copy()
        tile_max = depth.copy()
        for dy in range(pixel_stride):
            for dx in range(pixel_stride):
                block = frame.depth[dy::pixel_stride, dx::pixel_stride]
                if not block.size:
                    continue
                bh, bw = block.shape
                valid[:bh, :bw] &= source_valid[dy::pixel_stride, dx::pixel_stride]
                tile_min[:bh, :bw] = np.minimum(tile_min[:bh, :bw], block)
                tile_max[:bh, :bw] = np.maximum(tile_max[:bh, :bw], block)
        valid &= (tile_max - tile_min <= max_depth_jump)
        local = np.stack(((xx - camera.cx) * depth / camera.fx,
                          (yy - camera.cy) * depth / camera.fy, depth), axis=-1).reshape(-1, 3)
        world = local @ frame.camera_to_world[:3, :3].T + frame.camera_to_world[:3, 3]
        flat_depth = depth.ravel()
        face = candidate[np.all(valid.ravel()[candidate], axis=1)]
        if not len(face):
            continue
        z = flat_depth[face]
        face = face[np.ptp(z, axis=1) <= max_depth_jump]
        if not len(face):
            continue
        xyz = world[face]
        edge01 = np.linalg.norm(xyz[:, 1] - xyz[:, 0], axis=1)
        edge12 = np.linalg.norm(xyz[:, 2] - xyz[:, 1], axis=1)
        edge20 = np.linalg.norm(xyz[:, 0] - xyz[:, 2], axis=1)
        cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        area2 = np.linalg.norm(cross, axis=1)
        face = face[(np.maximum.reduce((edge01, edge12, edge20)) <= max_edge_length) & (area2 > 1e-10)]
        if not len(face):
            continue
        used = np.unique(face)
        mapped = np.searchsorted(used, face).astype(np.int32)
        points = world[used].astype(np.float32)
        if frame.color is None:
            colors = np.full((len(used), 3), 160, dtype=np.uint8)
        else:
            colors = frame.color[::pixel_stride, ::pixel_stride].reshape(-1, 3)[used].copy()
        normals = np.zeros((len(used), 3), dtype=np.float64)
        tri_points = world[face]
        tri_normals = np.cross(tri_points[:, 1] - tri_points[:, 0],
                               tri_points[:, 2] - tri_points[:, 0])
        np.add.at(normals, mapped.ravel(), np.repeat(tri_normals, 3, axis=0))
        lengths = np.linalg.norm(normals, axis=1)
        normals[lengths > 0] /= lengths[lengths > 0, None]
        total_vertices += len(used)
        total_faces += len(face)
        if total_vertices > max_vertices or total_faces > max_faces:
            raise DataError("Projective mesh exceeds resource budget")
        chunks.append((points, mapped, normals.astype(np.float32), colors, frame_index))
    if not chunks:
        raise DataError("No supported depth triangles; refusing empty Mesh")

    offsets = np.cumsum([0] + [len(c[0]) for c in chunks[:-1]])
    mesh = Mesh(
        vertices=np.concatenate([c[0] for c in chunks]),
        faces=np.concatenate([c[1] + offset for c, offset in zip(chunks, offsets)]),
        normals=np.concatenate([c[2] for c in chunks]),
        colors=np.concatenate([c[3] for c in chunks]),
        observations=np.ones(total_vertices, dtype=np.uint32),
        first_frame=np.concatenate([np.full(len(c[0]), c[4], dtype=np.int32) for c in chunks]),
        last_frame=np.concatenate([np.full(len(c[0]), c[4], dtype=np.int32) for c in chunks]),
    )
    mesh.validate()
    return mesh, sources


def run_projective(args: argparse.Namespace) -> dict:
    expected = args.expected_manifest_sha256
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise DataError("Expected immutable lowercase manifest SHA256")
    manifest = args.recording.resolve()
    output = args.output.resolve()
    if output == manifest.parent or output.is_relative_to(manifest.parent):
        raise DataError("Output must be outside input source directory")
    recording = load_recording(manifest)
    if file_sha256(manifest) != expected:
        raise DataError("Manifest differs from frozen expected SHA256")
    mesh, dependencies = projective_mesh(
        recording, pixel_stride=args.pixel_stride, frame_stride=args.frame_stride,
        max_depth=args.max_depth, max_depth_jump=args.max_depth_jump,
        max_edge_length=args.max_edge_length,
    )
    for name, digest in dependencies.items():
        if file_sha256(safe_path(recording.root, name)) != digest:
            raise DataError(f"Source changed during execution: {name}")
    if file_sha256(manifest) != expected:
        raise DataError("Manifest changed during execution")
    with output_transaction(output) as temp:
        write_ply(temp / "mesh.ply", mesh, comment="ECHO projective depth triangles; metres; known poses")
        receipt = {
            "schema": "echo4d-projective-depth-1",
            "status": "executed_unscored",
            "algorithm": "per-frame-projective-triangles-v1",
            "fusion": False,
            "manifest_sha256": expected,
            "source_claim": recording.source,
            "source_dependencies_sha256": dependencies,
            "mesh_sha256": file_sha256(temp / "mesh.ply"),
            "geometry": mesh_summary(mesh),
            "parameters": {k: getattr(args, k) for k in
                           ("pixel_stride", "frame_stride", "max_depth",
                            "max_depth_jump", "max_edge_length")},
            "independent_surface_truth": False,
            "human_visual_review": False,
            "quality": "not_evaluated",
            "limitations": ["Unmerged overlapping frame patches; not TSDF or watertight fusion",
                            "Known poses and metric depth supplied by caller",
                            "No unseen surfaces reconstructed",
                            "No independent geometry or #74 visual acceptance"],
        }
        write_json(temp / "report.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-depth", type=float, default=6.)
    parser.add_argument("--max-depth-jump", type=float, default=.12)
    parser.add_argument("--max-edge-length", type=float, default=.30)
    args = parser.parse_args(argv)
    try:
        result = run_projective(args)
    except (DataError, OSError, MemoryError, ValueError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}))
        return 2
    print(json.dumps({"status": result["status"], "mesh_sha256": result["mesh_sha256"],
                      "vertices": result["geometry"]["vertices"],
                      "triangles": result["geometry"]["triangles"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
