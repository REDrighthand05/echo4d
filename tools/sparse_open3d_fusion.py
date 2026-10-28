"""Optional Open3D sparse RGB-D TSDF: real measured depth + known poses only.

Unlike ECHO's bounded dense TSDF, Open3D allocates voxel blocks near observations.
No unknown surfaces are filled and no measured depth is estimated. This module is
an independent geometry provider, not an input converter or provenance framework.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np

from echo4d.fusion import Mesh
from echo4d.model import Camera, DataError, file_sha256, load_recording
from echo4d.output import output_transaction, write_json, write_ply


def validate_settings(voxel_m: float, truncation_m: float, max_depth_m: float,
                      pixel_stride: int, edge_guard_m: float = 0.0,
                      consistency_guard_m: float = 0.0, edge_halo_pixels: int = 1) -> None:
    if not all(isinstance(v, (float, int)) and not isinstance(v, bool)
               and math.isfinite(v) for v in (voxel_m, truncation_m, max_depth_m, edge_guard_m, consistency_guard_m)):
        raise DataError("Sparse TSDF distances must be finite numbers")
    if not 0.005 <= voxel_m <= 0.10:
        raise DataError("Sparse voxel must be between 5mm and 10cm")
    if not 2 * voxel_m <= truncation_m <= 0.5:
        raise DataError("Sparse truncation must be at least two voxels and at most 50cm")
    if not 0.2 < max_depth_m <= 10:
        raise DataError("Sparse maximum depth must be within (0.2,10] metres")
    if type(pixel_stride) is not int or pixel_stride not in (1, 2, 4, 8):
        raise DataError("Sparse depth pixel stride must be 1, 2, 4 or 8")
    if not 0 <= edge_guard_m <= 0.30:
        raise DataError("Sparse depth edge guard must be between 0 and 30cm")
    if not 0 <= consistency_guard_m <= 0.30:
        raise DataError("Cross-frame depth guard must be between 0 and 30cm")
    if type(edge_halo_pixels) is not int or edge_halo_pixels not in (1, 2):
        raise DataError("Edge halo must be one or two measured pixels")


def reject_far_side_of_depth_edges(depth: np.ndarray, jump_m: float,
                                   halo_px: int = 1) -> tuple[np.ndarray, int]:
    """Suppress only the farther pixel at measured occlusion jumps, never fill holes.

    The far-side pixel next to a foreground occluder is prone to mixed-pixel
    depth and phantom TSDF bridges. Keep the nearer side and untouched interiors.
    Zero/invalid depth does not trigger rejection of adjacent valid observations.
    This is an experimental geometry filter, not a confidence measurement.
    """
    if type(halo_px) is not int or halo_px not in (1, 2):
        raise DataError("Edge halo must be one or two measured pixels")
    if jump_m == 0:
        return depth, 0
    if depth.ndim != 2:
        raise DataError("Edge guard requires a 2D depth image")
    valid = np.isfinite(depth) & (depth > 0)
    reject = np.zeros(depth.shape, dtype=bool)
    for a_slice, b_slice in ((np.s_[:-1, :], np.s_[1:, :]),
                             (np.s_[:, :-1], np.s_[:, 1:])):
        a, b = depth[a_slice], depth[b_slice]
        both = valid[a_slice] & valid[b_slice]
        # A relative floor avoids over-rejecting distant surfaces.
        threshold = np.maximum(jump_m, 0.04 * np.minimum(a, b))
        discontinuity = both & (np.abs(a - b) > threshold)
        reject[a_slice] |= discontinuity & (a > b)
        reject[b_slice] |= discontinuity & (b > a)
    if halo_px == 2:
        # Grow one pixel along the same measured far surface only; no hole fill.
        grown = np.zeros_like(reject)
        for a_slice, b_slice in ((np.s_[:-1, :], np.s_[1:, :]),
                                 (np.s_[:, :-1], np.s_[:, 1:])):
            a, b = depth[a_slice], depth[b_slice]
            both = valid[a_slice] & valid[b_slice]
            tolerance = np.maximum(jump_m, 0.04 * np.minimum(a, b))
            same_surface = both & (np.abs(a - b) <= tolerance)
            grown[a_slice] |= same_surface & reject[b_slice]
            grown[b_slice] |= same_surface & reject[a_slice]
        reject |= grown
    result = depth.copy()
    result[reject] = 0
    return result, int(np.count_nonzero(reject))



def select_cross_frame_neighbors(poses: list[np.ndarray], *, max_neighbors: int = 4
                                 ) -> list[list[int]]:
    """Choose real *training* witnesses from distinct camera positions, not holdouts.

    Prefer two views on each side in time, with useful parallax but overlapping
    optical axes. No neighbor is synthesized and no extra frame is imported.
    """
    centers = np.stack([pose[:3, 3] for pose in poses])
    axes = np.stack([pose[:3, 2] for pose in poses])
    result = []
    for i in range(len(poses)):
        left, right = [], []
        for j in range(len(poses)):
            if i == j:
                continue
            baseline = float(np.linalg.norm(centers[i] - centers[j]))
            if not (0.025 <= baseline <= 0.40 and
                    float(axes[i] @ axes[j]) >= 0.90):
                continue
            (left if j < i else right).append((baseline, abs(i - j), j))
        left.sort()
        right.sort()
        chosen = [v[2] for v in left[:max_neighbors // 2] +
                  right[:max_neighbors - max_neighbors // 2]]
        if len(chosen) < 2:
            chosen = [v[2] for v in sorted(left + right)[:max_neighbors]]
        result.append(chosen)
    return result


def reject_cross_frame_foreground(depth: np.ndarray, camera_to_world: np.ndarray,
                                  witnesses: list[tuple[np.ndarray, np.ndarray]],
                                  camera: Camera, guard_m: float) -> tuple[np.ndarray, int]:
    """Conservatively remove a foreground depth contradicted by two observed views.

    Project each measured 3D point into other *training* cameras. A stable,
    farther measured surface is contradictory (the source foreground should
    occlude it); a nearer surface is merely an occluder and NOT a contradiction.
    Require >=2 contradictions and zero agreements. Missing observations never
    vote. This can remove genuine thin geometry missed by both witnesses; it
    cannot recover surfaces absent from the source views.
    """
    if depth.ndim != 2 or depth.shape != (camera.height, camera.width):
        raise DataError("Cross-frame guard requires calibrated depth resolution")
    if not math.isfinite(guard_m) or guard_m < 0 or guard_m > .30:
        raise DataError("Invalid cross-frame depth guard")
    if guard_m == 0 or len(witnesses) < 2:
        return depth, 0
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0))
    if len(rows) == 0:
        return depth, 0
    z = depth[rows, cols].astype(np.float64)
    points = np.stack(((cols - camera.cx) * z / camera.fx,
                       (rows - camera.cy) * z / camera.fy, z))
    contradictions = np.zeros(len(z), dtype=np.uint8)
    agreements = np.zeros(len(z), dtype=np.uint8)
    for other_depth, other_pose in witnesses:
        if other_depth.shape != depth.shape:
            raise DataError("Cross-frame witness depth resolution mismatch")
        relative = np.linalg.inv(other_pose) @ camera_to_world
        projected = relative[:3, :3] @ points + relative[:3, 3, None]
        predicted = projected[2]
        safe_z = np.where(predicted > .15, predicted, 1.)
        u = np.rint(np.clip(camera.fx * projected[0] / safe_z + camera.cx,
                            -2 * camera.width, 2 * camera.width)).astype(np.int32)
        v = np.rint(np.clip(camera.fy * projected[1] / safe_z + camera.cy,
                            -2 * camera.height, 2 * camera.height)).astype(np.int32)
        inside = ((predicted > .15) & (u >= 1) & (u < camera.width - 1) &
                  (v >= 1) & (v < camera.height - 1))
        selected = np.flatnonzero(inside)
        if len(selected) == 0:
            continue
        x, y = u[selected], v[selected]
        # Require locally stable depth: do not compare across an occlusion edge.
        samples = np.stack((other_depth[y, x], other_depth[y - 1, x],
                            other_depth[y + 1, x], other_depth[y, x - 1],
                            other_depth[y, x + 1]))
        stable = (np.isfinite(samples).all(axis=0) &
                  (samples > 0).all(axis=0) &
                  ((samples.max(axis=0) - samples.min(axis=0)) <= guard_m))
        selected = selected[stable]
        if len(selected) == 0:
            continue
        measured = samples[0, stable]
        difference = measured - predicted[selected]
        tolerance = np.maximum(guard_m, .04 * predicted[selected])
        contradictions[selected] += (difference > tolerance).astype(np.uint8)
        agreements[selected] += (np.abs(difference) <= tolerance).astype(np.uint8)
    reject = (contradictions >= 2) & (agreements == 0)
    result = depth.copy()
    result[rows[reject], cols[reject]] = 0
    return result, int(np.count_nonzero(reject))


def fuse_sparse(recording_path: Path, destination: Path, *, voxel_m: float = 0.03,
                truncation_m: float = 0.09, max_depth_m: float = 4.0,
                pixel_stride: int = 2, edge_guard_m: float = 0.0,
                consistency_guard_m: float = 0.0,
                edge_halo_pixels: int = 1) -> dict:
    """Integrate registered RGB and camera-Z metric depth; output strict ECHO color PLY.

    Open3D expects world-to-camera extrinsics, whereas ECHO stores camera-to-world.
    Unobserved/excluded pixels are zeroed before integration. The exporter does not
    pretend to know per-vertex independent observation counts.
    """
    validate_settings(voxel_m, truncation_m, max_depth_m, pixel_stride,
                      edge_guard_m, consistency_guard_m, edge_halo_pixels)
    recording = load_recording(recording_path)
    if len(recording) > 120:
        raise DataError("Sparse quality experiment is bounded to 120 input frames")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise DataError("Open3D 0.19.0 must be installed for sparse TSDF") from exc

    camera = recording.camera
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_m), sdf_trunc=float(truncation_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        volume_unit_resolution=16, depth_sampling_stride=pixel_stride)
    count, valid_pixels, edge_removed_pixels, cross_removed_pixels = 0, 0, 0, 0
    start = time.perf_counter()
    prepared = []
    for _, frame, _ in recording.iter_frames():
        if frame.color is None:
            raise DataError("Sparse colored TSDF requires registered training RGB")
        depth = np.ascontiguousarray(frame.depth, dtype=np.float32).copy()
        invalid = (depth < 0.15) | (depth > max_depth_m)
        if frame.excluded is not None:
            invalid |= frame.excluded
        depth[invalid] = 0
        depth, removed = reject_far_side_of_depth_edges(depth, edge_guard_m,
                                                       edge_halo_pixels)
        edge_removed_pixels += removed
        prepared.append((frame, depth))
    neighbors = (select_cross_frame_neighbors([f.camera_to_world for f, _ in prepared])
                 if consistency_guard_m else [[] for _ in prepared])
    with output_transaction(destination) as output:
        for index, (frame, depth) in enumerate(prepared):
            if consistency_guard_m:
                witnesses = [(prepared[j][1], prepared[j][0].camera_to_world)
                             for j in neighbors[index]]
                depth, removed = reject_cross_frame_foreground(
                    depth, frame.camera_to_world, witnesses, camera, consistency_guard_m)
                cross_removed_pixels += removed
            valid_pixels += int(np.count_nonzero(depth))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(frame.color)),
                o3d.geometry.Image(depth), depth_scale=1.0,
                depth_trunc=float(max_depth_m), convert_rgb_to_intensity=False)
            # ECHO: cameraToWorld. Open3D integrate: worldToCamera.
            extrinsic = np.linalg.inv(frame.camera_to_world)
            volume.integrate(rgbd, intrinsic, extrinsic)
            count += 1
        if not valid_pixels:
            raise DataError("No valid observed depth for sparse TSDF")
        surface = volume.extract_triangle_mesh()
        if not surface.has_triangles():
            raise DataError("Sparse TSDF did not extract an observed surface")
        surface.compute_vertex_normals()
        vertices = np.asarray(surface.vertices, dtype=np.float64).copy()
        faces = np.asarray(surface.triangles, dtype=np.int32).copy()
        normals = np.asarray(surface.vertex_normals, dtype=np.float32).copy()
        rgb = np.asarray(surface.vertex_colors, dtype=np.float64)
        if rgb.shape != vertices.shape or not np.isfinite(rgb).all():
            raise DataError("Open3D did not return registered vertex RGB")
        if (len(vertices) > 300000 or len(faces) > 500000):
            raise DataError("Sparse Mesh exceeds the bounded 3+2 review renderer")
        colors = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        # Mesh.observations is required for validation but NOT exported in PLY;
        # these ones are placeholders, not an estimate of observation support.
        mesh = Mesh(vertices, faces, normals, colors,
                    np.ones(len(vertices), dtype=np.uint32),
                    np.zeros(len(vertices), dtype=np.int32),
                    np.full(len(vertices), count - 1, dtype=np.int32))
        write_ply(output / "mesh.ply", mesh,
                  comment="Open3D sparse TSDF; measured RGB-D; known poses; no inferred depth")
        result = {
            "provider": "open3d-scalable-tsdf-0.19.0",
            "source": recording.source,
            "camera_resolution": [camera.width, camera.height],
            "frames_integrated": count, "valid_depth_pixels": valid_pixels,
            "vertices": len(vertices), "triangles": len(faces),
            "voxel_m": voxel_m, "truncation_m": truncation_m,
            "max_depth_m": max_depth_m, "pixel_stride": pixel_stride,
            "edge_guard_m": edge_guard_m,
            "edge_halo_pixels": edge_halo_pixels,
            "edge_removed_pixels": edge_removed_pixels,
            "consistency_guard_m": consistency_guard_m,
            "consistency_removed_pixels": cross_removed_pixels,
            "consistency_witness_counts": [len(x) for x in neighbors],
            "consistency_policy": "two farther stable training views, zero agreements; no hole fill",
            "seconds": round(time.perf_counter() - start, 3),
            "input_manifest_sha256": recording.manifest_sha256,
            "mesh_sha256": file_sha256(output / "mesh.ply"),
            "geometry_source": "observed metric sensor depth and known camera poses",
            "vertex_observation_support": "not measured by Open3D exporter",
            "unknown_surface_fill": False,
        }
        write_json(output / "report.json", result)
    return result
