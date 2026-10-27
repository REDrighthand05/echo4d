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
from echo4d.model import DataError, file_sha256, load_recording
from echo4d.output import output_transaction, write_json, write_ply


def validate_settings(voxel_m: float, truncation_m: float, max_depth_m: float,
                      pixel_stride: int) -> None:
    if not all(isinstance(v, (float, int)) and not isinstance(v, bool)
               and math.isfinite(v) for v in (voxel_m, truncation_m, max_depth_m)):
        raise DataError("Sparse TSDF distances must be finite numbers")
    if not 0.005 <= voxel_m <= 0.10:
        raise DataError("Sparse voxel must be between 5mm and 10cm")
    if not 2 * voxel_m <= truncation_m <= 0.5:
        raise DataError("Sparse truncation must be at least two voxels and at most 50cm")
    if not 0.2 < max_depth_m <= 10:
        raise DataError("Sparse maximum depth must be within (0.2,10] metres")
    if type(pixel_stride) is not int or pixel_stride not in (1, 2, 4, 8):
        raise DataError("Sparse depth pixel stride must be 1, 2, 4 or 8")


def fuse_sparse(recording_path: Path, destination: Path, *, voxel_m: float = 0.03,
                truncation_m: float = 0.09, max_depth_m: float = 4.0,
                pixel_stride: int = 2) -> dict:
    """Integrate registered RGB and camera-Z metric depth; output strict ECHO color PLY.

    Open3D expects world-to-camera extrinsics, whereas ECHO stores camera-to-world.
    Unobserved/excluded pixels are zeroed before integration. The exporter does not
    pretend to know per-vertex independent observation counts.
    """
    validate_settings(voxel_m, truncation_m, max_depth_m, pixel_stride)
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
    count, valid_pixels = 0, 0
    start = time.perf_counter()
    with output_transaction(destination) as output:
        for _, frame, _ in recording.iter_frames():
            if frame.color is None:
                raise DataError("Sparse colored TSDF requires registered training RGB")
            depth = np.ascontiguousarray(frame.depth, dtype=np.float32).copy()
            invalid = (depth < 0.15) | (depth > max_depth_m)
            if frame.excluded is not None:
                invalid |= frame.excluded
            depth[invalid] = 0
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
            "seconds": round(time.perf_counter() - start, 3),
            "input_manifest_sha256": recording.manifest_sha256,
            "mesh_sha256": file_sha256(output / "mesh.ply"),
            "geometry_source": "observed metric sensor depth and known camera poses",
            "vertex_observation_support": "not measured by Open3D exporter",
            "unknown_surface_fill": False,
        }
        write_json(output / "report.json", result)
    return result
