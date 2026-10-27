"""Reproducible real-TUM Mesh-only 3+2 review renders; never publishes source RGB.

Uses the SAME frozen 48-frame Mesh, three fixed withheld camera poses and two
explicit free cameras. Greyscale pixels encode metric camera Z, not texture.
No geometry quality or human recognition claim follows from image production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from echo4d.holdout import rasterize_mesh_depth, read_echo_ply
from echo4d.model import Camera, DataError, file_sha256, load_recording, rigid_matrix, safe_path
from echo4d.output import output_transaction, write_json
from tum_composable_graph import HOLDOUT, TOTAL
from tum_public_roundtrip import PINNED_ARCHIVE_SHA256

VIEWS = (4, 24, 55)  # frozen before rendering; all withheld
FAR_M = 20.0
NEAR_M = 0.15


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Right-handed camera-to-world; camera axes right/down/forward."""
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    if eye.shape != (3,) or target.shape != (3,) or not np.isfinite(eye).all() or not np.isfinite(target).all():
        raise DataError("Free camera coordinates must be finite 3-vectors")
    forward = target - eye
    if np.linalg.norm(forward) < 1e-6:
        raise DataError("Free camera eye coincides with target")
    forward /= np.linalg.norm(forward)
    up = np.array([0., 0., 1.])
    if abs(float(forward @ up)) > 0.98:
        up = np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = eye
    return rigid_matrix(pose.flatten(order="F"))


def depth_image(depth: np.ndarray) -> Image.Image:
    """Fixed metric mapping, black means no projected surface; no auto contrast."""
    d = np.asarray(depth)
    if d.ndim != 2 or not np.isfinite(d).all() or np.any(d < 0):
        raise DataError("Invalid rendered depth")
    valid = (d >= NEAR_M) & (d <= FAR_M)
    pixels = np.zeros(d.shape, dtype=np.uint8)
    pixels[valid] = np.clip(np.rint(255 - (d[valid] - NEAR_M) * 254 / (FAR_M - NEAR_M)), 1, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L")



def read_vertex_rgb(path: Path, vertex_count: int) -> np.ndarray:
    """Recover training-derived RGB from an already strict-validated ECHO PLY."""
    with Path(path).open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\\n"):
            line = stream.readline()
            if not line or len(header) + len(line) > 4096:
                raise DataError("Invalid bounded ECHO PLY header")
            header.extend(line)
        dtype = np.dtype([(k, "<f4") for k in ("x", "y", "z", "nx", "ny", "nz")]
                         + [(k, "u1") for k in ("red", "green", "blue")])
        values = np.fromfile(stream, dtype=dtype, count=vertex_count)
    if len(values) != vertex_count:
        raise DataError("Truncated Mesh vertex colors")
    return np.column_stack([values[k] for k in ("red", "green", "blue")])


def rasterize_vertex_rgb(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray,
                         camera: Camera, camera_to_world: np.ndarray,
                         near: float = NEAR_M, far: float = FAR_M) -> tuple[np.ndarray, int]:
    """Conservative triangle z-buffer with perspective-correct vertex RGB.

    No texture from held-out RGB, no hole filling, no claim of photo accuracy.
    """
    points, triangles, rgb = np.asarray(vertices), np.asarray(faces), np.asarray(colors)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not 3 <= len(points) <= 300000
            or not np.isfinite(points).all() or triangles.ndim != 2
            or triangles.shape[1:] != (3,) or triangles.dtype.kind not in "iu"
            or not 1 <= len(triangles) <= 500000
            or np.any(triangles < 0) or np.any(triangles >= len(points))
            or rgb.shape != (len(points), 3) or rgb.dtype != np.uint8
            or not 0 < near < far <= 100 or not np.isfinite([near, far]).all()):
        raise DataError("Invalid bounded Mesh RGB render inputs")
    pose = np.asarray(camera_to_world, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise DataError("Invalid RGB render pose")
    pose = rigid_matrix(pose.flatten(order="F"))
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    zbuffer = np.full((camera.height, camera.width), np.inf)
    image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    for face in triangles:
        p = local[face]
        if np.any(p[:, 2] <= near):
            continue  # no clipping/invented near-plane surfaces
        uv = np.column_stack((camera.fx * p[:, 0] / p[:, 2] + camera.cx,
                              camera.fy * p[:, 1] / p[:, 2] + camera.cy))
        x0 = max(0, int(np.ceil(np.min(uv[:, 0]) - 1e-9)))
        x1 = min(camera.width - 1, int(np.floor(np.max(uv[:, 0]) + 1e-9)))
        y0 = max(0, int(np.ceil(np.min(uv[:, 1]) - 1e-9)))
        y1 = min(camera.height - 1, int(np.floor(np.max(uv[:, 1]) + 1e-9)))
        if x0 > x1 or y0 > y1:
            continue
        a, b, c = uv
        denominator = ((b[1] - c[1]) * (a[0] - c[0])
                       + (c[0] - b[0]) * (a[1] - c[1]))
        if abs(denominator) < 1e-11:
            continue
        xx, yy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0 = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        w2 = 1. - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        inv_z = w0 / p[0, 2] + w1 / p[1, 2] + w2 / p[2, 2]
        valid = inside & (inv_z > 0) & (inv_z >= 1. / far)
        if not np.any(valid):
            continue
        depth = 1. / np.maximum(inv_z, 1e-100)
        region = zbuffer[y0:y1 + 1, x0:x1 + 1]
        closer = valid & (depth < region)
        if not np.any(closer):
            continue
        weighted = (w0[..., None] * rgb[face[0]] / p[0, 2]
                    + w1[..., None] * rgb[face[1]] / p[1, 2]
                    + w2[..., None] * rgb[face[2]] / p[2, 2])
        region[closer] = depth[closer]
        image[y0:y1 + 1, x0:x1 + 1][closer] = np.clip(
            np.rint(weighted[closer] / inv_z[closer, None]), 0, 255).astype(np.uint8)
    return image, int(np.count_nonzero(np.isfinite(zbuffer)))


def verify_original_rgb(dataset_root: Path, association: dict, recording, index: int) -> dict:
    """Bind a 640x480 TUM RGB PNG to the actual imported 160x120 pixels."""
    entry = association["frames"][index]
    if entry.get("index") != index or not isinstance(entry.get("rgb_path"), str):
        raise DataError("Association index or source RGB path mismatch")
    source = safe_path(dataset_root, entry["rgb_path"])
    original_sha = file_sha256(source)
    if original_sha != entry.get("rgb_sha256"):
        raise DataError("Original RGB differs from importer association SHA")
    with Image.open(source) as image:
        if image.size != (640, 480):
            raise DataError("Original RGB is not the expected 640x480")
        original = np.asarray(image.convert("RGB"))
    imported = safe_path(recording.root, recording.entries[index]["color"])
    imported_sha = file_sha256(imported)
    with Image.open(imported) as image:
        if image.size != (160, 120):
            raise DataError("Imported RGB is not 160x120")
        sampled = np.asarray(image.convert("RGB"))
    if not np.array_equal(original[::4, ::4], sampled):
        raise DataError("Imported RGB pixels differ from 4x source sampling")
    if file_sha256(source) != original_sha or file_sha256(imported) != imported_sha:
        raise DataError("RGB source changed during review preparation")
    return {"source_relative_path": entry["rgb_path"], "source_sha256": original_sha,
            "source_resolution": [640, 480], "imported_sha256": imported_sha,
            "imported_resolution": [160, 120], "sampling": "source[::4,::4], no interpolation",
            "source_redistributed": False}


def render(args: argparse.Namespace) -> dict:
    full_path, train_path, mesh_path = (p.resolve() for p in
                                        (args.full_manifest, args.train_manifest, args.mesh))
    full, train = load_recording(full_path), load_recording(train_path)
    if (len(full) != TOTAL or len(train) != TOTAL - len(HOLDOUT)
            or full.source != "recorded" or train.source != "recorded"
            or full.camera.width != 160 or full.camera.height != 120
            or full.camera != train.camera or full.depth_scale != train.depth_scale
            or [f for i, f in enumerate(full.entries) if i not in HOLDOUT] != train.entries):
        raise DataError("Frozen real 60/48 source and train split required")
    report = json.loads(args.report.read_text(encoding="utf8"))
    execution = json.loads(args.execution.read_text(encoding="utf8"))
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf8"))
    association = json.loads(args.association.read_text(encoding="utf8"))
    mesh_sha = file_sha256(mesh_path)
    if (report.get("status") != "executed_unscored"
            or report.get("source", {}).get("archive_sha256") != PINNED_ARCHIVE_SHA256
            or report.get("input", {}).get("original_manifest_sha256") != file_sha256(full_path)
            or report.get("input", {}).get("train_manifest_sha256") != file_sha256(train_path)
            or report.get("input", {}).get("association_sha256") != file_sha256(args.association)
            or report.get("mesh_sha256") != mesh_sha
            or execution.get("status") != "executed_unscored"
            or diagnostic.get("status") != "diagnostic_only"
            or diagnostic.get("mesh_sha256") != mesh_sha
            or diagnostic.get("source_full_manifest_sha256") != file_sha256(full_path)
            or diagnostic.get("source_train_manifest_sha256") != file_sha256(train_path)
            or diagnostic.get("diagnostic", {}).get("heldout_indices") != list(HOLDOUT)
            or association.get("version") != "echo-tum-association-1"
            or len(association.get("frames", [])) != TOTAL):
        raise DataError("Missing or inconsistent frozen real graph, source or diagnostic receipts")
    target_port = execution.get("target")
    if (not isinstance(target_port, str)
            or execution.get("outputs", {}).get(target_port, {}).get("sha256") != mesh_sha):
        raise DataError("Graph receipt does not pin this Mesh")
    vertices, faces = read_echo_ply(mesh_path)
    vertex_rgb = read_vertex_rgb(mesh_path, len(vertices))
    camera = Camera(640, 480, full.camera.fx * 4, full.camera.fy * 4,
                    full.camera.cx * 4, full.camera.cy * 4)
    if not np.isfinite(vertices).all() or not set(VIEWS).issubset(set(HOLDOUT)):
        raise DataError("Invalid geometry or predeclared withheld view selection")
    views = []
    with output_transaction(args.output) as output:
        for index in VIEWS:
            rgb = verify_original_rgb(args.dataset_root, association, full, index)
            pose = rigid_matrix(full.entries[index]["cameraToWorld"])
            depth, stats = rasterize_mesh_depth(vertices, faces, camera, pose,
                                                 near=NEAR_M, far=FAR_M)
            name = f"heldout-{index:02d}-mesh-depth.png"
            depth_image(depth).save(output / name)
            color_image, color_pixels = rasterize_vertex_rgb(vertices, faces, vertex_rgb, camera, pose)
            if color_pixels != stats["predicted_pixels"]:
                raise DataError("RGB and depth Mesh coverage disagree")
            color_name = f"heldout-{index:02d}-mesh-vertex-rgb.png"
            Image.fromarray(color_image, mode="RGB").save(output / color_name)
            views.append({"kind": "heldout_original_camera",
                          "vertex_rgb_render": color_name,
                          "vertex_rgb_sha256": file_sha256(output / color_name), "frame_index": index,
                          "t_s": full.entries[index]["t"], "rgb": rgb,
                          "camera_to_world_column_major": pose.flatten(order="F").tolist(),
                          "render": name, "render_sha256": file_sha256(output / name),
                          "predicted_pixels": stats["predicted_pixels"],
                          "render_resolution": [640, 480]})
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        span = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 0.5)
        for j, offset in enumerate(((0.8, -1.0, 0.55), (-0.9, -0.8, 0.65)), 1):
            pose = look_at(center + span * np.asarray(offset), center)
            depth, stats = rasterize_mesh_depth(vertices, faces, camera, pose,
                                                 near=NEAR_M, far=FAR_M)
            name = f"free-{j}-mesh-depth.png"
            depth_image(depth).save(output / name)
            color_image, color_pixels = rasterize_vertex_rgb(vertices, faces, vertex_rgb, camera, pose)
            if color_pixels != stats["predicted_pixels"]:
                raise DataError("RGB and depth Mesh coverage disagree")
            color_name = f"free-{j}-mesh-vertex-rgb.png"
            Image.fromarray(color_image, mode="RGB").save(output / color_name)
            views.append({"kind": "free_third_person",
                          "vertex_rgb_render": color_name,
                          "vertex_rgb_sha256": file_sha256(output / color_name), "camera_to_world_column_major":
                          pose.flatten(order="F").tolist(), "target_world_m": center.tolist(),
                          "render": name, "render_sha256": file_sha256(output / name),
                          "predicted_pixels": stats["predicted_pixels"],
                          "render_resolution": [640, 480]})
        if file_sha256(mesh_path) != mesh_sha:
            raise DataError("Mesh changed while rendering")
        receipt = {"schema": "echo4d-real-mesh-visual-1", "status": "rendered_unreviewed",
                   "source_archive_sha256": PINNED_ARCHIVE_SHA256,
                   "full_manifest_sha256": file_sha256(full_path),
                   "train_manifest_sha256": file_sha256(train_path),
                   "mesh_sha256": mesh_sha, "graph_receipt_sha256": file_sha256(args.execution),
                   "holdout_diagnostic_sha256": file_sha256(args.diagnostic),
                   "render_type": "fixed-scale camera-Z grayscale plus z-buffered Mesh training-vertex RGB; not photorealistic",
                   "color_source": "training 160x120 imported RGB embedded in Mesh; heldout RGB not used for texture",
                   "unknown_pixels": "black; not inferred surfaces",
                   "source_original_rgb": "hash-verified and locally compared; not redistributed",
                   "views": views, "human_visual_review": False,
                   "independent_surface_truth": False, "issue_74": "FAIL / NOT ACCEPTED"}
        write_json(output / "visual-receipt.json", receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("full-manifest", "train-manifest", "mesh", "execution",
                "report", "diagnostic", "association", "dataset-root", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_name(args.output.name + ".failed.json").exists():
        parser.error("Output or previous failure receipt already exists")
    try:
        result = render(args)
    except Exception as error:
        failure = {"schema": "echo4d-real-mesh-visual-1",
                   "status": "blocked" if isinstance(error, (DataError, OSError)) else "failed",
                   "error_type": type(error).__name__, "error": str(error),
                   "human_visual_review": False, "issue_74": "FAIL / NOT ACCEPTED"}
        write_json(args.output.with_name(args.output.name + ".failed.json"), failure)
        print(json.dumps(failure))
        return 1
    print(json.dumps({"status": result["status"], "views": len(result["views"]),
                      "mesh_sha256": result["mesh_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
