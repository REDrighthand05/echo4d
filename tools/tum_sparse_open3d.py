"""Real TUM fr1_xyz: dense 320 RGB-D vs Open3D sparse 640 RGB-D.

48 identical known-pose training frames, 12 frozen sensor-depth holdouts,
three aligned original RGB views and two free-camera views of actual colored
Mesh triangles. The sparse result is an EXPERIMENT, not an accepted improvement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from echo4d.holdout import evaluate_frames, read_echo_ply
from echo4d.model import DataError, file_sha256, load_recording, safe_path
from sparse_open3d_fusion import fuse_sparse
from tum_composable_graph import HOLDOUT, TOTAL
from tum_graph_visual import look_at, rasterize_vertex_rgb, read_vertex_rgb
from tum_highres_tsdf import face_components, invoke, reconstruct, training_manifest
from tum_public_roundtrip import (BOUNDS, PINNED_ARCHIVE_BYTES,
    PINNED_ARCHIVE_SHA256, ROOT, URL, download, extract)

VIEWS = (4, 24, 55)
DENSE_VOXEL_M = 0.035
SPARSE_VOXEL_M = 0.03
SPARSE_TRUNCATION_M = 0.09
MAX_DEPTH_M = 4.0


def verify_inputs(dense_dir: Path, sparse_dir: Path) -> dict:
    dense, sparse = (load_recording(p / "recording.json")
                     for p in (dense_dir, sparse_dir))
    if (len(dense) != TOTAL or len(sparse) != TOTAL
            or dense.source != "recorded" or sparse.source != "recorded"
            or (dense.camera.width, dense.camera.height) != (320, 240)
            or (sparse.camera.width, sparse.camera.height) != (640, 480)
            or sparse.camera.scaled(2) != dense.camera):
        raise DataError("Expected 60 matched calibrated 320 and 640 real RGB-D frames")
    a = json.loads((dense_dir / "association.json").read_text(encoding="utf8"))
    b = json.loads((sparse_dir / "association.json").read_text(encoding="utf8"))
    if a != b or len(a.get("frames", [])) != TOTAL:
        raise DataError("The original TUM source associations differ")
    for x, y in zip(dense.entries, sparse.entries):
        if x["t"] != y["t"] or not np.allclose(
                x["cameraToWorld"], y["cameraToWorld"], atol=1e-10, rtol=0):
            raise DataError("The two routes use different camera poses or timestamps")
    return {"same_original_associations": True, "same_poses_and_timestamps": True,
            "original_resolution": [640, 480],
            "dense_resolution": [320, 240], "sparse_resolution": [640, 480],
            "dense_intrinsics": dense.camera.to_dict(),
            "sparse_intrinsics": sparse.camera.to_dict()}


def comparison_sheet(original: Image.Image | None, dense: np.ndarray,
                     sparse: np.ndarray, path: Path, title: str) -> None:
    if any(x.shape != (240, 320, 3) or x.dtype != np.uint8 for x in (dense, sparse)):
        raise DataError("Both real Mesh renders must be 320x240 uint8 RGB")
    sheet = Image.new("RGB", (1920, 514), (22, 26, 32))
    draw = ImageDraw.Draw(sheet)
    if original is not None:
        if original.size != (640, 480):
            raise DataError("Original RGB must be 640x480")
        sheet.paste(original.convert("RGB"), (0, 34))
    for x, pixels in ((640, dense), (1280, sparse)):
        sheet.paste(Image.fromarray(pixels).resize(
            (640, 480), resample=Image.Resampling.NEAREST), (x, 34))
    labels = (title + (" | original RGB" if original else " | free camera"),
              "Dense 320x240 / 35mm", "Open3D sparse 640x480 / 30mm")
    for x, label in zip((0, 640, 1280), labels):
        draw.text((x + 8, 10), label, fill=(238, 238, 238))
    sheet.save(path)


def render_five(dataset: Path, dense_dir: Path, output: Path) -> list[dict]:
    dense = load_recording(dense_dir / "recording.json")
    assoc = json.loads((dense_dir / "association.json").read_text(encoding="utf8"))
    meshes = {}
    for key in ("dense-320", "sparse-640"):
        path = output / key / "mesh.ply"
        vertices, faces = read_echo_ply(path)
        meshes[key] = (vertices, faces, read_vertex_rgb(path, len(vertices)))
    visual = output / "visual"
    visual.mkdir()
    results = []
    for index in VIEWS:
        pose = np.asarray(dense.entries[index]["cameraToWorld"]).reshape(
            4, 4, order="F")
        source_spec = assoc["frames"][index]
        rgb_path = safe_path(dataset, source_spec["rgb_path"])
        if file_sha256(rgb_path) != source_spec["rgb_sha256"]:
            raise DataError("Original source RGB changed before visual comparison")
        with Image.open(rgb_path) as img:
            original = img.convert("RGB").copy()
        renders = {}
        for key, (v, f, rgb) in meshes.items():
            renders[key], _ = rasterize_vertex_rgb(
                v, f, rgb, dense.camera, pose, near=.15, far=MAX_DEPTH_M)
        name = f"heldout-{index:02d}-source-dense-sparse.png"
        comparison_sheet(original, renders["dense-320"], renders["sparse-640"],
                         visual / name, f"Withheld TUM camera {index}")
        results.append({"kind": "original_camera", "index": index, "image": name,
                        "sha256": file_sha256(visual / name),
                        "source_rgb_sha256": source_spec["rgb_sha256"],
                        "holdout_rgb_used_for_fusion": False})
    vertices = meshes["sparse-640"][0]
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    span = max(float(np.max(np.ptp(vertices, axis=0))), 0.5)
    for j, offset in enumerate(((0.45, -1.25, 0.3), (1.2, 0.35, 0.45)), 1):
        pose = look_at(center + span * np.asarray(offset), center)
        renders = {}
        for key, (v, f, rgb) in meshes.items():
            renders[key], _ = rasterize_vertex_rgb(
                v, f, rgb, dense.camera, pose, near=.15, far=10.)
        name = f"free-{j}-dense-sparse.png"
        comparison_sheet(None, renders["dense-320"], renders["sparse-640"],
                         visual / name, f"Free viewpoint {j}")
        results.append({"kind": "free_camera", "image": name,
                        "sha256": file_sha256(visual / name),
                        "camera_to_world": pose.flatten(order="F").tolist()})
    return results


def experiment(args: argparse.Namespace, report: dict) -> None:
    work, target = args.work.resolve(), args.report_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / "source.tgz"
    if args.archive:
        if args.archive.stat().st_size != PINNED_ARCHIVE_BYTES:
            raise DataError("Offline source has wrong byte count")
        shutil.copyfile(args.archive.resolve(), archive)
        report["download"] = {"bytes": archive.stat().st_size,
                              "sha256": file_sha256(archive), "method": "offline"}
    else:
        download(archive, report)
    if (report["download"]["bytes"] != PINNED_ARCHIVE_BYTES or
            report["download"]["sha256"] != PINNED_ARCHIVE_SHA256):
        raise DataError("TUM source differs from pinned real archive")
    report["extraction"] = extract(archive, work / "dataset")
    dataset = work / "dataset" / ROOT
    inputs = {}
    for key, factor in (("dense-320", 2), ("sparse-640", 1)):
        folder = work / key
        invoke("import-" + key, [sys.executable, "-m", "echo4d", "import-tum",
            str(dataset), "--output", str(folder),
            "--intrinsics", "525", "525", "319.5", "239.5",
            "--stride", "12", "--limit", "60", "--downsample", str(factor),
            "--world-up", "z"], report, timeout=900)
        training_manifest(folder)
        inputs[key] = folder
    report["matched_inputs"] = verify_inputs(inputs["dense-320"], inputs["sparse-640"])
    reconstruct(inputs["dense-320"], target / "dense-320", DENSE_VOXEL_M, report)
    report["sparse_fusion"] = fuse_sparse(
        inputs["sparse-640"] / "recording-train.json", target / "sparse-640",
        voxel_m=SPARSE_VOXEL_M, truncation_m=SPARSE_TRUNCATION_M,
        max_depth_m=MAX_DEPTH_M, pixel_stride=2)
    for key in ("dense-320", "sparse-640"):
        path = target / key / "mesh.ply"
        vertices, faces = read_echo_ply(path)
        score = evaluate_frames(vertices, faces,
            load_recording(inputs["dense-320"] / "recording.json"),
            list(HOLDOUT), max_depth=MAX_DEPTH_M)
        report.setdefault("results", {})[key] = {
            "mesh_sha256": file_sha256(path),
            "vertices": len(vertices), "triangles": len(faces),
            "connectivity": face_components(len(vertices), faces),
            "same_sensor_holdout_diagnostic_only": {k: score[k] for k in
                ("heldout_frames", "observed_depth_coverage", "rmse_m", "mae_m", "p95_abs_m")}}
    report["visual"] = render_five(dataset, inputs["dense-320"], target)
    report["status"] = "two_real_meshes_rendered_visual_review_pending"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "TUM-fr1_xyz-dense320-vs-open3d-sparse640-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkout_sha": os.environ.get("GITHUB_SHA"),
        "dataset_url": URL, "dataset_sha256": PINNED_ARCHIVE_SHA256,
        "original_rgb_resolution": [640, 480],
        "training_frames": 48, "frozen_holdout_indices": list(HOLDOUT),
        "dense_voxel_m": DENSE_VOXEL_M, "sparse_voxel_m": SPARSE_VOXEL_M,
        "sparse_truncation_m": SPARSE_TRUNCATION_M,
        "comparison": "exploratory; 35mm dense selected after prior same-scene trials",
        "independent_surface_truth": "not_available",
        "human_visual_review": "not_performed",
        "desk_v2": "FAIL 16.7779cm > 15cm; unchanged",
        "issue_74": "FAIL / NOT ACCEPTED",
        "commands": [], "status": "started",
    }
    start = time.perf_counter()
    try:
        experiment(args, report)
        code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"SPARSE_REAL_EXPERIMENT_FAILED: {exc}", file=sys.stderr)
        code = 1
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        (args.report_dir / "sparse-quality-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
        (args.report_dir / "SUMMARY.md").write_text("\n".join([
            "# Real TUM dense 320 vs sparse Open3D 640 Mesh",
            f"- Status: {report['status']}",
            "- Original 640x480 RGB-D, known poses, 48 training / 12 holdout.",
            f"- Git SHA: {report['checkout_sha']}",
            f"- Mesh results: {list(report.get('results', {}))}",
            f"- Three original-camera plus two free-camera views: {len(report.get('visual', []))}",
            "- Same-sensor heldout is not independent surface truth.",
            "- #74 remains FAIL until a human verifies the geometry.",
            f"- Failure: {report.get('failure', {}).get('message', 'none')}"]) + "\n",
            encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
