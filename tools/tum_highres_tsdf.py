"""Pinned real TUM fr1_xyz: compare 160x120 / 5cm TSDF against 320x240 / 2.5cm TSDF.

Both use identical original RGB-D frames, known TUM poses, 48 training views and
12 untouched sensor holdouts. No independent surface ground truth is claimed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from echo4d.holdout import evaluate_frames, read_echo_ply
from echo4d.model import DataError, file_sha256, load_recording, safe_path
from tum_composable_graph import HOLDOUT, TOTAL, split_training
from tum_graph_visual import look_at, rasterize_vertex_rgb, read_vertex_rgb
from tum_public_roundtrip import (BOUNDS, PINNED_ARCHIVE_BYTES,
    PINNED_ARCHIVE_SHA256, ROOT, URL, download, extract)

# Predeclared before reading any candidate holdout metrics.
CONFIGS = {
    "baseline-160": {"downsample": 4, "voxel_m": 0.05},
    "candidate-320": {"downsample": 2, "voxel_m": 0.035},
}
VIEWS = (4, 24, 55)
MAX_VOXELS = 4_000_000
MAX_DEPTH_M = 4.0


def invoke(label: str, argv: list[str], report: dict, timeout: int = 1200) -> None:
    item = {"label": label, "argv": argv, "status": "started"}
    report.setdefault("commands", []).append(item)
    start = time.perf_counter()
    try:
        proc = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        item.update(returncode=proc.returncode, stdout_tail=proc.stdout[-1000:],
                    stderr_tail=proc.stderr[-2000:])
        if proc.returncode:
            raise RuntimeError(f"{label} exit {proc.returncode}: {proc.stderr[-800:]}")
        item["status"] = "passed"
    except Exception:
        item["status"] = "failed"
        raise
    finally:
        item["seconds"] = round(time.perf_counter() - start, 3)


def verify_same_source(low_dir: Path, high_dir: Path) -> dict:
    low, high = (load_recording(p / "recording.json") for p in (low_dir, high_dir))
    if (len(low) != TOTAL or len(high) != TOTAL or
            low.source != "recorded" or high.source != "recorded" or
            (low.camera.width, low.camera.height) != (160, 120) or
            (high.camera.width, high.camera.height) != (320, 240) or
            low.camera != high.camera.scaled(2)):
        raise DataError("Expected 60 matched 160/320 real frames and calibrated intrinsics")
    a = json.loads((low_dir / "association.json").read_text(encoding="utf8"))
    b = json.loads((high_dir / "association.json").read_text(encoding="utf8"))
    if a != b or len(a.get("frames", [])) != TOTAL:
        raise DataError("Different original RGB/depth/pose associations")
    for left, right in zip(low.entries, high.entries):
        if left["t"] != right["t"] or not np.allclose(
                left["cameraToWorld"], right["cameraToWorld"], rtol=0, atol=1e-10):
            raise DataError("Source timestamp or camera pose changed")
    return {"same_source_associations": True, "same_poses_and_timestamps": True,
            "original_resolution": [640, 480], "baseline_resolution": [160, 120],
            "candidate_resolution": [320, 240],
            "baseline_intrinsics": low.camera.to_dict(),
            "candidate_intrinsics": high.camera.to_dict(),
            "association_sha256": file_sha256(low_dir / "association.json")}


def training_manifest(source: Path) -> None:
    target = source / "recording-train.json"
    target.write_text(json.dumps(split_training(json.loads(
        (source / "recording.json").read_text(encoding="utf8"))),
        sort_keys=True, allow_nan=False) + "\n", encoding="utf8")
    if len(load_recording(target)) != TOTAL - len(HOLDOUT):
        raise DataError("Expected exactly 48 training frames")


def face_components(vertex_count: int, faces: np.ndarray) -> dict:
    """Actual mesh vertex-sharing components, not cosmetic deletion or hole fill."""
    if vertex_count < 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise DataError("Invalid mesh for connectivity analysis")
    parent = np.arange(vertex_count, dtype=np.int32)
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    for a, b, c in faces:
        root = find(int(a))
        for index in (int(b), int(c)):
            other = find(index)
            parent[other] = root
    labels = np.asarray([find(i) for i in range(vertex_count)], dtype=np.int32)
    _, inverse = np.unique(labels[faces[:, 0]], return_inverse=True)
    counts = np.bincount(inverse)
    return {"face_components": int(len(counts)),
            "largest_component_face_fraction": float(counts.max() / len(faces)),
            "largest_component_faces": int(counts.max())}


def reconstruct(source: Path, target: Path, voxel: float, report: dict) -> None:
    invoke("fuse-" + source.name, [sys.executable, "-m", "echo4d", "reconstruct",
        str(source / "recording-train.json"), "--output", str(target),
        "--bounds", *BOUNDS, "--voxel", str(voxel),
        "--max-voxels", str(MAX_VOXELS), "--max-depth", str(MAX_DEPTH_M),
        "--min-observations", "2", "--quiet", "--no-scene"], report)
    if not (target / "mesh.ply").is_file():
        raise DataError("Reconstruction did not produce a real Mesh")


def contact_sheet(original: Image.Image | None, low: np.ndarray, high: np.ndarray,
                  output: Path, title: str) -> None:
    if any(x.shape != (240, 320, 3) or x.dtype != np.uint8 for x in (low, high)):
        raise DataError("Actual Mesh RGB renders must be 320x240")
    sheet = Image.new("RGB", (1920, 514), (22, 26, 32))
    draw = ImageDraw.Draw(sheet)
    if original is not None:
        sheet.paste(original.convert("RGB"), (0, 34))
    for x, image in ((640, low), (1280, high)):
        sheet.paste(Image.fromarray(image).resize((640, 480),
                    resample=Image.Resampling.NEAREST), (x, 34))
    for x, caption in ((0, title + " | original 640x480" if original else title + " | no original"),
                       (640, "TSDF 160x120 / 5cm"), (1280, "TSDF 320x240 / 2.5cm")):
        draw.text((x + 8, 10), caption, fill=(238, 238, 238))
    sheet.save(output)


def render_views(dataset: Path, low_dir: Path, high_dir: Path,
                 outputs: dict[str, Path], report_dir: Path) -> list[dict]:
    meshes = {}
    for key, folder in outputs.items():
        path = folder / "mesh.ply"
        vertices, faces = read_echo_ply(path)
        meshes[key] = (vertices, faces, read_vertex_rgb(path, len(vertices)))
    low, high = (load_recording(p / "recording.json") for p in (low_dir, high_dir))
    assoc = json.loads((low_dir / "association.json").read_text(encoding="utf8"))
    visual = report_dir / "visual"
    visual.mkdir(exist_ok=True)
    results = []
    for index in VIEWS:
        pose = np.asarray(low.entries[index]["cameraToWorld"]).reshape((4, 4), order="F")
        spec = assoc["frames"][index]
        source = safe_path(dataset, spec["rgb_path"])
        if file_sha256(source) != spec["rgb_sha256"]:
            raise DataError("Original RGB source SHA changed")
        with Image.open(source) as im:
            if im.size != (640, 480):
                raise DataError("Source RGB is not 640x480")
            original = im.convert("RGB").copy()
        images = {}
        for key, (verts, faces, rgb) in meshes.items():
            images[key], _ = rasterize_vertex_rgb(
                verts, faces, rgb, high.camera, pose, near=.15, far=4.)
        name = f"heldout-{index:02d}-original-vs-mesh.png"
        contact_sheet(original, images["baseline-160"], images["candidate-320"],
                      visual / name, f"TUM withheld {index}")
        results.append({"kind": "original_camera", "index": index, "image": name,
                        "sha256": file_sha256(visual / name),
                        "source_rgb_sha256": spec["rgb_sha256"],
                        "holdout_rgb_used_to_color_mesh": False})
    vertices = meshes["candidate-320"][0]
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    span = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), .5)
    for j, offset in enumerate(((.8, -1., .55), (-.9, -.8, .65)), 1):
        pose = look_at(center + span * np.asarray(offset), center)
        images = {}
        for key, (verts, faces, rgb) in meshes.items():
            images[key], _ = rasterize_vertex_rgb(
                verts, faces, rgb, high.camera, pose, near=.15, far=8.)
        name = f"free-{j}-two-real-meshes.png"
        contact_sheet(None, images["baseline-160"], images["candidate-320"],
                      visual / name, f"Free view {j}")
        results.append({"kind": "free_camera", "image": name,
                        "sha256": file_sha256(visual / name),
                        "camera_to_world": pose.flatten(order="F").tolist()})
    return results


def experiment(args: argparse.Namespace, report: dict) -> None:
    work, target = args.work.resolve(), args.report_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / "source.tgz"
    if args.archive:
        source = args.archive.resolve()
        if source.stat().st_size != PINNED_ARCHIVE_BYTES:
            raise DataError("Offline source has wrong byte count")
        shutil.copyfile(source, archive)
        report["download"] = {"bytes": archive.stat().st_size,
                              "sha256": file_sha256(archive), "method": "offline"}
    else:
        download(archive, report)
    if (report["download"]["bytes"] != PINNED_ARCHIVE_BYTES or
            report["download"]["sha256"] != PINNED_ARCHIVE_SHA256):
        raise DataError("TUM archive differs from pinned source")
    report["extraction"] = extract(archive, work / "dataset")
    dataset = work / "dataset" / ROOT
    inputs, outputs = {}, {}
    for key, cfg in CONFIGS.items():
        source = work / key
        invoke("import-" + key, [sys.executable, "-m", "echo4d", "import-tum",
            str(dataset), "--output", str(source),
            "--intrinsics", "525", "525", "319.5", "239.5",
            "--stride", "12", "--limit", "60",
            "--downsample", str(cfg["downsample"]), "--world-up", "z"], report, 600)
        training_manifest(source)
        inputs[key] = source
    report["matched_inputs"] = verify_same_source(inputs["baseline-160"], inputs["candidate-320"])
    for key, cfg in CONFIGS.items():
        output = target / key
        reconstruct(inputs[key], output, cfg["voxel_m"], report)
        outputs[key] = output
        verts, faces = read_echo_ply(output / "mesh.ply")
        # Both candidates scored against identical frozen 160x120 sensor pixels.
        diagnostic = evaluate_frames(verts, faces,
            load_recording(inputs["baseline-160"] / "recording.json"),
            list(HOLDOUT), max_depth=MAX_DEPTH_M)
        report.setdefault("results", {})[key] = {
            "mesh_sha256": file_sha256(output / "mesh.ply"),
            "mesh_vertices": len(verts), "mesh_faces": len(faces),
            "connectivity": face_components(len(verts), faces),
            "reconstruction_report_sha256": file_sha256(output / "report.json"),
            "training_manifest_sha256": file_sha256(inputs[key] / "recording-train.json"),
            "same_sensor_holdout_diagnostic_only": {k: diagnostic[k] for k in
                ("heldout_frames", "observed_depth_coverage", "rmse_m", "mae_m", "p95_abs_m")}}
    report["visual"] = render_views(dataset, inputs["baseline-160"],
                                    inputs["candidate-320"], outputs, target)
    report["status"] = "real_meshes_rendered_unreviewed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {"experiment": "TUM-fr1_xyz-highres-TSDF-160-vs-320-v2",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "checkout_sha": os.environ.get("GITHUB_SHA"),
              "dataset_url": URL, "dataset_sha256": PINNED_ARCHIVE_SHA256,
              "original_rgb_resolution": [640, 480],
              "frozen_holdout_indices": list(HOLDOUT), "configurations": CONFIGS,
              "research_note": "35mm candidate selected after reviewing 25mm same-scene results; exploratory only, not fresh validation",
              "desk_v2": "FAIL 16.7779cm > 15cm; unchanged",
              "issue_74": "FAIL / NOT ACCEPTED", "status": "started",
              "commands": [], "independent_surface_truth": "not_available",
              "human_visual_review": "not_performed"}
    started = time.perf_counter()
    try:
        experiment(args, report)
        code = 0
    except Exception as exc:
        code = 1
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"HIGHRES_REAL_EXPERIMENT_FAILED: {exc}", file=sys.stderr)
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        (args.report_dir / "quality-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf8")
        (args.report_dir / "SUMMARY.md").write_text("\n".join([
            "# Real TUM 160 versus 320 TSDF quality experiment", "",
            f"- Status: {report['status']}",
            "- Original RGB: 640x480; fused: 160x120 versus 320x240",
            f"- Git checkout: {report['checkout_sha']}",
            f"- Meshes: {list(report.get('results', {}))}",
            f"- Three source-aligned plus two free views: {len(report.get('visual', []))}",
            "- Same-sensor heldout is NOT independent surface truth.",
            "- No claim of #74 visual PASS.",
            f"- Failure: {report.get('failure', {}).get('message', 'none')}"]) + "\n",
            encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
