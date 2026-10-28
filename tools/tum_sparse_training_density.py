"""Real TUM fr1_xyz: does denser *training-view coverage* improve sparse RGB-D Mesh?

Compare the same 640x480 Open3D fusion at 48 frozen training views versus
additional interleaved source frames. Original 12 holdouts and their +/-0.24s
temporal neighborhoods are excluded from BOTH training manifests. No hole fill.
The 12 sensor-depth scores are exploratory and not independent surface truth.
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
from tum_composable_graph import HOLDOUT, TOTAL, split_training
from tum_graph_visual import look_at, rasterize_vertex_rgb, read_vertex_rgb
from tum_highres_tsdf import face_components, invoke
from tum_public_roundtrip import (
    PINNED_ARCHIVE_BYTES, PINNED_ARCHIVE_SHA256, ROOT, URL, download, extract,
)

VOXEL_M = 0.03
TRUNC_M = 0.09
MAX_DEPTH_M = 4.0
HOLDOUT_GUARD_SECONDS = 0.24
VIEWS = (4, 24, 55)


def select_extra_training(base: dict, expanded: dict, base_assoc: dict,
                          expanded_assoc: dict, guard_s: float = HOLDOUT_GUARD_SECONDS) -> tuple[dict, dict]:
    """Preserve all 48 baseline train frames and exclude nearby held-out times."""
    old, full = base.get("frames"), expanded.get("frames")
    a, b = base_assoc.get("frames"), expanded_assoc.get("frames")
    if not all(isinstance(v, list) for v in (old, full, a, b)):
        raise DataError("Missing real source frame arrays")
    if (len(old), len(full), len(a), len(b)) != (60, 120, 60, 120):
        raise DataError("Expected matched 60/120 frame source selections")
    if not 0.0 < guard_s <= 0.30:
        raise DataError("Invalid temporal holdout guard")
    for i in range(TOTAL):
        old_frame, full_frame = old[i], full[2 * i]
        if abs(float(old_frame["t"]) - float(full_frame["t"])) > 1e-8:
            raise DataError("Stride-6 even frames do not match stride-12 timestamps")
        if not np.allclose(old_frame["cameraToWorld"], full_frame["cameraToWorld"],
                           rtol=0, atol=1e-9):
            raise DataError("Stride-6 and stride-12 poses disagree")
        for key in ("rgb_path", "depth_path", "rgb_sha256", "depth_sha256"):
            if key not in a[i] or a[i][key] != b[2 * i].get(key):
                raise DataError("Stride-6 source association mismatch: " + key)
    frozen_times = [float(old[i]["t"]) for i in HOLDOUT]
    kept = [frame for frame in full if
            min(abs(float(frame["t"]) - t) for t in frozen_times) > guard_s]
    baseline = split_training(base)["frames"]
    selected_times = {float(f["t"]) for f in kept}
    if any(float(f["t"]) not in selected_times for f in baseline):
        raise DataError("Guard removed a baseline training frame; cannot compare fairly")
    if len(kept) <= len(baseline) or len(kept) > 108:
        raise DataError("Expanded training set did not add valid interleaved views")
    result = dict(expanded)
    result["frames"] = kept
    result["notes"] = ("TUM xyz extra interleaved measured RGB-D frames; original 12 "
                       "holdouts and +/-0.24s neighbors excluded; exploratory geometry")
    return result, {
        "baseline_train_frames": len(baseline),
        "expanded_train_frames": len(kept),
        "additional_frames": len(kept) - len(baseline),
        "frozen_holdout_indices_in_stride12": list(HOLDOUT),
        "holdout_guard_seconds": guard_s,
        "nearest_training_to_holdout_seconds": min(
            abs(float(f["t"]) - t) for f in kept for t in frozen_times),
        "baseline_train_preserved": True,
        "identical_original_source_associations": True,
    }



def select_pose_diverse_training(base: dict, dense: dict, base_assoc: dict,
                                 dense_assoc: dict, *, budget: int,
                                 guard_s: float = HOLDOUT_GUARD_SECONDS) -> tuple[dict, dict]:
    """Use the SAME frame count as uniform interleaving, but choose novel camera poses.

    Preserve all 48 frozen training frames. Extra source frames are chosen from
    stride-3 (rather than stride-6) TUM RGB-D by farthest-point camera-center
    and optical-axis distance. This is a measured-pose selection heuristic,
    NOT new surface observations, depth completion or a visual quality claim.
    """
    original, pool = base.get("frames"), dense.get("frames")
    a, b = base_assoc.get("frames"), dense_assoc.get("frames")
    if not all(isinstance(v, list) for v in (original, pool, a, b)):
        raise DataError("Missing TUM source frames or associations")
    if tuple(map(len, (original, pool, a, b))) != (60, 240, 60, 240):
        raise DataError("Expected matched 60/240 source selections")
    if type(budget) is not int or not 48 < budget <= 120:
        raise DataError("Invalid pose-diverse training frame budget")
    if not 0 < guard_s <= .30:
        raise DataError("Invalid temporal holdout guard")
    for i, frame in enumerate(original):
        other = pool[4 * i]
        if abs(float(frame["t"]) - float(other["t"])) > 1e-8:
            raise DataError("Stride-3 timestamps differ from frozen stride-12")
        if not np.allclose(frame["cameraToWorld"], other["cameraToWorld"],
                           rtol=0, atol=1e-9):
            raise DataError("Stride-3 poses differ from frozen stride-12")
        for key in ("rgb_path", "depth_path", "rgb_sha256", "depth_sha256"):
            if a[i].get(key) is None or a[i][key] != b[4 * i].get(key):
                raise DataError("Stride-3 source association mismatch: " + key)
    held = [float(original[i]["t"]) for i in HOLDOUT]
    allowed = [i for i, frame in enumerate(pool)
               if min(abs(float(frame["t"]) - t) for t in held) > guard_s]
    baseline = [4 * i for i in range(TOTAL) if i not in HOLDOUT]
    if not set(baseline).issubset(allowed):
        raise DataError("Temporal guard removed frozen baseline training")
    extras = [i for i in allowed if i not in set(baseline)]
    if len(extras) < budget - len(baseline):
        raise DataError("Not enough real extra camera poses outside holdout guard")

    matrices = np.stack([np.asarray(frame["cameraToWorld"], dtype=np.float64)
                         .reshape(4, 4, order="F") for frame in pool])
    if not np.isfinite(matrices).all():
        raise DataError("Nonfinite real camera poses")
    centers = matrices[:, :3, 3]
    directions = matrices[:, :3, 2]
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    # 0.35m per radian: optical-axis novelty and translation both matter.
    # This only prioritizes actual camera viewpoints; it never creates surfaces.
    distances = (np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
                 + .35 * np.arccos(np.clip(directions @ directions.T, -1., 1.)))
    chosen = list(baseline)
    nearest = distances[:, chosen].min(axis=1)
    for _ in range(budget - len(baseline)):
        winner = max(extras, key=lambda i: (float(nearest[i]), -i))
        chosen.append(winner)
        extras.remove(winner)
        nearest = np.minimum(nearest, distances[:, winner])
    chosen.sort()
    result = dict(dense)
    result["frames"] = [pool[i] for i in chosen]
    result["notes"] = ("Real TUM known-pose RGB-D; 48 frozen baseline + "
                       "pose-diverse extra frames; 12 holdouts +/-0.24s excluded")
    return result, {
        "method": "farthest camera center + optical axis; 0.35m/radian",
        "original_frames": len(pool), "baseline_frames_preserved": len(baseline),
        "training_frames": len(chosen), "added_frames": len(chosen) - len(baseline),
        "source_stride": 3, "selected_source_indices": chosen,
        "holdout_guard_seconds": guard_s,
        "nearest_training_to_holdout_seconds": min(
            abs(float(pool[i]["t"]) - t) for i in chosen for t in held),
        "same_frozen_baseline_associations": True,
    }

def sheet(original: Image.Image | None, old: np.ndarray, new: np.ndarray,
          target: Path, title: str) -> None:
    if any(x.shape != (240, 320, 3) or x.dtype != np.uint8 for x in (old, new)):
        raise DataError("Mesh renders must be real 320x240 RGB arrays")
    image = Image.new("RGB", (1920, 514), (22, 26, 32))
    draw = ImageDraw.Draw(image)
    if original is not None:
        if original.size != (640, 480):
            raise DataError("Expected untouched original 640x480 RGB")
        image.paste(original, (0, 34))
    for x, render in ((640, old), (1280, new)):
        image.paste(Image.fromarray(render).resize(
            (640, 480), resample=Image.Resampling.NEAREST), (x, 34))
    for x, label in zip((0, 640, 1280), (
            title + (" | original 640x480" if original else " | free camera"),
            "Sparse TSDF 640 | uniform extra views",
            "Sparse TSDF 640 | pose-diverse extra views")):
        draw.text((x + 7, 9), label, fill=(238, 238, 238))
    image.save(target)


def render_views(dataset: Path, original_import: Path, target: Path) -> list[dict]:
    rec = load_recording(original_import / "recording.json")
    assoc = json.loads((original_import / "association.json").read_text(encoding="utf8"))
    meshes = {}
    for name in ("sparse-expanded", "sparse-pose-diverse"):
        path = target / name / "mesh.ply"
        v, f = read_echo_ply(path)
        meshes[name] = (v, f, read_vertex_rgb(path, len(v)))
    visual = target / "visual"
    visual.mkdir(exist_ok=True)
    camera = rec.camera.scaled(2)
    result = []
    for i in VIEWS:
        pose = np.asarray(rec.entries[i]["cameraToWorld"]).reshape(4, 4, order="F")
        spec = assoc["frames"][i]
        rgb_path = safe_path(dataset, spec["rgb_path"])
        if file_sha256(rgb_path) != spec["rgb_sha256"]:
            raise DataError("Original RGB differs from pinned association")
        with Image.open(rgb_path) as image:
            original = image.convert("RGB").copy()
        renders = [rasterize_vertex_rgb(*meshes[key], camera, pose,
                   near=.15, far=MAX_DEPTH_M)[0]
                   for key in ("sparse-expanded", "sparse-pose-diverse")]
        name = f"heldout-{i:02d}-uniform-vs-pose.png"
        sheet(original, *renders, visual / name, f"TUM withheld camera {i}")
        result.append({"kind": "original_camera", "index": i, "image": name,
                       "sha256": file_sha256(visual / name)})
    verts = meshes["sparse-pose-diverse"][0]
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    span = max(float(np.max(np.ptp(verts, axis=0))), .5)
    for j, offset in enumerate(((.45, -1.25, .3), (1.2, .35, .45)), 1):
        pose = look_at(center + span * np.asarray(offset), center)
        renders = [rasterize_vertex_rgb(*meshes[key], camera, pose,
                   near=.15, far=10.)[0]
                   for key in ("sparse-expanded", "sparse-pose-diverse")]
        name = f"free-{j}-uniform-vs-pose.png"
        sheet(None, *renders, visual / name, f"Free camera {j}")
        result.append({"kind": "free_camera", "image": name,
                       "sha256": file_sha256(visual / name),
                       "camera_to_world": pose.flatten(order="F").tolist()})
    return result


def run(args: argparse.Namespace, report: dict) -> None:
    work, target = args.work.resolve(), args.report_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / "source.tgz"
    if args.archive:
        source = args.archive.resolve()
        if source.stat().st_size != PINNED_ARCHIVE_BYTES:
            raise DataError("Offline TUM archive byte count mismatch")
        shutil.copyfile(source, archive)
        report["download"] = {"bytes": archive.stat().st_size,
                              "sha256": file_sha256(archive), "method": "offline"}
    else:
        download(archive, report)
    if (report["download"]["bytes"] != PINNED_ARCHIVE_BYTES or
            report["download"]["sha256"] != PINNED_ARCHIVE_SHA256):
        raise DataError("Pinned public TUM source changed")
    report["extraction"] = extract(archive, work / "dataset")
    dataset = work / "dataset" / ROOT
    folders = {}
    for name, stride, limit in (("original-60", 12, 60), ("expanded-120", 6, 120),
                                ("diverse-240", 3, 240)):
        dest = work / name
        invoke("import-" + name, [sys.executable, "-m", "echo4d", "import-tum",
               str(dataset), "--output", str(dest),
               "--intrinsics", "525", "525", "319.5", "239.5",
               "--stride", str(stride), "--limit", str(limit),
               "--downsample", "1", "--world-up", "z"], report, timeout=900)
        folders[name] = dest
    old, expanded, dense = (folders[k] for k in ("original-60", "expanded-120", "diverse-240"))
    old_rec, expanded_rec, dense_rec = [load_recording(p / "recording.json")
                                        for p in (old, expanded, dense)]
    if (old_rec.source != "recorded" or expanded_rec.source != "recorded"
            or old_rec.camera != expanded_rec.camera or old_rec.camera != dense_rec.camera
            or (old_rec.camera.width, old_rec.camera.height) != (640, 480)):
        raise DataError("Both imports must be 640x480 recorded RGB-D with identical calibration")
    baseline = split_training(json.loads((old / "recording.json").read_text(encoding="utf8")))
    extra, selection = select_extra_training(
        json.loads((old / "recording.json").read_text(encoding="utf8")),
        json.loads((expanded / "recording.json").read_text(encoding="utf8")),
        json.loads((old / "association.json").read_text(encoding="utf8")),
        json.loads((expanded / "association.json").read_text(encoding="utf8")))
    pose, diversity = select_pose_diverse_training(
        json.loads((old / "recording.json").read_text(encoding="utf8")),
        json.loads((dense / "recording.json").read_text(encoding="utf8")),
        json.loads((old / "association.json").read_text(encoding="utf8")),
        json.loads((dense / "association.json").read_text(encoding="utf8")),
        budget=len(extra["frames"]))
    report["training_selection"] = {"uniform": selection, "pose_diverse": diversity}
    manifests = {}
    for name, folder, data in (("sparse-expanded", expanded, extra),
                               ("sparse-pose-diverse", dense, pose)):
        manifest = folder / ("train-" + name + ".json")
        manifest.write_text(json.dumps(data, allow_nan=False, sort_keys=True) + "\n", encoding="utf8")
        manifests[name] = manifest
        if len(load_recording(manifest)) != len(data["frames"]):
            raise DataError("Training manifest is not loadable")
    for name, manifest in manifests.items():
        report.setdefault("fusion", {})[name] = fuse_sparse(
            manifest, target / name, voxel_m=VOXEL_M, truncation_m=TRUNC_M,
            max_depth_m=MAX_DEPTH_M, pixel_stride=2)
        mesh = target / name / "mesh.ply"
        vertices, faces = read_echo_ply(mesh)
        score = evaluate_frames(vertices, faces, old_rec, list(HOLDOUT),
                                max_depth=MAX_DEPTH_M)
        report.setdefault("results", {})[name] = {
            "mesh_sha256": file_sha256(mesh), "vertices": len(vertices),
            "triangles": len(faces),
            "connectivity": face_components(len(vertices), faces),
            "same_sensor_holdout_diagnostic_only": {k: score[k] for k in
                ("heldout_frames", "observed_depth_coverage", "rmse_m", "mae_m", "p95_abs_m")}}
    report["visual"] = render_views(dataset, old, target)
    report["status"] = "real_meshes_rendered_visual_review_pending"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "TUM-fr1_xyz-Open3D-sparse-88-uniform-vs-pose-diverse-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkout_sha": os.environ.get("GITHUB_SHA"),
        "dataset_url": URL, "dataset_sha256": PINNED_ARCHIVE_SHA256,
        "original_rgb_resolution": [640, 480],
        "frozen_holdout_indices": list(HOLDOUT),
        "geometry": {"voxel_m": VOXEL_M, "truncation_m": TRUNC_M,
                     "pixel_stride": 2, "max_depth_m": MAX_DEPTH_M},
        "independent_surface_truth": "not_available",
        "human_visual_review": "not_performed",
        "issue_74": "FAIL / NOT ACCEPTED",
        "desk_v2": "FAIL 16.7779cm > 15cm; unchanged",
        "commands": [], "status": "started"}
    start = time.perf_counter()
    try:
        run(args, report)
        code = 0
    except Exception as exc:
        code = 1
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"TUM_DENSITY_REAL_EXPERIMENT_FAILED: {exc}", file=sys.stderr)
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        (args.report_dir / "training-density-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
        (args.report_dir / "SUMMARY.md").write_text("\n".join([
            "# TUM real sparse RGB-D camera-pose diversity comparison",
            f"- Status: {report['status']}",
            "- Original 640x480 RGB-D; same 30mm sparse TSDF, matched-count uniform vs pose-diverse frames.",
            f"- Training: {report.get('training_selection', {})}",
            f"- Meshes: {list(report.get('results', {}))}",
            f"- Actual 3+2 visual images: {len(report.get('visual', []))}",
            "- Sensor-depth holdout is not independent surface truth.",
            "- #74 visual acceptance remains FAIL.",
            f"- Failure: {report.get('failure', {}).get('message', 'none')}"]) + "\n", encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
