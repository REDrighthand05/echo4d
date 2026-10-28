"""Real TUM fr1_room: full-room loop trajectory, known poses + measured RGB-D.

Compare 96 real training views with and without bounded frame-to-TSDF ICP
on the SAME room scene with 12 frozen holdouts. Neither route uses holdout depth for fusion or geometry selection.
No inferred depth, mesh completion, or claim of independent surface truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

import numpy as np
from PIL import Image

from echo4d.holdout import evaluate_frames, read_echo_ply
from echo4d.model import DataError, file_sha256, load_recording, safe_path
from sparse_open3d_fusion import fuse_sparse
from room_pose_icp import refine_training_poses
from pathlib import PurePosixPath
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from tum_graph_visual import look_at, rasterize_vertex_rgb, read_vertex_rgb
from tum_highres_tsdf import face_components, invoke
from tum_sparse_training_density import sheet

IMPORT_FRAMES = 180
HOLDOUT = tuple(range(7, IMPORT_FRAMES, 15))
BASE_FRAMES = 48
EXPANDED_FRAMES = 96
DENSE_FRAMES = 120
GUARD_SECONDS = 0.16
VOXEL_M = 0.03
TRUNC_M = 0.09
MAX_DEPTH_M = 4.0
ROOT = "rgbd_dataset_freiburg1_room"
URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz"
MAX_ARCHIVE_BYTES = 1200 * 1024**2
MAX_UNPACKED_BYTES = 4 * 1024**3

def validate_archive_name(name: str, kind: str, size: int) -> tuple[str, ...]:
    path = PurePosixPath(name)
    parts = path.parts
    if (path.is_absolute() or not parts or parts[0] != ROOT or ".." in parts
            or kind not in {"dir", "file"} or size < 0 or size > 50 * 1024**2):
        raise DataError("Unsafe TUM room archive member")
    return parts


def get_archive(destination: Path, receipt: dict, offline: Path | None) -> None:
    """Fetch official HTTPS room sequence with a strict size bound and actual hash."""
    record = {"official_url": URL, "vendor_signed_checksum": False}
    receipt["archive"] = record
    if offline is not None:
        if not offline.is_file() or not 1000 <= offline.stat().st_size <= MAX_ARCHIVE_BYTES:
            raise DataError("Offline room archive missing or oversized")
        shutil.copyfile(offline, destination)
        record["origin"] = "offline, origin not independently authenticated"
    else:
        temporary = destination.with_suffix(".part")
        try:
            with urlopen(Request(URL, headers={"User-Agent": "Echo4D-room/1.0"}), timeout=120) as response:
                location = urlsplit(response.geturl())
                if location.scheme != "https" or location.hostname not in {
                    "cvg.cit.tum.de", "webshare.cvg.cit.tum.de"
                }:
                    raise DataError("Room archive redirected off official HTTPS hosts")
                record["resolved_url"] = response.geturl()
                if int(response.headers.get("Content-Length", "0")) > MAX_ARCHIVE_BYTES:
                    raise DataError("Room download exceeds size budget")
                count = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > MAX_ARCHIVE_BYTES:
                            raise DataError("Room download exceeds size budget")
                        output.write(chunk)
            temporary.replace(destination)
            record["origin"] = "official HTTPS"
        finally:
            temporary.unlink(missing_ok=True)
    record["bytes"] = destination.stat().st_size
    record["sha256"] = file_sha256(destination)



def extract_rgbd(archive: Path, destination: Path) -> dict:
    """Bounded extraction of the pre-pinned public archive, no symlinks or traversal."""
    extracted, unpacked, rgb, depth, indexes = 0, 0, 0, 0, set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            parts = validate_archive_name(
                member.name, "dir" if member.isdir() else
                "file" if member.isfile() else "other", member.size)
            if member.isdir():
                continue
            extracted += 1
            unpacked += member.size
            if extracted > 25000 or unpacked > MAX_UNPACKED_BYTES:
                raise DataError("TUM room archive exceeds extraction limits")
            relative = parts[1:]
            is_index = len(relative) == 1 and relative[0] in {
                "rgb.txt", "depth.txt", "groundtruth.txt"}
            is_rgb = len(relative) == 2 and relative[0] == "rgb" and relative[1].endswith(".png")
            is_depth = len(relative) == 2 and relative[0] == "depth" and relative[1].endswith(".png")
            if not (is_index or is_rgb or is_depth):
                continue
            path = destination.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            if stream is None:
                raise DataError("Cannot read TUM archive member")
            with stream, path.open("xb") as output:
                shutil.copyfileobj(stream, output)
            if path.stat().st_size != member.size:
                raise DataError("Short TUM RGB-D extraction")
            if is_index:
                indexes.add(relative[0])
            rgb += int(is_rgb)
            depth += int(is_depth)
    if indexes != {"rgb.txt", "depth.txt", "groundtruth.txt"} or min(rgb, depth) < 900:
        raise DataError("TUM room missing real RGB-D frames or known poses")
    return {"rgb_pngs": rgb, "depth_pngs": depth, "indexes": sorted(indexes),
            "archive_members_checked": extracted}


def select_training(manifest: dict, *, baseline: int = BASE_FRAMES,
                    expanded: int = EXPANDED_FRAMES,
                    guard_s: float = GUARD_SECONDS) -> tuple[dict, dict, dict]:
    """Select real views only; 48 baseline subset of 96, frozen 12 holdouts.

    Pose novelty is calculated only from TRAINING poses. No holdout depth, color,
    pose, or geometric score enters the selection.
    """
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not 140 <= len(frames) <= IMPORT_FRAMES:
        raise DataError("Expected 140-180 synchronized real room RGB-D frames")
    holdout = tuple(int(round(v)) for v in np.linspace(7, len(frames) - 8, 12))
    if len(set(holdout)) != 12:
        raise DataError("Room holdout selection must contain 12 distinct frames")
    if not (0 < guard_s <= 0.3 and 0 < baseline < expanded <= 120):
        raise DataError("Invalid desk training selection budget")
    times = np.array([float(f["t"]) for f in frames])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise DataError("Nonmonotonic TUM room timestamps")
    held_times = times[list(holdout)]
    candidates = [i for i in range(len(frames)) if i not in holdout and
                  np.min(np.abs(held_times - times[i])) > guard_s]
    if len(candidates) < expanded:
        raise DataError("Not enough independent real training views outside holdout guard")
    # Evenly distributed baseline across the complete measured trajectory.
    initial = np.linspace(0, len(candidates) - 1, baseline)
    base = sorted({candidates[int(round(x))] for x in initial})
    if len(base) != baseline:
        raise DataError("Baseline sampling unexpectedly duplicated a real frame")
    poses = np.stack([np.asarray(f["cameraToWorld"], dtype=float).reshape(
        4, 4, order="F") for f in frames])
    if not np.isfinite(poses).all():
        raise DataError("Nonfinite measured camera poses")
    centers = poses[:, :3, 3]
    axes = poses[:, :3, 2]
    axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
    # Greedy novelty in camera center and optical axis; never fabricate a view.
    selected = list(base)
    extras = set(candidates) - set(base)
    while len(selected) < expanded:
        winner = max(extras, key=lambda j: (
            min(float(np.linalg.norm(centers[j] - centers[i]) +
                      0.35 * np.arccos(np.clip(axes[j] @ axes[i], -1., 1.)))
                for i in selected), -j))
        selected.append(winner)
        extras.remove(winner)
    def subset(indices: list[int], name: str) -> dict:
        out = dict(manifest)
        out["frames"] = [frames[i] for i in sorted(indices)]
        out["notes"] = "TUM fr1_room measured RGB-D and known poses; " + name + "; no holdout fusion"
        return out
    trajectory_extent = np.ptp(centers[candidates], axis=0).tolist()
    return subset(base, "48 real views"), subset(selected, "96 pose-diverse real views"), {
        "source_frames": len(frames), "holdout_indices": list(holdout),
        "holdout_guard_seconds": guard_s, "eligible_train": len(candidates),
        "baseline_indices": base, "expanded_indices": sorted(selected),
        "expanded_contains_baseline": set(base).issubset(selected),
        "training_camera_center_extent_m": trajectory_extent,
        "min_train_holdout_time_gap_s": float(min(
            abs(times[i] - t) for i in selected for t in held_times)),
    }


def render_five(dataset: Path, imported: Path, target: Path, holdout: list[int]) -> list[dict]:
    rec = load_recording(imported / "recording.json")
    assoc = json.loads((imported / "association.json").read_text(encoding="utf8"))
    keys = ("room-96-4m", "room-96-icp")
    meshes = {}
    for key in keys:
        mesh_path = target / key / "mesh.ply"
        v, f = read_echo_ply(mesh_path)
        meshes[key] = (v, f, read_vertex_rgb(mesh_path, len(v)))
    camera = rec.camera.scaled(2)
    visual = target / "visual"
    visual.mkdir(exist_ok=True)
    images = []
    for i in (holdout[0], holdout[len(holdout) // 2], holdout[-1]):
        pose = np.asarray(rec.entries[i]["cameraToWorld"]).reshape(4, 4, order="F")
        source = assoc["frames"][i]
        original_path = safe_path(dataset, source["rgb_path"])
        if file_sha256(original_path) != source["rgb_sha256"]:
            raise DataError("Original room RGB changed")
        with Image.open(original_path) as raw:
            original = raw.convert("RGB").copy()
        views = [rasterize_vertex_rgb(*meshes[key], camera, pose,
                 near=.15, far=MAX_DEPTH_M)[0] for key in keys]
        name = f"heldout-{i:03d}-room-96-vs-icp.png"
        sheet(original, *views, visual / name, f"Real TUM fr1_room camera {i}",
              ("96 known poses / measured depth", "96 training-only ICP poses / measured depth"))
        images.append({"kind": "heldout_camera", "index": i, "name": name,
                       "sha256": file_sha256(visual / name),
                       "source_rgb_sha256": source["rgb_sha256"]})
    # Previous free cameras sat >10m from the room center but clipped at 10m,
    # producing black renders independent of Mesh geometry. Shift two real
    # camera poses for free third-person views, without using held-out RGB-D.
    for i, (index, lateral) in enumerate(((holdout[2], .35),
                                          (holdout[-3], -.35)), 1):
        anchor = np.asarray(rec.entries[index]["cameraToWorld"]).reshape(
            4, 4, order="F")
        eye = anchor[:3, 3] + lateral * anchor[:3, 0] - .12 * anchor[:3, 1]
        target_point = anchor[:3, 3] + 1.8 * anchor[:3, 2]
        pose = look_at(eye, target_point)
        renders = [rasterize_vertex_rgb(*meshes[key], camera, pose,
                   near=.15, far=10.) for key in keys]
        views = [render[0] for render in renders]
        name = f"free-{i}-room-96-vs-icp.png"
        sheet(None, *views, visual / name, f"Real TUM room shifted free camera {i}",
              ("96 known poses / measured depth", "96 bounded ICP poses / measured depth"))
        images.append({"kind": "free_camera", "name": name,
                       "sha256": file_sha256(visual / name),
                       "anchor_holdout_index": index,
                       "rendered_mesh_pixels": [int(render[1]) for render in renders],
                       "camera_to_world": pose.flatten(order="F").tolist()})
    return images


def run(args: argparse.Namespace, report: dict) -> None:
    work = args.work.resolve()
    target = args.report_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / "room.tgz"
    get_archive(archive, report, args.archive.resolve() if args.archive else None)
    report["archive_observed_sha256"] = file_sha256(archive)
    report["extraction"] = extract_rgbd(archive, work / "source")
    dataset = work / "source" / ROOT
    imported = work / "room-import"
    invoke("import-room-180", [sys.executable, "-m", "echo4d", "import-tum",
           str(dataset), "--output", str(imported),
           "--intrinsics", "525", "525", "319.5", "239.5",
           "--stride", "8", "--limit", str(IMPORT_FRAMES),
           "--downsample", "1", "--world-up", "z"], report, timeout=900)
    rec = load_recording(imported / "recording.json")
    if rec.source != "recorded" or (rec.camera.width, rec.camera.height) != (640, 480):
        raise DataError("Expected full 640x480 measured RGB-D")
    manifest = json.loads((imported / "recording.json").read_text(encoding="utf8"))
    _, extra, selection = select_training(manifest)
    report["training_selection"] = selection
    baseline_manifest = imported / "train-room-96-4m.json"
    baseline_manifest.write_text(
        json.dumps(extra, allow_nan=False, sort_keys=True) + "\n", encoding="utf8")
    baseline_target = target / "room-96-4m"
    report.setdefault("fusion", {})["room-96-4m"] = fuse_sparse(
        baseline_manifest, baseline_target, voxel_m=VOXEL_M, truncation_m=TRUNC_M,
        max_depth_m=MAX_DEPTH_M, pixel_stride=2)
    refined_manifest = imported / "train-room-96-icp.json"
    report["pose_refinement"] = refine_training_poses(
        baseline_manifest, baseline_target / "mesh.ply", refined_manifest)
    report["fusion"]["room-96-icp"] = fuse_sparse(
        refined_manifest, target / "room-96-icp",
        voxel_m=VOXEL_M, truncation_m=TRUNC_M,
        max_depth_m=MAX_DEPTH_M, pixel_stride=2)
    for name in ("room-96-4m", "room-96-icp"):
        mesh = target / name / "mesh.ply"
        vertices, faces = read_echo_ply(mesh)
        score = evaluate_frames(vertices, faces, rec, selection["holdout_indices"],
                                max_depth=MAX_DEPTH_M)
        report.setdefault("results", {})[name] = {
            "mesh_sha256": file_sha256(mesh),
            "vertices": len(vertices), "triangles": len(faces),
            "connectivity": face_components(len(vertices), faces),
            "same_sensor_holdout_diagnostic_only": {k: score[k] for k in (
                "heldout_frames", "observed_depth_coverage", "rmse_m", "mae_m", "p95_abs_m")}}
    report["visual"] = render_five(dataset, imported, target, selection["holdout_indices"])
    report["status"] = "real_room_known_pose_vs_bounded_icp_meshes_rendered_visual_review_pending"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "TUM-fr1_room-known-pose-vs-bounded-training-ICP-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkout_sha": os.environ.get("GITHUB_SHA"),
        "official_source": URL, "archive_checksum_status": "observed at run; not vendor-signed",
        "original_rgb_resolution": [640, 480],
        "camera_poses": "TUM motion-capture ground truth (not inferred)",
        "depth": "TUM sensor measurement (not predicted)",
        "geometry": {"voxel_m": VOXEL_M, "truncation_m": TRUNC_M,
                     "pixel_stride": 2, "depth_limit_m": MAX_DEPTH_M,
                     "training_view_variants": [EXPANDED_FRAMES, EXPANDED_FRAMES],
                     "frozen_holdout_max_depth_m": MAX_DEPTH_M},
        "independent_surface_truth": "not_available",
        "human_visual_review": "not_performed",
        "issue_74": "FAIL / NOT ACCEPTED",
        "desk_v2_frozen": "FAIL 16.7779cm > 15cm; unchanged protocol",
        "commands": [], "status": "started"}
    start = time.perf_counter()
    try:
        run(args, report)
        code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"TUM_ROOM_SPARSE_REAL_FAILED: {exc}", file=sys.stderr)
        code = 1
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        (args.report_dir / "room-sparse-quality-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
        (args.report_dir / "SUMMARY.md").write_text("\n".join([
            "# Real fr1_room 96 known-pose vs bounded training-only ICP at 4m",
            f"- Status: {report['status']}",
            "- Original RGB 640x480; 96 known-pose measured-depth views per variant, both 4m.",
            f"- Mesh results: {list(report.get('results', {}))}",
            f"- 3 real camera + 2 free camera images: {len(report.get('visual', []))}",
            "- ICP reference is training-only and not independent surface truth; #74 FAIL.",
            f"- Failure: {report.get('failure', {}).get('message', 'none')}"]) + "\n",
            encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
