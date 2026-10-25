"""Reproduce a real TUM RGB-D roundtrip through the strict COLMAP-format adapter.

IMPORTANT: COLMAP text here is derived from TUM ground-truth poses.
No COLMAP SfM or independent scene surface accuracy is being evaluated.
The report records the external download even when later steps fail.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from scipy.spatial.transform import Rotation

from echo4d.evaluation import compare_points
from echo4d.model import DataError, file_sha256, load_recording

URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
ROOT = "rgbd_dataset_freiburg1_xyz"
MAX_ARCHIVE = 700 * 1024**2
MAX_UNPACKED = 3 * 1024**3
BOUNDS = ["-2.45", "-0.25", "-2.25", "1.15", "1.45", "1.65"]
VOXEL = "0.05"


def download(target: Path, receipt: dict) -> None:
    """HTTPS official only; stream bytes, hash, bound size, never store in git."""
    record = {"requested_url": URL, "bytes": 0}
    receipt["download"] = record
    request = Request(URL, headers={"User-Agent": "echo4d-repro/1.0"})
    partial = target.with_suffix(".part")
    try:
        with urlopen(request, timeout=90) as response:
            resolved = response.geturl()
            parsed = urlsplit(resolved)
            if parsed.scheme != "https" or parsed.hostname not in {
                "cvg.cit.tum.de", "webshare.cvg.cit.tum.de"}:
                raise DataError("Dataset URL redirected to an unapproved host")
            record["resolved_url"] = resolved
            record["http_status"] = response.status
            record["etag"] = response.headers.get("ETag")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_ARCHIVE:
                raise DataError("Archive exceeds the allowed download size")
            digest = hashlib.sha256()
            with partial.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    record["bytes"] += len(chunk)
                    if record["bytes"] > MAX_ARCHIVE:
                        raise DataError("Dataset downloaded more than the allowed byte budget")
                    digest.update(chunk)
                    file.write(chunk)
        if record["bytes"] < 1024:
            raise DataError("Archive unexpectedly empty")
        record["sha256"] = digest.hexdigest()
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def extract(archive: Path, dest: Path) -> dict:
    """Bounded, selective extraction; never tar.extractall()."""
    count, total, extracted_images, extracted_indexes = 0, 0, 0, 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            parts = name.parts
            if name.is_absolute() or ".." in parts or not parts or parts[0] != ROOT:
                raise DataError(f"Unsafe archive path: {member.name[:160]}")
            if member.isdir():
                continue
            if not member.isfile():
                raise DataError("Archive contains a non-regular file")
            count += 1
            total += member.size
            if (count > 20000 or total > MAX_UNPACKED or
                    member.size < 0 or member.size > 50 * 1024**2):
                raise DataError("Dataset extraction budget exceeded")
            relative = parts[1:]
            index = len(relative) == 1 and relative[0] in {
                "rgb.txt", "depth.txt", "groundtruth.txt"}
            image = (len(relative) == 2 and relative[0] in {"rgb", "depth"}
                     and relative[1].lower().endswith(".png"))
            if not index and not image:
                continue
            output = dest.joinpath(*parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            handle = tar.extractfile(member)
            if handle is None:
                raise DataError("Unable to read TUM archive member")
            with handle, output.open("wb") as stream:
                shutil.copyfileobj(handle, stream)
            if output.stat().st_size != member.size:
                raise DataError("Extracted member differs from declared archive size")
            extracted_images += int(image)
            extracted_indexes += int(index)
    if extracted_indexes != 3 or extracted_images < 100:
        raise DataError("TUM archive is missing metadata or RGB-D images")
    return {"regular_members": count, "uncompressed_bytes": total,
            "indexes": extracted_indexes, "rgb_depth_images": extracted_images}


def invoke(label: str, command: list[str], receipt: dict, timeout: int = 600) -> None:
    step = {"name": label, "command": command, "status": "started"}
    receipt["commands"].append(step)
    start = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        step.update({"return_code": result.returncode,
                     "stdout_tail": result.stdout[-1500:],
                     "stderr_tail": result.stderr[-2500:]})
        if result.returncode:
            raise RuntimeError(f"{label} failed: {result.stderr[-1200:]}")
        step["status"] = "passed"
    except Exception:
        step["status"] = "failed"
        raise
    finally:
        step["seconds"] = round(time.perf_counter() - start, 3)


def known_pose_colmap(source: Path, output: Path) -> dict:
    """Represent known TUM poses as COLMAP TEXT. This is NOT COLMAP reconstruction."""
    input_record = load_recording(source / "recording.json")
    output.mkdir(parents=True, exist_ok=False)
    camera = input_record.camera
    (output / "cameras.txt").write_text(
        f"1 PINHOLE {camera.width} {camera.height} "
        f"{camera.fx:.17g} {camera.fy:.17g} {camera.cx:.17g} {camera.cy:.17g}\n",
        encoding="utf8")
    lines = ["# TUM ground truth expressed as COLMAP text; NOT COLMAP SfM\n"]
    with (output / "timestamps.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", "t"])
        for i, entry in enumerate(input_record.entries, 1):
            camera_to_world = np.asarray(entry["cameraToWorld"], dtype=float).reshape(4, 4, order="F")
            world_to_camera = np.linalg.inv(camera_to_world)
            q = Rotation.from_matrix(world_to_camera[:3, :3]).as_quat()
            values = [i, q[3], q[0], q[1], q[2],
                      *world_to_camera[:3, 3], 1, PurePosixPath(entry["color"]).name]
            line = " ".join(format(value, ".17g") if isinstance(value, (float, np.floating))
                            else str(value) for value in values)
            lines.append(line + "\n\n")
            writer.writerow([PurePosixPath(entry["color"]).name, repr(entry["t"])])
    (output / "images.txt").write_text("".join(lines), encoding="utf8")
    return {"frames": len(input_record), "pose_source": "TUM provided ground truth",
            "colmap_sfm_ran": False, "scale_source": "TUM ground truth meters",
            "cameras_sha256": file_sha256(output / "cameras.txt"),
            "images_sha256": file_sha256(output / "images.txt")}


def verify_frames(reference: Path, roundtrip: Path) -> dict:
    a = load_recording(reference / "recording.json")
    b = load_recording(roundtrip / "recording.json")
    if len(a) != len(b) or a.camera != b.camera:
        raise DataError("Mismatch in frame count or intrinsics")
    max_pose_error, max_time_error, pixels, frames = 0., 0., 0, 0
    for (_, left, _), (_, right, _) in zip(a.iter_frames(), b.iter_frames()):
        max_pose_error = max(max_pose_error, float(np.max(np.abs(
            left.camera_to_world - right.camera_to_world))))
        max_time_error = max(max_time_error, abs(left.t - right.t))
        if not np.array_equal(left.depth, right.depth):
            raise DataError("Adapter altered real metric depth pixels")
        if not np.array_equal(left.color, right.color):
            raise DataError("Adapter altered RGB pixels")
        pixels += left.depth.size
        frames += 1
    if max_pose_error > 1e-9 or max_time_error > 1e-9:
        raise DataError("Camera matrix/timestamp differed beyond 1e-9 tolerance")
    return {"frames": frames, "depth_pixels": pixels, "rgb_and_depth_exact": True,
            "max_pose_element_error": max_pose_error,
            "max_time_error_seconds": max_time_error}


def experiment(args: argparse.Namespace, receipt: dict) -> None:
    work, report = args.work.resolve(), args.report_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / "source.tgz"
    start = time.perf_counter()
    if args.archive:
        provided = args.archive.resolve()
        if not provided.is_file() or provided.stat().st_size > MAX_ARCHIVE:
            raise DataError("Provided offline archive missing or oversized")
        shutil.copyfile(provided, archive)
        receipt["download"] = {"method": "offline supplied archive, origin unverified",
                               "bytes": archive.stat().st_size, "sha256": file_sha256(archive)}
    else:
        download(archive, receipt)
    receipt["duration_download_s"] = round(time.perf_counter() - start, 3)
    start = time.perf_counter()
    receipt["extraction"] = extract(archive, work / "dataset")
    receipt["duration_extract_s"] = round(time.perf_counter() - start, 3)
    source_root = work / "dataset" / ROOT
    imported = work / "tum"
    invoke("import_tum_groundtruth", [sys.executable, "-m", "echo4d", "import-tum",
           str(source_root), "--output", str(imported), "--intrinsics",
           "525", "525", "319.5", "239.5", "--stride", "12", "--limit", "60",
           "--downsample", "4", "--world-up", "z"], receipt)
    model = work / "colmap-text-from-tum"
    receipt["fixture"] = known_pose_colmap(imported, model)
    converted = work / "converted"
    invoke("colmap_format_adapter", [sys.executable, "-m", "echo4d.colmap",
           "--model", str(model), "--images", str(imported / "color"),
           "--depth", str(imported / "depth"), "--timestamps",
           str(model / "timestamps.csv"), "--meters-per-colmap-unit", "1",
           "--scale-source", "TUM ground-truth trajectory in meters; no SfM scale estimated",
           "--output", str(converted)], receipt)
    receipt["frame_contract"] = verify_frames(imported, converted)
    reconstructed = []
    for label, source in (("baseline", imported), ("adapter", converted)):
        output = work / f"reconstruction-{label}"
        cmd = [sys.executable, "-m", "echo4d", "reconstruct", str(source / "recording.json"),
               "--output", str(output), "--bounds", *BOUNDS, "--voxel", VOXEL,
               "--max-depth", "4", "--min-observations", "2", "--quiet"]
        if label == "baseline":
            cmd.append("--no-scene")
        invoke("tsdf_" + label, cmd, receipt)
        reconstructed.append(output)
    base, candidate = reconstructed
    score = compare_points(np.load(candidate / "vertices.npy", allow_pickle=False),
                           np.load(base / "vertices.npy", allow_pickle=False), 0.03)
    receipt["adapter_mesh_consistency_not_surface_accuracy"] = {
        **score,
        "sha256_equal": file_sha256(base / "mesh.ply") == file_sha256(candidate / "mesh.ply"),
        "baseline_ply_sha256": file_sha256(base / "mesh.ply"),
        "adapter_ply_sha256": file_sha256(candidate / "mesh.ply")}
    receipt["independent_surface_accuracy"] = "not_measured"
    if score["point_rmse_m"] > .02 or score["reference_rmse_m"] > .02 or score["f_score"] < .98:
        raise DataError("Two paths disagree beyond the pre-registered adapter tolerances")
    shutil.copyfile(candidate / "mesh.ply", report / "mesh.ply")
    scene = candidate / "scene.echo.json"
    if scene.is_file() and scene.stat().st_size < 35 * 1024**2:
        shutil.copyfile(scene, report / "scene.echo.json")
    receipt["saved_result"] = {"mesh_ply": True,
                               "scene_echo_json": (report / "scene.echo.json").is_file()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--archive", type=Path, help="Offline reuse; origin remains unverified")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    r = {"experiment": "TUM-freiburg1_xyz-COLMAP-format-contract-v1",
         "started_utc": datetime.now(timezone.utc).isoformat(),
         "repo_sha": os.environ.get("ECHO_HEAD_SHA", os.environ.get("GITHUB_SHA")),
         "actions_url": (f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/"
                         f"{os.environ.get('GITHUB_RUN_ID')}"
                         if os.environ.get("GITHUB_RUN_ID") else None),
         "status": "running", "official_dataset_url": URL,
         "parameters": {"stride": 12, "limit": 60, "downsample": 4, "voxel_m": float(VOXEL),
                        "bounds_m": [float(v) for v in BOUNDS],
                        "calibration": "TUM fr1 default PINHOLE", "world_up": "z"},
         "not_tested": ["COLMAP SfM", "automatic depth estimation",
                        "independent metric scale inference", "surface ground-truth accuracy"],
         "commands": []}
    start = time.perf_counter()
    code = 0
    try:
        experiment(args, r)
        r["status"] = "passed"
    except Exception as exc:
        code = 1
        r["status"] = "failed"
        r["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"TUM_REPRO_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        r["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        report = args.report_dir / "report.json"
        report.write_text(json.dumps(r, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf8")
        lines = ["# TUM real-data contract roundtrip", "",
                 f"- Status: **{r['status']}**",
                 f"- GitHub run: {r['actions_url']}",
                 f"- Commit: {r['repo_sha']}",
                 f"- Archive SHA256: {r.get('download', {}).get('sha256', 'not_downloaded')}",
                 f"- Frames: {r.get('frame_contract', {}).get('frames', 'not_run')}",
                 f"- Max pose error: {r.get('frame_contract', {}).get('max_pose_element_error', 'not_run')}",
                 f"- Adapter F1 at 3 cm: {r.get('adapter_mesh_consistency_not_surface_accuracy', {}).get('f_score', 'not_run')}",
                 "- **Independent surface accuracy: not measured**",
                 f"- Failure: {r.get('error', {}).get('message', 'none')}",
                 "", "See report.json for input, commands, hashes, timing and diagnostics."]
        (args.report_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
        print(json.dumps({"status": r["status"], "report": str(report)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
