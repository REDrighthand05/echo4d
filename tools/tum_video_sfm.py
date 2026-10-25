"""Real-data experiment: official TUM RGB -> encoded MP4 -> CPU COLMAP -> post-hoc GT.

Only original RGB frames reach the pose estimator. TUM GT is read *after*
the estimator subprocess returns successfully. Failure always leaves receipts.
No GT poses are substituted for failed COLMAP estimates.
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

from echo4d.model import DataError, file_sha256
from echo4d.sfm_evaluation import read_tum_groundtruth, evaluate_tum
from tum_public_roundtrip import ROOT, URL, PINNED_ARCHIVE_SHA256, download, extract

START = 120  # fixed pre-registered slice; no quality-dependent selection
COUNT = 120
FPS = 30


def rgb_index(path: Path) -> list[dict]:
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or not parts[1].startswith("rgb/") or Path(parts[1]).suffix != ".png":
            raise DataError("Unexpected TUM rgb.txt entry")
        try:
            timestamp = float(parts[0])
        except ValueError as exc:
            raise DataError("Non-numeric TUM RGB timestamp") from exc
        if not __import__("math").isfinite(timestamp):
            raise DataError("Invalid TUM RGB timestamp")
        entries.append({"t": timestamp, "path": parts[1]})
    if len(entries) < START + COUNT:
        raise DataError("TUM RGB sequence too short for pre-registered slice")
    if any(b["t"] <= a["t"] for a, b in zip(entries, entries[1:])):
        raise DataError("TUM RGB frames are not strictly ordered")
    return entries[START:START + COUNT]


def encode_video(dataset: Path, frames: list[dict], video: Path, workspace: Path) -> dict:
    staging = workspace / "rgb_for_video"
    staging.mkdir(parents=True, exist_ok=False)
    selected = []
    for i, frame in enumerate(frames, 1):
        source = (dataset / frame["path"]).resolve()
        if not source.is_relative_to(dataset.resolve()) or not source.is_file():
            raise DataError("Unsafe or missing TUM RGB source frame")
        shutil.copyfile(source, staging / f"{i:06d}.png")
        selected.append({**frame, "sha256": file_sha256(source)})
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-framerate", str(FPS), "-start_number", "1", "-i",
           str(staging / "%06d.png"), "-frames:v", str(COUNT),
           "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", str(video)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode or not video.is_file():
        raise DataError(f"Video encoding failed: {result.stderr[-1000:]}")
    return {"command": cmd, "video_sha256": file_sha256(video),
            "selected_source_frames": selected,
            "origin": "real TUM RGB PNGs, re-encoded as CFR MP4; NOT original phone video",
            "sample_selection": {"start_index_zero_based": START,
                                 "count": COUNT, "fps_assigned": FPS}}


def execute(work: Path, artifact: Path) -> None:
    if work.exists():
        raise DataError("Work directory already exists")
    work.mkdir(parents=True)
    artifact.mkdir(parents=True, exist_ok=False)
    receipt: dict = {"version": "echo-tum-video-sfm-experiment-1",
                     "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                     "status": "running", "groundtruth_used_for_sfm": False,
                     "public_data": {"url": URL, "expected_sha256": PINNED_ARCHIVE_SHA256},
                     "git_head": os.environ.get("ECHO_HEAD_SHA", "not supplied"),
                     "thresholds": {"min_registered": 3, "gt_max_time_gap_s": 0.1},
                     "comparability": "TUM RGB sourced MP4; model fitted on RGB only"}
    began = time.perf_counter()
    try:
        archive = work / "tum.tgz"
        download(archive, receipt)
        receipt["archive_extraction"] = extract(archive, work / "data")
        dataset = work / "data" / ROOT
        frames = rgb_index(dataset / "rgb.txt")
        video = work / "tum_rgb_encoded.mp4"
        receipt["encoding"] = encode_video(dataset, frames, video, work)
        output = work / "sfm"
        # This command cannot read GT; GT is loaded below only if it succeeds.
        cmd = [sys.executable, "-m", "echo4d.video_sfm", str(video),
               "--output", str(output), "--max-frames", str(COUNT)]
        sfm_start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        receipt["sfm"] = {"command": cmd, "exit_code": result.returncode,
                          "duration_seconds": round(time.perf_counter() - sfm_start, 2),
                          "stdout_tail": result.stdout[-2000:],
                          "stderr_tail": result.stderr[-5000:]}
        if (output / "report.json").exists():
            shutil.copyfile(output / "report.json", artifact / "sfm-report.json")
        if result.returncode:
            raise DataError("Real CPU COLMAP failed; inspect sfm-report.json and workflow logs")
        trajectory = json.loads((output / "trajectory.json").read_text())
        sampling = json.loads((output / "report.json").read_text())["sampling"]
        receipt["sfm"]["registered"] = len(trajectory["poses"])
        receipt["sfm"]["sampled"] = len(sampling["samples"])
        if len(trajectory["poses"]) < 3:
            raise DataError("Not enough registered poses for independent Sim(3) scoring")
        # Evaluation starts strictly AFTER SfM has completed.
        gt_times, gt_poses = read_tum_groundtruth(dataset / "groundtruth.txt")
        metrics = evaluate_tum(trajectory, sampling, frames, gt_times, gt_poses)
        (artifact / "trajectory.json").write_text(
            json.dumps(trajectory, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (artifact / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        receipt["metrics"] = metrics
        receipt["status"] = "passed"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}
        raise
    finally:
        receipt["elapsed_seconds"] = round(time.perf_counter() - began, 2)
        (artifact / "report.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
        status = receipt["status"]
        scored = receipt.get("metrics")
        lines = [
            "# Real TUM RGB → MP4 → CPU COLMAP → post-hoc Sim(3)",
            f"**Status: {status.upper()}**",
            f"Dataset pinned SHA-256: \`{PINNED_ARCHIVE_SHA256}\`",
            "RGB only for SfM; TUM GT used after solve; eval scale does not make SfM metric.",
            "Derived CFR MP4 is not a native phone video.",
        ]
        if scored is not None:
            lines += [f"Paired poses: {scored['pairs']}",
                      f"Registered/sample: {scored['registered_images']}/{scored['sampled_images']}",
                      f"ATE RMSE (meters, after GT Sim3): {scored['ate_rmse_m_after_sim3']:.6f}",
                      f"Rotation mean error (degrees): {scored['rotation_mean_deg_after_sim3']:.4f}"]
        if receipt.get("failure"):
            lines += [f"Failure: {receipt['failure']['type']}: {receipt['failure']['message']}"]
        (artifact / "SUMMARY.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        execute(args.work, args.report_dir)
        return 0
    except Exception as exc:
        print(f"Real TUM SfM experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
