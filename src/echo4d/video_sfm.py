"""Video-only COLMAP SfM experiment: outputs arbitrary-scale poses, never metric depth.

Command failures preserve a report with provenance and bounded command diagnostics.
The original RGB video is the sole input to SfM; no TUM ground truth enters.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .colmap import _cameras, _images
from .model import DataError, file_sha256


def _exec(label: str, command: list[str], report: dict, timeout: int = 1800) -> str:
    """Never use a shell. Record command, duration, status, and errors even on failure."""
    entry = {"name": label, "argv": [str(x) for x in command], "status": "started"}
    report["commands"].append(entry)
    begin = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout,
                                check=False)
        entry["exit_code"] = result.returncode
        entry["stderr_tail"] = result.stderr[-3000:]
        entry["stdout_tail"] = result.stdout[-1000:] if label != "ffprobe" else "(parsed separately)"
        if result.returncode != 0:
            raise DataError(f"{label} exited with code {result.returncode}; see report.json")
        entry["status"] = "passed"
        return result.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        entry["failure"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        raise DataError(f"{label} could not complete; see report.json") from exc
    finally:
        entry["seconds"] = round(time.perf_counter() - begin, 3)
        if entry["status"] == "started":
            entry["status"] = "failed"


def _probe_timestamps(video: Path, report: dict, ffprobe: str) -> list[float]:
    raw = _exec("ffprobe", [ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)], report,
        timeout=240)
    if len(raw) > 8_000_000:
        raise DataError("ffprobe output exceeds frame metadata budget")
    try:
        frames = json.loads(raw)["frames"]
        if not isinstance(frames, list) or not 2 <= len(frames) <= 100_000:
            raise DataError("Video must contain 2..100000 decoded frames")
        values = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataError("Missing or invalid decoded video timestamps") from exc
    if not all(math.isfinite(t) and t >= 0 for t in values):
        raise DataError("Video timestamps must be finite and nonnegative")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise DataError("Decoded timestamps must be strictly increasing")
    return values


def _select_colmap_model(sparse: Path, output: Path, lookup: dict,
                         report: dict, colmap: str) -> list[dict]:
    """Select the largest valid disconnected COLMAP model, not blindly sparse/0.

    All candidates, including rejected ones, remain visible in report.json.
    Ties use numeric model ID ascending for deterministic selection. Never fuse
    separately reconstructed coordinate systems into one trajectory.
    """
    candidates = sorted(
        (p for p in sparse.iterdir()
         if p.is_dir() and not p.is_symlink() and p.name.isascii() and p.name.isdigit()),
        key=lambda p: (int(p.name), p.name))
    if not candidates:
        raise DataError("COLMAP mapper produced no numbered sparse models")
    if len(candidates) > 32:
        raise DataError("COLMAP produced more than 32 sparse models; manual review required")
    report["model_candidates"] = []
    best: tuple[int, int, str, list[dict]] | None = None
    for model_dir in candidates:
        entry: dict = {"model_id": model_dir.name, "status": "rejected"}
        report["model_candidates"].append(entry)
        try:
            text_dir = output / f"model_txt_{model_dir.name}"
            text_dir.mkdir()
            _exec(f"colmap_model_converter_{model_dir.name}",
                  [colmap, "model_converter", "--input_path", str(model_dir),
                   "--output_path", str(text_dir), "--output_type", "TXT"], report)
            cameras = _cameras(text_dir / "cameras.txt")
            images = _images(text_dir / "images.txt")
            poses = []
            for name, camera_id, pose in images:
                if name not in lookup or camera_id not in cameras:
                    raise DataError("COLMAP model references unknown image or camera")
                if (not np.isfinite(pose).all() or
                        not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5) or
                        not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
                    raise DataError("COLMAP pose is not a finite right-handed rigid transform")
                poses.append({"name": name, "camera_id": camera_id,
                              "source_pts_seconds": lookup[name]["source_pts_seconds"],
                              "camera_to_world_row_major": pose.tolist()})
            entry["registered_images"] = len(poses)
            if len(poses) < 2:
                entry["reason"] = "fewer than two registered frames"
                continue
            poses.sort(key=lambda p: p["source_pts_seconds"])
            entry["status"] = "valid"
            key = (-len(poses), int(model_dir.name), model_dir.name)
            if best is None or key < best[:3]:
                best = (*key, poses)
                report["selected_model_id"] = model_dir.name
        except (DataError, OSError, ValueError) as exc:
            entry["reason"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    if best is None:
        raise DataError("No valid COLMAP model with >=2 registered frames; see model_candidates")
    return best[3]


def run(video: Path, output: Path, *, max_frames: int = 120,
        ffprobe: str = "ffprobe", ffmpeg: str = "ffmpeg", colmap: str = "colmap") -> dict:
    if not 2 <= max_frames <= 500:
        raise DataError("max_frames must be in [2, 500]")
    video = Path(video).resolve(strict=True)
    if not video.is_file():
        raise DataError("Input video must be a regular file")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)  # never overwrite previous evidence
    report = {"version": "echo-video-sfm-1", "status": "running",
              "source": {"path": str(video), "sha256": file_sha256(video)},
              "scale": "unknown; COLMAP monocular reconstruction units",
              "ground_truth_used_for_sfm": False, "commands": []}
    try:
        times = _probe_timestamps(video, report, ffprobe)
        step = math.ceil(len(times) / max_frames)
        indices = list(range(0, len(times), step))
        if len(indices) < 2:
            raise DataError("Sampling yielded fewer than two frames")
        image_dir = output / "images"
        image_dir.mkdir()
        _exec("ffmpeg", [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-map", "0:v:0", "-an", "-sn", "-vf",
            f"select=not(mod(n\\,{step}))", "-fps_mode", "passthrough",
            "-frames:v", str(len(indices)), str(image_dir / "%06d.png")], report,
            timeout=600)
        names = [f"{n + 1:06d}.png" for n in range(len(indices))]
        extracted = sorted(p.name for p in image_dir.iterdir() if p.is_file())
        if extracted != names:
            raise DataError("FFmpeg extracted frames do not match decoded sampling indices")
        samples = [{"name": name, "decoded_frame_index": idx,
                    "source_pts_seconds": times[idx],
                    "relative_pts_seconds": times[idx] - times[0]}
                   for name, idx in zip(names, indices)]
        report["sampling"] = {"decoded_frames": len(times), "stride": step,
                              "extracted_frames": len(samples), "samples": samples}
        database = output / "database.db"
        sparse = output / "sparse"
        sparse.mkdir()
        _exec("colmap_feature_extractor", [colmap, "feature_extractor",
            "--database_path", str(database), "--image_path", str(image_dir),
            "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "PINHOLE",
            "--SiftExtraction.use_gpu", "0"], report)
        _exec("colmap_sequential_matcher", [colmap, "sequential_matcher",
            "--database_path", str(database), "--SiftMatching.use_gpu", "0"], report)
        _exec("colmap_mapper", [colmap, "mapper", "--database_path", str(database),
            "--image_path", str(image_dir), "--output_path", str(sparse)], report)
        lookup = {s["name"]: s for s in samples}
        poses = _select_colmap_model(sparse, output, lookup, report, colmap)
        trajectory = {"version": "echo-colmap-trajectory-1",
                      "coordinate_system": "COLMAP arbitrary world axes",
                      "translation_unit": "unknown_colmap_units",
                      "metric_scale_known": False, "poses": poses}
        (output / "trajectory.json").write_text(
            json.dumps(trajectory, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        report["registration"] = {"registered": len(poses), "sampled": len(samples),
                                  "fraction": len(poses) / len(samples)}
        report["status"] = "passed"
        return {"output": str(output), "registered": len(poses),
                "sampled": len(samples), "scale": "unknown"}
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        raise
    finally:
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MP4 -> FFmpeg frames -> CPU COLMAP poses (unknown scale)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--colmap", default="colmap")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.video, args.output, max_frames=args.max_frames,
                             ffprobe=args.ffprobe, ffmpeg=args.ffmpeg, colmap=args.colmap),
                         ensure_ascii=False))
        return 0
    except (DataError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
