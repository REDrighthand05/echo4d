"""Extract actual archived video frames and pair ONLY externally synchronized metric RGB-D.

The historical media is preserved; the mesh-ready recording is a derived dataset.
Browser IMU is NEVER interpreted as a world-space camera pose.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from .model import Camera, DataError, Frame, file_sha256, finite_number, rigid_matrix, safe_path, load_recording
from .output import output_transaction, write_json
from .history import TIMELINE_VERSION, _read_json

VERSION = "echo-history-rgbd-alignment-1"
MAX_FRAMES = 256


def _command(command: list[str], timeout: int) -> str:
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DataError(f"Video decoder unavailable, timed out, or failed: {command[0]}") from error
    if run.returncode:
        raise DataError(f"{command[0]} failed: {run.stderr[-1200:]}")
    return run.stdout


def _video_pts(media: Path) -> list[float]:
    raw = _command(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "frame=best_effort_timestamp_time",
                    "-of", "json", str(media)], timeout=90)
    try:
        parsed = json.loads(raw)
        times = [float(frame["best_effort_timestamp_time"])
                 for frame in parsed["frames"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DataError("Cannot read every decoded video presentation timestamp") from error
    if not times or not all(math.isfinite(t) for t in times):
        raise DataError("Video has no usable finite frame timestamps")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise DataError("Video frame presentation timestamps do not strictly increase")
    return [t - times[0] for t in times]


def prepare_history_rgbd(history: Path, alignment_path: Path, output: Path) -> dict:
    """Decode selected source frames once; pair measured depth and known poses."""
    history, alignment_path, output = Path(history), Path(alignment_path), Path(output)
    timeline_path = history / "history.timeline.json"
    timeline = _read_json(timeline_path)
    if timeline.get("version") != TIMELINE_VERSION or timeline.get("geometry_status") != "not_reconstructed":
        raise DataError("A genuine unreconstructed history timeline is required")
    media_info = timeline.get("media")
    if not isinstance(media_info, dict):
        raise DataError("History timeline lacks original media information")
    media = safe_path(history, media_info.get("archive_relative_path"))
    media_sha = file_sha256(media)
    if (media_info.get("sha256") != media_sha or
            media_info.get("bytes") != media.stat().st_size):
        raise DataError("Historical video was modified after archival")

    alignment = _read_json(alignment_path)
    if (alignment.get("version") != VERSION or
            alignment.get("source_media_sha256") != media_sha):
        raise DataError("RGB-D alignment must bind the exact original video")
    sync = alignment.get("clock_sync")
    if not isinstance(sync, dict):
        raise DataError("Explicit video-to-history clock calibration required")
    offset = finite_number(sync.get("video_to_history_offset_s"), "video-to-history clock offset")
    if abs(offset) > 86400:
        raise DataError("Clock offset outside a day")
    clock_source = sync.get("source")
    if not isinstance(clock_source, str) or len(clock_source.strip()) < 8:
        raise DataError("Clock alignment requires an external provenance description")
    if alignment.get("depth_kind") != "measured_registered_camera_z":
        raise DataError("Predicted or non-registered depth is not accepted as measured Z")
    depth_source = alignment.get("depth_source")
    pose_source = alignment.get("pose_source")
    for source in (depth_source, pose_source):
        if not isinstance(source, str) or len(source.strip()) < 8:
            raise DataError("Real depth and world pose must have explicit source descriptions")
    camera = Camera.from_dict(alignment.get("intrinsics"))
    records = alignment.get("frames")
    if not isinstance(records, list) or not 2 <= len(records) <= MAX_FRAMES:
        raise DataError(f"Provide 2-{MAX_FRAMES} selected video frames")
    if timeline.get("duration_s") is None:
        raise DataError("History duration is missing")
    duration = finite_number(timeline["duration_s"], "history duration")
    if duration <= 0 or duration > 86400:
        raise DataError("Invalid history duration")

    pts = _video_pts(media)
    checked = []
    previous_index = -1
    previous_time = -1.0
    for record in records:
        if not isinstance(record, dict):
            raise DataError("Alignment frames must be objects")
        index = record.get("video_frame_index")
        if type(index) is not int or index <= previous_index or index >= len(pts):
            raise DataError("Selected video frame indices must increase and exist in the video")
        t = finite_number(record.get("t_s"), "history frame time")
        declared_pts = finite_number(record.get("video_pts_s"), "video PTS")
        depth_t = finite_number(record.get("depth_t_s"), "measured depth time")
        pose_t = finite_number(record.get("pose_t_s"), "external pose time")
        if (t <= previous_time or t < 0 or t > duration or
                abs(declared_pts - pts[index]) > 0.005 or
                abs(t - (pts[index] + offset)) > 0.03 or
                abs(t - depth_t) > 0.03 or abs(t - pose_t) > 0.03):
            raise DataError("Video, depth, pose and history timestamps are not aligned within 30ms")
        pose = rigid_matrix(record.get("cameraToWorld"))
        depth_path = safe_path(alignment_path.parent, record.get("depth"))
        if depth_path.suffix.lower() != ".npy" or depth_path.stat().st_size > 64 * 1024 * 1024:
            raise DataError("Only bounded metric float NPY depth frames are accepted")
        depth_sha = file_sha256(depth_path)
        try:
            depth = np.load(depth_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise DataError("Cannot load a measured-depth NPY frame") from error
        if depth.dtype.kind != "f" or depth.shape != (camera.height, camera.width):
            raise DataError("Depth must be a full-resolution numeric float camera-Z image in metres")
        if not np.isfinite(depth).all() or (depth < 0).any() or (depth > 100).any():
            raise DataError("Measured depth is outside the metre range [0, 100]")
        checked.append((index, t, depth_path, depth_sha, pose, depth.astype(np.float32)))
        previous_index, previous_time = index, t

    if output.exists():
        raise DataError("Dataset output already exists")
    with output_transaction(output) as dest:
        (dest / "color").mkdir()
        (dest / "depth").mkdir()
        # Decode by frame index in a single FFmpeg invocation. The selected
        # images are derived from the actual preserved video, never substitutes.
        expression = "+".join(f"eq(n\\,{index})" for index, *_ in checked)
        command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(media),
                   "-map", "0:v:0", "-an", "-sn", "-dn",
                   "-vf", f"select={expression}", "-vsync", "0",
                   "-frames:v", str(len(checked)), "-start_number", "0",
                   str(dest / "color" / "%05d.png")]
        _command(command, timeout=180)
        produced = sorted((dest / "color").glob("*.png"))
        if len(produced) != len(checked):
            raise DataError("FFmpeg did not decode all selected historical frames")
        entries, receipts = [], []
        for n, (index, t, source_depth, sha, pose, depth) in enumerate(checked):
            color = produced[n]
            with Image.open(color) as im:
                if im.size != (camera.width, camera.height):
                    raise DataError("Actual decoded video dimensions do not match calibration")
                rgb = np.asarray(im.convert("RGB"))
            Frame(t, pose, depth, rgb).validate(camera)
            depth_rel = f"depth/{n:05d}.npy"
            np.save(dest / depth_rel, depth, allow_pickle=False)
            entries.append({
                "t": t, "color": f"color/{n:05d}.png", "depth": depth_rel,
                "cameraToWorld": pose.flatten(order="F").tolist(),
            })
            receipts.append({
                "video_frame_index": index, "actual_video_pts_s": pts[index],
                "history_t_s": t, "source_depth_sha256": sha,
                "source_depth_file": str(source_depth.relative_to(alignment_path.parent.resolve())),
                "extracted_rgb_sha256": file_sha256(color),
            })
        write_json(dest / "recording.json", {
            "version": "echo-dataset-1", "name": "history-known-pose-rgbd",
            "source": "recorded", "units": "meters", "duration": duration,
            "depth_scale": 1, "intrinsics": camera.to_dict(), "frames": entries,
            "notes": "Externally aligned measured depth and known poses; source is archived video.",
        })
        write_json(dest / "history-alignment.json", {
            "version": VERSION,
            "history_timeline_sha256": file_sha256(timeline_path),
            "source_media_sha256": media_sha,
            "alignment_sha256": file_sha256(alignment_path),
            "depth_kind": alignment["depth_kind"],
            "depth_source": depth_source, "pose_source": pose_source,
            "clock_sync": sync, "frames": receipts,
            "limitations": [
                "External camera poses and depth were supplied, not solved from the video",
                "Unknown areas remain unknown",
                "Browser IMU alone is not metric position or depth",
            ],
        })
        # Read the full derived ECHO dataset with the existing reconstruction loader.
        data = load_recording(dest / "recording.json")
        if len(data) != len(checked):
            raise DataError("Generated recording cannot be read")
        if file_sha256(media) != media_sha:
            raise DataError("Source video changed before result publication")
        if any(file_sha256(path) != sha for _, _, path, sha, _, _ in checked):
            raise DataError("Measured depth changed during video decoding")
    return {"output": str(output.resolve()), "frames": len(checked),
            "source_media_sha256": media_sha,
            "next_command": "echo4d reconstruct <output>/recording.json --output <mesh-dir> --bounds <six scene coordinates>",
            "geometry": "ready_for_known_pose_measured_depth_tsdf"}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare_history_rgbd(args.history, args.alignment, args.output),
                         ensure_ascii=False))
        return 0
    except (DataError, OSError, ValueError) as error:
        parser.exit(2, f"History RGB-D bridge failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
