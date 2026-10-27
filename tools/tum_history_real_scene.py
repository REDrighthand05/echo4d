"""Real TUM RGB-D camera recording -> ECHO history -> decoded source RGB + measured depth -> TSDF Mesh.

Derived variable-frame-rate reel preserves the selected TUM image timestamps to <= 30ms.
This is an RGB-D camera laboratory sequence, not wearable glasses data and not an
independently verified true surface or dynamic life reconstruction.
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
from PIL import Image

from echo4d.history import import_history
from echo4d.history_rgbd import _video_pts, prepare_history_rgbd
from echo4d.holdout import rasterize_mesh_depth, read_echo_ply
from echo4d.model import DataError, file_sha256, load_recording
from echo4d.output import write_json
from tum_public_roundtrip import (
    BOUNDS, PINNED_ARCHIVE_BYTES, PINNED_ARCHIVE_SHA256, ROOT,
    URL, VOXEL, download, extract,
)

DEFAULT_FRAMES = 60
STRIDE = 12


def cmd(argv: list[str], timeout: int = 800) -> str:
    proc = subprocess.run(argv, check=False, capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode:
        raise DataError(f"{argv[0]} failed ({proc.returncode}): {proc.stderr[-1400:]}")
    return proc.stdout


def video_from_real_rgb(imported: Path, recording, destination: Path) -> list[float]:
    """FFmpeg concat uses irregular TUM frame gaps, not a fabricated 30FPS timeline."""
    listfile = imported / "source-timing.ffconcat"
    lines = ["ffconcat version 1.0"]
    entries = recording.entries
    for i, frame in enumerate(entries):
        # Color frames here are outputs of TUM import-tum at downsample=1.
        lines.append(f"file 'color/{i:05d}.png'")
        if i + 1 < len(entries):
            difference = entries[i + 1]["t"] - frame["t"]
            if difference <= 0 or difference > 30:
                raise DataError("Source RGB-D time gaps are invalid")
            lines.append(f"duration {difference:.8f}")
    listfile.write_text("\n".join(lines) + "\n", encoding="utf8")
    cmd(["ffmpeg", "-y", "-nostdin", "-v", "error", "-f", "concat",
         "-safe", "0", "-i", str(listfile), "-fps_mode", "vfr",
         "-c:v", "libvpx-vp9", "-lossless", "1", "-pix_fmt", "yuv444p",
         "-deadline", "good", "-cpu-used", "5", str(destination)], timeout=600)
    pts = _video_pts(destination)
    if len(pts) != len(entries):
        raise DataError("Derived reel did not preserve every selected real TUM frame")
    if any(abs(t - e["t"]) > .03 for t, e in zip(pts, entries)):
        raise DataError("Derived variable-rate video drifted beyond 30 milliseconds")
    return pts


def visualize(report_dir: Path, dataset: Path, mesh_file: Path) -> dict:
    """Three camera-paired original RGB images and conservative Mesh-only depth renders."""
    recording = load_recording(dataset / "recording.json")
    verts, faces = read_echo_ply(mesh_file)
    dest = report_dir / "views"
    dest.mkdir(parents=True, exist_ok=False)
    choices = sorted(set([0, len(recording)//2, len(recording)-1]))
    results = []
    render_camera = recording.camera.scaled(4)  # Mesh views are lower-res previews, never original photos.
    for index in choices:
        frame = next(frame for i, frame, _ in recording.iter_frames() if i == index)
        # Use the original source RGB, not decoded video as an independent high-quality reference.
        photo = recording.root / recording.entries[index]["color"]
        target = dest / f"view-{index:02d}-original-rgb-640.png"
        shutil.copyfile(photo, target)
        render, stats = rasterize_mesh_depth(verts, faces, render_camera, frame.camera_to_world)
        values = np.zeros(render.shape, dtype=np.uint8)
        mask = (render >= .15) & (render <= 4)
        values[mask] = np.clip(np.rint(255 - (render[mask] - .15)*254/3.85), 1, 255).astype(np.uint8)
        rendered = dest / f"view-{index:02d}-mesh-depth-160.png"
        Image.fromarray(values).save(rendered)
        results.append({"frame": index, "source_file": target.name,
                        "mesh_projection": rendered.name,
                        "projected_pixels": int(stats["predicted_pixels"]),
                        "view": "same_known_camera_pose"})
    # Two unrestricted cameras around real Mesh center. This is a view, not proof of geometry accuracy.
    center = np.median(verts, axis=0)
    def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        f = target-eye
        f /= np.linalg.norm(f)
        up = np.array([0., 1., 0.])
        if abs(f @ up) > .97:
            up = np.array([0., 0., 1.])
        right = np.cross(f, up)
        right /= np.linalg.norm(right)
        down = np.cross(f, right)
        pose = np.eye(4)
        pose[:3,:3] = np.column_stack((right, down, f))
        pose[:3,3] = eye
        return pose
    for n, delta in enumerate(([1.0, .2, -.5], [-1.0, .4, .8])):
        eye = center+np.array(delta)
        pose = look_at(eye, center)
        render, stats = rasterize_mesh_depth(verts, faces, render_camera, pose)
        values = np.zeros(render.shape, dtype=np.uint8)
        mask = (render >= .15) & (render <= 4)
        values[mask] = np.clip(np.rint(255 - (render[mask] - .15)*254/3.85), 1, 255).astype(np.uint8)
        target = dest / f"free-{n+1}-mesh-depth-160.png"
        Image.fromarray(values).save(target)
        results.append({"view": "free_camera", "file": target.name,
                        "projected_pixels": int(stats["predicted_pixels"])})
    return {"views": results, "mesh_faces": int(len(faces)),
            "mesh_vertices": int(len(verts)), "mesh_sha256": file_sha256(mesh_file),
            "original_rgb_note": "Original 640x480 TUM source; original resolution preserved",
            "mesh_render_note": "160x120 grayscale metric-Z preview, not textured video or an RGB source"}


def real_trial(args: argparse.Namespace, report: dict) -> None:
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset is not None:
        source = args.dataset.resolve()
        if not all((source / n).is_file() for n in ("rgb.txt", "depth.txt", "groundtruth.txt")):
            raise DataError("Expected extracted official TUM xyz source with three indexes")
        report["source"] = "user_supplied_extracted_TUM_dataset_unverified_archive"
    else:
        source_archive = work / "pinned-source.tgz"
        download(source_archive, report)
        report["extraction"] = extract(source_archive, work / "dataset")
        source = work / "dataset" / ROOT
        report["source"] = "downloaded_pinned_official_TUM_rgbd_xyz"

    imported = work / "tum-640px"
    cmd([sys.executable, "-m", "echo4d", "import-tum", str(source),
         "--output", str(imported), "--intrinsics", "525", "525", "319.5", "239.5",
         "--stride", str(STRIDE), "--limit", str(args.frames),
         "--downsample", "1", "--world-up", "z"], timeout=600)
    recording = load_recording(imported / "recording.json")
    if len(recording) != args.frames or recording.camera.width != 640:
        raise DataError("Unexpected true source resolution or frame count")

    reel = work / "tum-observed-frames.webm"
    pts = video_from_real_rgb(imported, recording, reel)
    associations = json.loads((imported / "association.json").read_text("utf8"))
    start = datetime.fromtimestamp(associations["frames"][0]["depth_timestamp"], timezone.utc)
    caption = work / "capture.json"
    caption.write_text(json.dumps({
        "version": "echo-history-stream-0.1",
        "record_kind": "rgbd_camera_view_derived_reel_not_wearable",
        "created_utc": start.isoformat(),
        "time_base": "seconds_since_recording_start_monotonic",
        "duration_s": max(recording.duration, pts[-1])+.1,
        "media": {"file": reel.name, "mime_type": "video/webm",
                  "bytes": reel.stat().st_size, "source": "TUM_real_RGB_reencoded_variable_rate",
                  "actual_camera_pose": "unknown"},
        "imu": {"type": "none", "calibrated": False, "actual_head_pose": "unknown",
                "samples": []},
        "auxiliary_sensor_observations": [],
        "geometry_status": "not_reconstructed",
    }), encoding="utf8")
    archive = work / "history-archive"
    import_history(caption, reel, archive)

    # Use the real imported depth files in their original parent. No path aliases.
    actual_frames = []
    for i, (entry, pts_value) in enumerate(zip(recording.entries, pts)):
        t = entry["t"]
        actual_frames.append({
            "video_frame_index": i, "video_pts_s": pts_value, "t_s": t,
            "depth_t_s": t, "pose_t_s": t,
            "depth": f"depth/{i:05d}.npy",
            "cameraToWorld": entry["cameraToWorld"],
        })
    alignment = imported / "history-alignment-input.json"
    alignment.write_text(json.dumps({
        "version": "echo-history-rgbd-alignment-1",
        "source_media_sha256": file_sha256(reel),
        "clock_sync": {"video_to_history_offset_s": 0,
                       "source": "Derived reel frame-by-frame from original TUM timestamps; +/-30ms reel quantization"},
        "depth_kind": "measured_registered_camera_z",
        "depth_source": "TUM Freiburg1 xyz original Kinect measured depth scaled /5000; RGB registered",
        "pose_source": "TUM provided externally measured mocap groundtruth trajectory with interpolation; not IMU inferred",
        "intrinsics": recording.camera.to_dict(),
        "frames": actual_frames,
    }), encoding="utf8")

    converted = work / "decoded-and-aligned"
    bridge = prepare_history_rgbd(archive, alignment, converted)
    after = load_recording(converted / "recording.json")
    if len(after) != args.frames:
        raise DataError("Historical RGB-D bridge dropped original TUM frames")
    poses_match, depth_match = True, True
    for (_, a, _), (_, b, _) in zip(recording.iter_frames(), after.iter_frames()):
        poses_match &= bool(np.allclose(a.camera_to_world, b.camera_to_world, atol=1e-9))
        depth_match &= bool(np.array_equal(a.depth, b.depth))
    if not poses_match or not depth_match:
        raise DataError("Bridge changed actual measured depth or known camera pose")

    mesh_out = work / "real-tum-mesh"
    cmd([sys.executable, "-m", "echo4d", "reconstruct", str(converted / "recording.json"),
         "--output", str(mesh_out), "--bounds", *BOUNDS, "--voxel", VOXEL,
         "--max-depth", "4", "--min-observations", "2", "--quiet"], timeout=1200)
    final = json.loads((mesh_out / "report.json").read_text())
    if final["mesh"]["vertices"] < 3 or final["mesh"]["triangles"] < 1:
        raise DataError("Real TUM historical bridge produced no usable Mesh triangles")

    shutil.copyfile(mesh_out / "mesh.ply", report_dir / "mesh.ply")
    if (mesh_out / "scene.echo.json").is_file():
        shutil.copyfile(mesh_out / "scene.echo.json", report_dir / "scene.echo.json")
    report["real_rgbd_bridge"] = {
        "source_rgb_resolution": [640,480], "frames": args.frames,
        "pinned_archive_sha256": PINNED_ARCHIVE_SHA256 if args.dataset is None else None,
        "video_sha256": file_sha256(reel), "video_reencode": "lossless_VP9_yuv444p",
        "time_notes": "Derived VFR reel, not a raw continuously recorded wearable camera video",
        "depth_pixels_byte_exact": depth_match, "poses_same": poses_match,
        "bridge": bridge, "mesh": final["mesh"],
        "history_recording_has_no_imu": True,
        "historical_surface_accuracy": "not_measured",
    }
    report["visuals"] = visualize(report_dir, imported, mesh_out / "mesh.ply")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=[30, 60], default=DEFAULT_FRAMES,
                        help="Controlled 30 vs 60 real-frame ablation; same scene, bounds and voxel size")
    parser.add_argument("--dataset", type=Path,
                        help="Previously extracted real TUM freiburg1_xyz data directory (no network)")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    r = {"trial": "real-TUM-freiburg1-xyz-derived-history-to-Mesh-v1",
         "started_utc": datetime.now(timezone.utc).isoformat(),
         "official_source": URL, "source_archive_pin": PINNED_ARCHIVE_SHA256,
         "source_archive_bytes": PINNED_ARCHIVE_BYTES,
         "sample_count": args.frames, "stride": STRIDE, "source_downsample": 1,
         "commit": os.getenv("GITHUB_SHA"), "status": "started"}
    start = time.monotonic()
    try:
        real_trial(args, r)
        r["status"] = "passed_execution_not_human_recognizability"
        code = 0
    except Exception as exc:
        r["status"] = "failed"
        r["error"] = {"type": type(exc).__name__, "message": str(exc)}
        code = 1
        print("REAL_HISTORY_MESH_FAILED: "+repr(exc), file=sys.stderr)
    finally:
        r["elapsed_seconds"] = round(time.monotonic()-start, 2)
        write_json(args.report_dir / "real-history-report.json", r)
        (args.report_dir / "SUMMARY.md").write_text(
            "# Real TUM video-history-to-Mesh execution\n\n"
            f"Status: {r['status']}\n\nCommit: {r['commit']}\n\n"
            f"Real TUM frames: {r['sample_count']} at original 640x480\n\n"
            f"Mesh triangles: {r.get('real_rgbd_bridge',{}).get('mesh',{}).get('triangles','not_produced')}\n\n"
            f"Error: {r.get('error',{}).get('message','none')}\n\n"
            "This is NOT evidence of independently measured surface accuracy or full life reconstruction.\n",
            encoding="utf8")
        print(json.dumps({"status":r["status"],"report":str(args.report_dir/"real-history-report.json")},
                         ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
