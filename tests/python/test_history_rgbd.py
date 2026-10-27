"""Exercise a real encoded video -> frame/depth/pose -> actual ECHO TSDF path.

The video fixture is generated from a simple synthetic room, NOT a human scene
quality claim. FFmpeg is required for this integration test.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from echo4d.history_rgbd import prepare_history_rgbd
from echo4d.model import DataError, file_sha256, load_recording
from echo4d.fusion import FusionConfig, TSDFVolume


@pytest.fixture
def source(tmp_path: Path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required for the real video decoder test")
    src = tmp_path / "original"
    src.mkdir()
    width, height = 64, 48
    for n in range(3):
        pixels = np.full((height, width, 3), (15 + n * 25, 105, 145), dtype=np.uint8)
        pixels[14:35, 18:48] = [180, 60, 40]
        Image.fromarray(pixels).save(src / f"frame_{n:03d}.png")
    media = src / "capture.webm"
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-framerate", "2",
        "-i", str(src / "frame_%03d.png"), "-c:v", "libvpx",
        "-pix_fmt", "yuv420p", "-frames:v", "3", str(media)],
        capture_output=True, text=True, timeout=40)
    if p.returncode:
        pytest.skip("Test FFmpeg has no libvpx encoder: " + p.stderr[-150:])
    archive = tmp_path / "history"
    (archive / "media").mkdir(parents=True)
    shutil.copyfile(media, archive / "media" / "capture.webm")
    (archive / "history.timeline.json").write_text(json.dumps({
        "version": "echo-history-timeline-1", "started_utc": "2026-10-28T00:00:00Z",
        "duration_s": 1.5, "time_base": "seconds_since_recording_start_monotonic",
        "media": {"file": media.name, "archive_relative_path": "media/" + media.name,
                  "bytes": media.stat().st_size, "sha256": file_sha256(media)},
        "geometry_status": "not_reconstructed",
    }))
    depths = tmp_path / "alignment"
    depths.mkdir()
    pose = np.eye(4)
    records = []
    for n in range(3):
        depth = np.full((height, width), 1.3, dtype=np.float32)
        np.save(depths / f"depth_{n}.npy", depth, allow_pickle=False)
        pose[0, 3] = n * 0.02
        records.append({
            "video_frame_index": n, "video_pts_s": n / 2,
            "t_s": n / 2, "depth_t_s": n / 2, "pose_t_s": n / 2,
            "depth": f"depth_{n}.npy",
            "cameraToWorld": pose.flatten(order="F").tolist(),
        })
    alignment = depths / "alignment.json"
    params = {
        "version": "echo-history-rgbd-alignment-1",
        "source_media_sha256": file_sha256(media),
        "clock_sync": {"video_to_history_offset_s": 0,
                       "source": "Externally hardware-synchronized sample fixture"},
        "depth_kind": "measured_registered_camera_z",
        "depth_source": "Fixture camera-Z; test only, NOT measured human scene",
        "pose_source": "Fixture world transforms; test only, NOT real camera pose",
        "intrinsics": {"width": width, "height": height, "fx": 75.,
                       "fy": 75., "cx": 31.5, "cy": 23.5},
        "frames": records,
    }
    alignment.write_text(json.dumps(params), encoding="utf8")
    return archive, alignment, params, media


def test_encoded_history_video_reconstructs_real_mesh_pipeline(source, tmp_path):
    archive, alignment, _, original = source
    output = tmp_path / "rgbd"
    summary = prepare_history_rgbd(archive, alignment, output)
    assert summary["frames"] == 3
    assert summary["geometry"] == "ready_for_known_pose_measured_depth_tsdf"
    assert summary["source_media_sha256"] == file_sha256(original)
    recording = load_recording(output / "recording.json")
    assert recording.source == "recorded"
    assert len(recording) == 3
    for index, frame, _ in recording.iter_frames():
        assert frame.color.shape == (48, 64, 3)
        assert frame.depth.shape == (48, 64)
        assert np.allclose(frame.depth, 1.3)
        assert np.isclose(frame.t, index / 2)
    evidence = json.loads((output / "history-alignment.json").read_text())
    assert [e["video_frame_index"] for e in evidence["frames"]] == [0, 1, 2]
    assert evidence["source_media_sha256"] == file_sha256(original)

    # Real TSDF geometry calculation using the decoded video + sensor-like test Z.
    volume = TSDFVolume(FusionConfig(
        bounds=(-0.8, -0.7, 0.4, 0.8, 0.7, 2.1), voxel=0.07,
        truncation=0.21, max_voxels=100000))
    for index, frame, _ in recording.iter_frames():
        assert volume.integrate(frame, recording.camera, index) > 0
    mesh = volume.extract_mesh()
    assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
    assert not np.all(mesh.colors == [100, 154, 157])


@pytest.mark.parametrize("change", [
    lambda d: d["frames"][1].update(t_s=0.8),
    lambda d: d["frames"][0].update(video_pts_s=0.2),
    lambda d: d["frames"][2].update(pose_t_s=1.08),
    lambda d: d["frames"][1].update(video_frame_index=0),
    lambda d: d.update(depth_kind="predicted_camera_z"),
    lambda d: d["clock_sync"].update(source=""),
    lambda d: d["frames"][0].update(cameraToWorld=[0] * 16),
    lambda d: d.update(source_media_sha256="0"*64),
])
def test_inaccurate_geometry_inputs_fail_without_output(source, tmp_path, change):
    archive, alignment, data, _ = source
    change(data)
    alignment.write_text(json.dumps(data))
    output = tmp_path / "rgbd"
    with pytest.raises(DataError):
        prepare_history_rgbd(archive, alignment, output)
    assert not output.exists()


def test_depth_from_different_pixel_geometry_is_rejected(source, tmp_path):
    archive, alignment, _, _ = source
    p = alignment.parent / "depth_1.npy"
    np.save(p, np.ones((20, 20), dtype=np.float32), allow_pickle=False)
    with pytest.raises(DataError, match="full-resolution"):
        prepare_history_rgbd(archive, alignment, tmp_path / "rgbd")
    assert not (tmp_path / "rgbd").exists()


def test_existing_output_never_overwritten(source, tmp_path):
    archive, alignment, _, _ = source
    output = tmp_path / "rgbd"
    prepare_history_rgbd(archive, alignment, output)
    with pytest.raises(DataError, match="already exists"):
        prepare_history_rgbd(archive, alignment, output)
