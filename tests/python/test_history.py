"""Source-to-timeline acceptance for real browser-shaped bytes and timed sensor data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from echo4d.history import import_history
from echo4d.model import DataError, file_sha256


def fixture(tmp_path: Path):
    media = tmp_path / "real-capture.webm"
    media.write_bytes(b"\x1aE\xdf\xa3media fragment that came from recorder")
    capture = tmp_path / "real-capture.json"
    manifest = {
        "version": "echo-history-stream-0.1",
        "record_kind": "first_person_forward_camera_candidate",
        "created_utc": "2026-10-28T09:00:00.000Z",
        "time_base": "seconds_since_recording_start_monotonic",
        "duration_s": 3.5,
        "media": {
            "file": media.name, "mime_type": "video/webm",
            "bytes": media.stat().st_size, "role": "camera_original_compressed_capture",
            "actual_camera_pose": "unknown",
        },
        "imu": {
            "type": "browser_DeviceMotionEvent_optional",
            "calibrated": False, "actual_head_pose": "unknown",
            "samples": [
                {"t_s": 0.05, "acceleration_including_gravity_m_s2": [0.0, 9.8, 0.0],
                 "acceleration_without_gravity_m_s2": None,
                 "rotation_rate_deg_s": {"alpha": 1.0, "beta": None, "gamma": 0.0}},
                {"t_s": 2.5, "acceleration_including_gravity_m_s2": [0.1, 9.7, 0.2],
                 "acceleration_without_gravity_m_s2": None, "rotation_rate_deg_s": None},
            ],
        },
        "auxiliary_sensor_observations": [],
        "geometry_status": "not_reconstructed",
    }
    capture.write_text(json.dumps(manifest), encoding="utf-8")
    return capture, media, manifest


def test_browser_capture_to_real_portable_history(tmp_path):
    path, media, capture = fixture(tmp_path)
    original_sha = file_sha256(media)
    dest = tmp_path / "archive"
    result = import_history(path, media, dest)
    exported = json.loads((dest / "history.timeline.json").read_text())
    assert result["media_sha256"] == original_sha
    assert result["imu_samples"] == 2
    assert result["auxiliary_streams"] == 0
    assert result["reconstruction"] == "not_attempted"
    assert exported["version"] == "echo-history-timeline-1"
    assert exported["media"]["file"] == media.name
    assert exported["media"]["camera_pose"] == "unknown"
    assert exported["imu"]["head_pose"] == "unknown"
    assert exported["imu"]["samples"] == capture["imu"]["samples"]
    assert exported["geometry_status"] == "not_reconstructed"
    assert (dest / "media" / media.name).read_bytes() == media.read_bytes()
    assert (dest / "capture-source.json").is_file()


def test_auxiliary_depth_and_near_field_sources_remain_separate(tmp_path):
    path, media, _ = fixture(tmp_path)
    scans = []
    for ident, kind, time_base in (
        ("depth-a", "measured_depth", "seconds_since_recording_start_monotonic"),
        ("near-b", "near_field_observation", "device_clock_unaligned"),
    ):
        p = tmp_path / (ident + ".json")
        p.write_text(json.dumps({
            "version": "echo-history-aux-0.1", "sensor_id": ident, "kind": kind,
            "time_base": time_base, "units": "sensor_units",
            "spatial_calibration": "unknown",
            "samples": [{"t_s": 0.2, "values": [0.4, 1.3]},
                        {"t_s": 1.6, "values": [0.5, 1.2]}],
        }), encoding="utf8")
        scans.append(p)
    result = import_history(path, media, tmp_path / "archive", tuple(scans))
    t = json.loads((tmp_path / "archive" / "history.timeline.json").read_text())
    assert result["auxiliary_streams"] == 2
    assert t["auxiliary_streams"][0]["clock_status"] == "source_claim_same_clock_unverified"
    assert t["auxiliary_streams"][1]["clock_status"] == "unaligned"
    assert t["geometry_status"] == "not_reconstructed"
    assert (tmp_path / "archive" / "sensors" / "depth-a.json").is_file()


@pytest.mark.parametrize("mutation", [
    lambda c: c["media"].update(file="../escape.webm"),
    lambda c: c["media"].update(bytes=999999),
    lambda c: c["imu"]["samples"][1].update(t_s=0.01),
    lambda c: c["imu"]["samples"][0].update(acceleration_including_gravity_m_s2=[float("nan"), 0, 0]),
    lambda c: c.update(geometry_status="reconstructed"),
    lambda c: c.update(time_base="unknown"),
    lambda c: c["imu"].update(calibrated=True),
])
def test_bad_capture_fails_without_publishing_archive(tmp_path, mutation):
    path, media, manifest = fixture(tmp_path)
    mutation(manifest)
    path.write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(DataError):
        import_history(path, media, tmp_path / "archive")
    assert not (tmp_path / "archive").exists()


def test_no_silent_overwrite_and_empty_imu(tmp_path):
    path, media, manifest = fixture(tmp_path)
    manifest["imu"]["samples"] = []
    path.write_text(json.dumps(manifest), encoding="utf8")
    output = tmp_path / "archive"
    import_history(path, media, output)
    with pytest.raises(DataError, match="already exists"):
        import_history(path, media, output)
    data = json.loads((output / "history.timeline.json").read_text())
    assert data["imu"]["samples"] == []


def test_unsynced_stream_does_not_claim_time_alignment(tmp_path):
    path, media, _ = fixture(tmp_path)
    sensor = tmp_path / "aux.json"
    sensor.write_text(json.dumps({
        "version": "echo-history-aux-0.1", "sensor_id": "laser-a",
        "kind": "scan", "units": "raw",
        "time_base": "device_clock_unaligned", "spatial_calibration": "unknown",
        "samples": [{"t_s": 30, "values": [1, 2, 3]}],
    }), encoding="utf-8")
    import_history(path, media, tmp_path / "archive", (sensor,))
    data = json.loads((tmp_path / "archive" / "history.timeline.json").read_text())
    assert data["auxiliary_streams"][0]["clock_status"] == "unaligned"
    assert data["auxiliary_streams"][0]["first_t_s"] == 30


def test_auxiliary_invalid_values_rejected(tmp_path):
    path, media, _ = fixture(tmp_path)
    sensor = tmp_path / "bad.json"
    sensor.write_text(json.dumps({
        "version": "echo-history-aux-0.1", "sensor_id": "scanner",
        "kind": "measured_depth", "units": "meters",
        "time_base": "seconds_since_recording_start_monotonic",
        "spatial_calibration": "unknown",
        "samples": [{"t_s": 0.1, "values": ["not-a-distance"]}],
    }))
    with pytest.raises(DataError):
        import_history(path, media, tmp_path / "archive", (sensor,))
    assert not (tmp_path / "archive").exists()
