"""Fast negative tests: simulated executable output is not real SfM evidence."""
import json
from pathlib import Path
from unittest.mock import patch
import subprocess
import pytest
from echo4d.model import DataError
from echo4d.video_sfm import _probe_timestamps, run


def _video(tmp_path):
    file = tmp_path / "input.mp4"
    file.write_bytes(b"fixture, not real video")
    return file


def _result(stdout="", code=0):
    return subprocess.CompletedProcess([], code, stdout, "")


def test_nonincreasing_timestamp_rejected(tmp_path):
    report = {"commands": []}
    with patch("echo4d.video_sfm.subprocess.run", return_value=_result(
            '{"frames":[{"best_effort_timestamp_time":"0"},'
            '{"best_effort_timestamp_time":"0"}]}')):
        with pytest.raises(DataError, match="strictly increasing"):
            _probe_timestamps(_video(tmp_path), report, "ffprobe")


def test_invalid_timestamp_rejected(tmp_path):
    with patch("echo4d.video_sfm.subprocess.run", return_value=_result(
            '{"frames":[{"best_effort_timestamp_time":"0"},{"other":1}]}')):
        with pytest.raises(DataError, match="Missing or invalid"):
            _probe_timestamps(_video(tmp_path), {"commands": []}, "ffprobe")


def test_failed_ffprobe_preserves_report(tmp_path):
    with patch("echo4d.video_sfm.subprocess.run", return_value=_result(code=1)):
        with pytest.raises(DataError, match="ffprobe exited"):
            run(_video(tmp_path), tmp_path / "result")
    report = json.loads((tmp_path / "result/report.json").read_text())
    assert report["status"] == "failed"
    assert report["commands"][0]["status"] == "failed"
    assert not (tmp_path / "result/trajectory.json").exists()


def test_existing_output_not_overwritten(tmp_path):
    dest = tmp_path / "result"
    dest.mkdir()
    marker = dest / "original"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        run(_video(tmp_path), dest)
    assert marker.read_text() == "keep"


def test_frame_count_mismatch_rejected(tmp_path):
    def fake_run(cmd, **kwargs):
        if "ffprobe" in str(cmd[0]):
            return _result('{"frames":[{"best_effort_timestamp_time":"0"},'
                           '{"best_effort_timestamp_time":"0.1"}]}')
        return _result()
    with patch("echo4d.video_sfm.subprocess.run", side_effect=fake_run):
        with pytest.raises(DataError, match="extracted frames do not match"):
            run(_video(tmp_path), tmp_path / "result")
    assert json.loads((tmp_path / "result/report.json").read_text())["status"] == "failed"


def test_max_frames_rejected(tmp_path):
    with pytest.raises(DataError, match="max_frames"):
        run(_video(tmp_path), tmp_path / "result", max_frames=1)
    assert not (tmp_path / "result").exists()
