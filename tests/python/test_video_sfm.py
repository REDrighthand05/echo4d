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


# These cases exercise disconnected real-world COLMAP model selection with
# deterministic text fixtures. They do not emulate feature matching quality.
def _sparse_fixture(tmp_path, candidate_frames, monkeypatch, failed=()):
    from echo4d import video_sfm
    sparse, output = tmp_path / "sparse", tmp_path / "output"
    sparse.mkdir()
    output.mkdir()
    for model_id in candidate_frames:
        (sparse / str(model_id)).mkdir()
    names = [f"{i:06d}.png" for i in range(1, 6)]
    lookup = {name: {"source_pts_seconds": float(i)}
              for i, name in enumerate(names)}
    def fake_converter(label, argv, report):
        model_id = int(Path(argv[argv.index("--input_path") + 1]).name)
        if model_id in failed:
            raise DataError("simulated model_converter rejection")
        text_dir = Path(argv[argv.index("--output_path") + 1])
        (text_dir / "cameras.txt").write_text("1 PINHOLE 8 6 12 13 3.5 2.5\n")
        images = "".join(
            f"{i} 1 0 0 0 {i} 0 0 1 {name}\n\n"
            for i, name in enumerate(candidate_frames[model_id], 1))
        (text_dir / "images.txt").write_text(images)
    monkeypatch.setattr(video_sfm, "_exec", fake_converter)
    return sparse, output, lookup


def test_selects_largest_valid_model_instead_of_zero(tmp_path, monkeypatch):
    from echo4d.video_sfm import _select_colmap_model
    sparse, out, lookup = _sparse_fixture(
        tmp_path, {0: ["000001.png", "000002.png"],
                   1: ["000001.png", "000002.png", "000003.png"]}, monkeypatch)
    report = {"commands": []}
    poses = _select_colmap_model(sparse, out, lookup, report, "colmap")
    assert len(poses) == 3
    assert report["selected_model_id"] == "1"
    assert [item["registered_images"] for item in report["model_candidates"]] == [2, 3]
    assert all(item["status"] == "valid" for item in report["model_candidates"])


def test_invalid_first_model_kept_in_report_and_skipped(tmp_path, monkeypatch):
    from echo4d.video_sfm import _select_colmap_model
    sparse, out, lookup = _sparse_fixture(
        tmp_path, {0: ["000001.png", "000002.png"],
                   1: ["000001.png", "000003.png"]}, monkeypatch, failed={0})
    report = {"commands": []}
    assert len(_select_colmap_model(sparse, out, lookup, report, "colmap")) == 2
    assert report["selected_model_id"] == "1"
    assert report["model_candidates"][0]["status"] == "rejected"
    assert "simulated" in report["model_candidates"][0]["reason"]


def test_model_with_unknown_extracted_frame_is_rejected(tmp_path, monkeypatch):
    from echo4d.video_sfm import _select_colmap_model
    sparse, out, lookup = _sparse_fixture(
        tmp_path, {0: ["000001.png", "999999.png", "000003.png"],
                   1: ["000001.png", "000002.png"]}, monkeypatch)
    report = {"commands": []}
    assert len(_select_colmap_model(sparse, out, lookup, report, "colmap")) == 2
    assert report["selected_model_id"] == "1"
    assert "unknown image" in report["model_candidates"][0]["reason"]


def test_equal_registration_prefers_lowest_numeric_model_id(tmp_path, monkeypatch):
    from echo4d.video_sfm import _select_colmap_model
    sparse, out, lookup = _sparse_fixture(
        tmp_path, {10: ["000001.png", "000002.png"],
                   2: ["000001.png", "000003.png"]}, monkeypatch)
    report = {"commands": []}
    _select_colmap_model(sparse, out, lookup, report, "colmap")
    assert report["selected_model_id"] == "2"
    assert [entry["model_id"] for entry in report["model_candidates"]] == ["2", "10"]


def test_no_valid_model_raises_and_preserves_reasons(tmp_path, monkeypatch):
    from echo4d.video_sfm import _select_colmap_model
    sparse, out, lookup = _sparse_fixture(
        tmp_path, {0: ["000001.png"], 1: ["000002.png"]}, monkeypatch)
    report = {"commands": []}
    with pytest.raises(DataError, match="No valid COLMAP model"):
        _select_colmap_model(sparse, out, lookup, report, "colmap")
    assert [entry["registered_images"] for entry in report["model_candidates"]] == [1, 1]
    assert all(entry["status"] == "rejected" for entry in report["model_candidates"])


def test_too_many_disconnected_models_rejected(tmp_path):
    from echo4d.video_sfm import _select_colmap_model
    sparse, out = tmp_path / "sparse", tmp_path / "output"
    sparse.mkdir()
    out.mkdir()
    for i in range(33):
        (sparse / str(i)).mkdir()
    with pytest.raises(DataError, match="more than 32"):
        _select_colmap_model(sparse, out, {}, {"commands": []}, "colmap")
