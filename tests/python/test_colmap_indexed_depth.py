"""End-to-end CLI checks for indexed integer/float depth preparation."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from echo4d.cli import main
from echo4d.model import load_recording


def _fixture(tmp_path, *, kind="png"):
    for name in ("model", "images", "raw"):
        (tmp_path / name).mkdir()
    (tmp_path / "model" / "cameras.txt").write_text(
        "1 PINHOLE 8 6 12 13 3.5 2.5\n", encoding="utf-8")
    (tmp_path / "model" / "images.txt").write_text(
        "1 1 0 0 0 -1 -2 -3 1 a.png\n\n"
        "2 1 0 0 0 -2 -2 -3 1 b.png\n\n", encoding="utf-8")
    (tmp_path / "times.csv").write_text(
        "name,t\na.png,9\nb.png,10\n", encoding="utf-8")
    for name in ("a", "b"):
        Image.fromarray(np.full((6, 8, 3), 128, np.uint8)).save(
            tmp_path / "images" / f"{name}.png")
    if kind == "png":
        Image.fromarray(np.full((6, 8), 1500, np.uint16)).save(
            tmp_path / "raw" / "unrelated-01.png")
        Image.fromarray(np.full((6, 8), 2000, np.uint16)).save(
            tmp_path / "raw" / "unrelated-02.png")
    else:
        np.save(tmp_path / "raw" / "unrelated-01.npy",
                np.full((6, 8), 1.5, np.float32))
        np.save(tmp_path / "raw" / "unrelated-02.npy",
                np.full((6, 8), 2.0, np.float32))
    (tmp_path / "index.csv").write_text(
        "name,depth_path\n"
        f"a.png,unrelated-01.{kind}\n"
        f"b.png,unrelated-02.{kind}\n", encoding="utf-8")
    return [
        "import-colmap-indexed",
        "--model", str(tmp_path / "model"),
        "--images", str(tmp_path / "images"),
        "--depth-root", str(tmp_path / "raw"),
        "--depth-index", str(tmp_path / "index.csv"),
        "--depth-unit", "millimeter" if kind == "png" else "meter",
        "--depth-units-per-meter", "1000" if kind == "png" else "1",
        "--timestamps", str(tmp_path / "times.csv"),
        "--meters-per-colmap-unit", "2",
        "--scale-source", "independent physical reference anchors",
        "--output", str(tmp_path / "output"),
    ]


@pytest.mark.parametrize("kind", ["png", "npy"])
def test_indexed_cli_converts_once_and_delegates_to_strict_core(tmp_path, capsys, kind):
    args = _fixture(tmp_path, kind=kind)
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["frames"] == 2
    recording_path = tmp_path / "output" / "recording.json"
    recording = load_recording(recording_path)
    frames = [f for _, f, _ in recording.iter_frames()]
    # COLMAP centers are inverted, then independently measured scale ×2.
    assert np.allclose(frames[0].camera_to_world[:3, 3], [2, 4, 6])
    assert np.allclose(frames[1].camera_to_world[:3, 3], [4, 4, 6])
    # Raw 1500/2000 millimetres -> 1.5/2.0 metres, NOT ×2 again.
    assert np.allclose(frames[0].depth, 1.5)
    assert np.allclose(frames[1].depth, 2.0)
    assert json.loads((tmp_path / "output" / "import-provenance.json").read_text())[
        "scale_meters_per_colmap_unit"] == 2
    provenance = json.loads(
        (tmp_path / "output" / "indexed-depth-provenance.json").read_text())
    assert provenance["raw_depth_units_per_meter"] == (1000 if kind == "png" else 1)
    assert len(provenance["frames"]) == 2
    assert all(len(f["raw_depth_sha256"]) == 64 for f in provenance["frames"])
    assert main(["validate", str(recording_path)]) == 0


def _fails_cleanly(args, output, capsys, message):
    assert main(args) == 2
    diagnostic = json.loads(capsys.readouterr().err)
    assert message.lower() in diagnostic["error"].lower()
    assert not output.exists()


@pytest.mark.parametrize("changed_index,expected", [
    ("name,depth_path\na.png,unrelated-01.png\n", "match registered images"),
    ("name,depth_path\na.png,unrelated-01.png\nb.png,unrelated-02.png\nc.png,unrelated-01.png\n", "match registered images"),
    ("name,depth_path\na.png,unrelated-01.png\na.png,unrelated-02.png\n", "Duplicate"),
    ("name,depth_path\na.png,../escape.png\nb.png,unrelated-02.png\n", "unsafe"),
    ("name,depth_path\na.png,unrelated-01.png\nb.png,unrelated-01.png\n", "cannot be assigned"),
])
def test_bad_index_has_no_published_output(tmp_path, capsys, changed_index, expected):
    args = _fixture(tmp_path)
    (tmp_path / "index.csv").write_text(changed_index, encoding="utf-8")
    _fails_cleanly(args, tmp_path / "output", capsys, expected)


def test_wrong_unit_rejected_without_output(tmp_path, capsys):
    args = _fixture(tmp_path)
    where = args.index("--depth-units-per-meter") + 1
    args[where] = "10"  # 1500 raw / 10 = invalid 150 m.
    _fails_cleanly(args, tmp_path / "output", capsys, "invalid values")


def test_bad_second_frame_rolls_back_first_conversion(tmp_path, capsys):
    args = _fixture(tmp_path)
    Image.fromarray(np.ones((3, 3), np.uint16)).save(
        tmp_path / "raw" / "unrelated-02.png")
    _fails_cleanly(args, tmp_path / "output", capsys, "shape")


def test_non_integer_png_rejected(tmp_path, capsys):
    args = _fixture(tmp_path)
    Image.fromarray(np.zeros((6, 8, 3), np.uint8)).save(
        tmp_path / "raw" / "unrelated-01.png")
    _fails_cleanly(args, tmp_path / "output", capsys, "single-channel integer")


def test_strict_core_failure_rollback(tmp_path, capsys):
    args = _fixture(tmp_path)
    (tmp_path / "times.csv").write_text("name,t\na.png,9\n", encoding="utf-8")
    _fails_cleanly(args, tmp_path / "output", capsys, "Timestamp CSV")


def test_reject_existing_output_without_overwrite(tmp_path, capsys):
    args = _fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "sentinel.txt").write_text("do not overwrite")
    assert main(args) == 2
    diagnostic = json.loads(capsys.readouterr().err)
    assert "already exists" in diagnostic["error"]
    assert (output / "sentinel.txt").read_text() == "do not overwrite"
    assert sorted(p.name for p in output.iterdir()) == ["sentinel.txt"]
