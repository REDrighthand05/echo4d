"""Indexed-depth compatibility is a front-end to the #6/#24 COLMAP importer."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image
import pytest

from echo4d.cli import main
from echo4d.model import load_recording


def fixture(tmp_path):
    for name in ("model", "rgb", "raw"):
        (tmp_path / name).mkdir()
    (tmp_path / "model" / "cameras.txt").write_text(
        "1 PINHOLE 8 6 12 13 3.5 2.5\n", encoding="utf-8")
    # 90-degree world-to-camera Z rotation and C=[1,2,3] in SfM units.
    (tmp_path / "model" / "images.txt").write_text(
        "1 0.7071067811865476 0 0 0.7071067811865475 2 -1 -3 1 a.png\n\n"
        "2 1 0 0 0 -2 0 0 1 b.png\n\n", encoding="utf-8")
    for name in ("a.png", "b.png"):
        Image.fromarray(np.full((6, 8, 3), 125, np.uint8)).save(tmp_path / "rgb" / name)
    Image.fromarray(np.full((6, 8), 1500, np.uint16)).save(tmp_path / "raw" / "first.png")
    Image.fromarray(np.full((6, 8), 2500, np.uint16)).save(tmp_path / "raw" / "second.png")
    (tmp_path / "times.csv").write_text("name,t\nb.png,9.25\na.png,9\n", encoding="utf-8")
    (tmp_path / "index.txt").write_text(
        "a.png first.png\nb.png second.png\n", encoding="utf-8")


def argv(tmp_path, *, divisor="1000", indexed=True, scale="2", source="measured ruler in scene"):
    args = ["import-colmap", "--model", str(tmp_path / "model"),
            "--images", str(tmp_path / "rgb"),
            "--depth", str(tmp_path / "raw"),
            "--timestamps", str(tmp_path / "times.csv"),
            "--meters-per-colmap-unit", scale,
            "--scale-source", source,
            "--output", str(tmp_path / "published")]
    if indexed:
        args += ["--depth-index", str(tmp_path / "index.txt")]
    if divisor is not None:
        args += ["--depth-scale", divisor]
    return args


def failure(tmp_path, capsys, expected, **options):
    assert main(argv(tmp_path, **options)) == 2
    response = capsys.readouterr()
    assert not response.out.strip()
    error = json.loads(response.err)
    assert expected.lower() in error["error"].lower()
    assert not (tmp_path / "published").exists()


def test_indexed_png_success_known_pose_scale_no_double_scaled_depth(tmp_path, capsys):
    fixture(tmp_path)
    assert main(argv(tmp_path)) == 0
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["frames"] == 2 and outcome["indexed_depth"] is True
    rec = load_recording(tmp_path / "published" / "recording.json")
    frames = list(rec.iter_frames())
    assert [entry[1].t for entry in frames] == pytest.approx([0, 0.25])
    assert np.allclose(frames[0][1].camera_to_world[:3, 3], [2, 4, 6])
    assert np.allclose(frames[0][1].camera_to_world[:3, :3],
                       [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(frames[0][1].depth, 1.5)
    assert np.allclose(frames[1][1].depth, 2.5)
    provenance = json.loads((tmp_path / "published" / "import-provenance.json").read_text())
    assert provenance["scale_meters_per_colmap_unit"] == 2
    assert provenance["scale_source"] == "measured ruler in scene"
    indexed = provenance["indexed_depth"]
    assert indexed["raw_units_per_metre"] == 1000
    assert len(indexed["frames"]) == 2
    assert [item["timestamp_seconds"] for item in indexed["frames"]] == [9, 9.25]
    assert all(len(item["raw_depth_sha256"]) == 64 and
               len(item["color_sha256"]) == 64 and
               len(item["metric_depth_sha256"]) == 64 for item in indexed["frames"])


def test_float_metric_npy_must_use_divisor_one(tmp_path, capsys):
    fixture(tmp_path)
    np.save(tmp_path / "raw" / "a.npy", np.full((6, 8), 1.5, dtype=np.float32))
    np.save(tmp_path / "raw" / "b.npy", np.full((6, 8), 2.5, dtype=np.float64))
    (tmp_path / "index.txt").write_text("a.png a.npy\nb.png b.npy\n")
    failure(tmp_path, capsys, "already in metres")
    assert main(argv(tmp_path, divisor="1")) == 0
    capsys.readouterr()
    frames = list(load_recording(tmp_path / "published" / "recording.json").iter_frames())
    assert np.allclose(frames[0][1].depth, 1.5)
    assert np.allclose(frames[1][1].depth, 2.5)


@pytest.mark.parametrize("index,reason", [
    ("a.png first.png\n", "exactly"),
    ("a.png first.png\na.png second.png\n", "Duplicate"),
    ("a.png first.png\nb.png ../secret.npy\n", "Unsafe"),
    ("a.png first.png\nb.png first.png\n", "Duplicate"),
    ("a.png first.png\nb.png second.png\nother.png third.png\n", "exactly"),
    ("a.png first.png\nb.png second.jpg\n", "Unsafe"),
])
def test_index_validation_fails_closed(tmp_path, capsys, index, reason):
    fixture(tmp_path)
    (tmp_path / "index.txt").write_text(index)
    failure(tmp_path, capsys, reason)


@pytest.mark.parametrize("divisor", [None, "0", "-1000", "nan", "inf"])
def test_missing_or_invalid_depth_divisor_rejected(tmp_path, capsys, divisor):
    fixture(tmp_path)
    failure(tmp_path, capsys, "depth-scale", divisor=divisor)


def test_missing_independent_pose_scale_source_rejected(tmp_path, capsys):
    fixture(tmp_path)
    failure(tmp_path, capsys, "scale-source", source="")


def test_raw_depth_wrong_resolution_fails_transaction(tmp_path, capsys):
    fixture(tmp_path)
    Image.fromarray(np.full((5, 8), 1500, np.uint16)).save(tmp_path / "raw" / "first.png")
    failure(tmp_path, capsys, "camera-sized")


def test_archive_disguised_as_npy_rejected_without_output(tmp_path, capsys):
    fixture(tmp_path)
    with (tmp_path / "raw" / "first.npy").open("wb") as stream:
        np.savez(stream, z=np.ones((6, 8)))
    (tmp_path / "index.txt").write_text("a.png first.npy\nb.png second.png\n")
    failure(tmp_path, capsys, "NPZ archive")


def test_nonfinite_or_negative_depth_rejected(tmp_path, capsys):
    fixture(tmp_path)
    arr = np.full((6, 8), 1.5, dtype=np.float32)
    arr[0, 0] = np.nan
    np.save(tmp_path / "raw" / "first.npy", arr)
    (tmp_path / "index.txt").write_text("a.png first.npy\nb.png second.png\n")
    failure(tmp_path, capsys, "finite", divisor="1")


def test_core_depth_contract_rejects_stray_divisor_without_index(tmp_path, capsys):
    fixture(tmp_path)
    failure(tmp_path, capsys, "requires --depth-index", indexed=False)


def test_never_overwrite_destination(tmp_path, capsys):
    fixture(tmp_path)
    out = tmp_path / "published"
    out.mkdir()
    (out / "sentinel").write_text("keep")
    assert main(argv(tmp_path)) == 2
    assert "already exists" in json.loads(capsys.readouterr().err)["error"]
    assert (out / "sentinel").read_text() == "keep"
