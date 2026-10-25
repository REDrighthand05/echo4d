"""Known-value roundtrip and adversarial checks for the COLMAP adapter."""
from argparse import Namespace
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from echo4d.colmap import import_colmap
from echo4d.model import DataError, load_recording


def fixture(tmp_path: Path) -> Namespace:
    model = tmp_path / "model"
    rgb = tmp_path / "rgb"
    depth = tmp_path / "metric-depth"
    for directory in (model, rgb, depth):
        directory.mkdir()
    (model / "cameras.txt").write_text(
        "# undistorted pinhole\n1 PINHOLE 8 6 12 13 3.5 2.5\n")
    # Image 1 has a +90 degree world-to-camera Z rotation.
    # C=[1,2,3], so t_wc=[2,-1,-3]. After scale 2, C=[2,4,6] meters.
    (model / "images.txt").write_text(
        "# COLMAP image poses\n"
        "1 0.7071067811865476 0 0 0.7071067811865475 2 -1 -3 1 a.png\n\n"
        "2 1 0 0 0 -2 0 0 1 b.png\n\n")
    Image.fromarray(np.full((6, 8, 3), 128, dtype=np.uint8)).save(rgb / "a.png")
    Image.fromarray(np.full((6, 8, 3), 180, dtype=np.uint8)).save(rgb / "b.png")
    np.save(depth / "a.npy", np.full((6, 8), 1.5, dtype=np.float32))
    np.save(depth / "b.npy", np.full((6, 8), 2.5, dtype=np.float32))
    stamps = tmp_path / "timestamps.csv"
    with stamps.open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["name", "t"])
        w.writerow(["b.png", 9.25])
        w.writerow(["a.png", 9.0])
    return Namespace(model=model, images=rgb, depth=depth, timestamps=stamps,
                     meters_per_colmap_unit=2.0, scale_source="measured two-meter target",
                     output=tmp_path / "converted")


def test_round_trip_rotation_translation_scale_and_time_order(tmp_path):
    args = fixture(tmp_path)
    result = import_colmap(args)
    assert result["frames"] == 2
    rec = load_recording(args.output / "recording.json")
    frames = list(rec.iter_frames())
    assert len(frames) == 2
    assert frames[0][1].t == 0.0
    assert frames[1][1].t == pytest.approx(0.25)
    assert np.allclose(frames[0][1].camera_to_world[:3, 3], [2, 4, 6], atol=1e-12)
    assert np.allclose(frames[0][1].camera_to_world[:3, :3],
                       [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert np.allclose(frames[1][1].camera_to_world[:3, 3], [4, 0, 0])
    assert np.allclose(frames[0][1].depth, 1.5)
    assert frames[0][1].color.shape == (6, 8, 3)
    provenance = json.loads((args.output / "import-provenance.json").read_text())
    assert provenance["scale_meters_per_colmap_unit"] == 2
    assert provenance["scale_source"].startswith("measured")


def test_scale_evidence_cannot_be_omitted(tmp_path):
    args = fixture(tmp_path)
    args.scale_source = ""
    with pytest.raises(DataError, match="scale-source"):
        import_colmap(args)
    assert not args.output.exists()


@pytest.mark.parametrize("model", [
    "1 OPENCV 8 6 12 13 3.5 2.5 0.1 0.1 0 0\n",
    "1 SIMPLE_RADIAL 8 6 12 3.5 2.5 0.1\n",
])
def test_distorted_camera_rejected(tmp_path, model):
    args = fixture(tmp_path)
    (args.model / "cameras.txt").write_text(model)
    with pytest.raises(DataError, match="undistorted"):
        import_colmap(args)
    assert not args.output.exists()


def test_integer_millimeter_depth_rejected(tmp_path):
    args = fixture(tmp_path)
    np.save(args.depth / "a.npy", np.full((6, 8), 1500, dtype=np.uint16))
    with pytest.raises(DataError, match="floating-point metric"):
        import_colmap(args)
    assert not args.output.exists()


def test_missing_timestamp_rejected(tmp_path):
    args = fixture(tmp_path)
    args.timestamps.write_text("name,t\na.png,9\n")
    with pytest.raises(DataError, match="exactly"):
        import_colmap(args)
    assert not args.output.exists()


def test_unknown_camera_rejected(tmp_path):
    args = fixture(tmp_path)
    path = args.model / "images.txt"
    path.write_text(path.read_text().replace(" -2 0 0 1 b.png", " -2 0 0 5 b.png"))
    with pytest.raises(DataError, match="single known"):
        import_colmap(args)
    assert not args.output.exists()


def test_unsafe_path_rejected(tmp_path):
    args = fixture(tmp_path)
    path = args.model / "images.txt"
    path.write_text(path.read_text().replace("a.png", "../a.png"))
    with pytest.raises(DataError, match="Unsafe"):
        import_colmap(args)
    assert not args.output.exists()


def test_image_size_mismatch_rejected(tmp_path):
    args = fixture(tmp_path)
    Image.fromarray(np.ones((7, 8, 3), dtype=np.uint8)).save(args.images / "a.png")
    with pytest.raises(DataError, match="Image size"):
        import_colmap(args)
    assert not args.output.exists()


def test_existing_destination_not_overwritten(tmp_path):
    args = fixture(tmp_path)
    args.output.mkdir()
    sentinel = args.output / "preserve"
    sentinel.write_text("original")
    with pytest.raises((DataError, FileExistsError, OSError)):
        import_colmap(args)
    assert sentinel.read_text() == "original"


def test_duplicate_timestamp_rejected(tmp_path):
    args = fixture(tmp_path)
    args.timestamps.write_text("name,t\na.png,9\nb.png,9\n")
    with pytest.raises(DataError, match="unique"):
        import_colmap(args)
    assert not args.output.exists()
