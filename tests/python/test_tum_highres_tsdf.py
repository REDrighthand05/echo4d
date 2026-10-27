"""Minimum quality experiment guards; synthetic fixture, not real quality evidence."""
import json
from pathlib import Path
import numpy as np
import pytest
from echo4d.model import Camera, DataError
from tum_highres_tsdf import CONFIGS, VIEWS, contact_sheet, verify_same_source


def source(root: Path, downsample: int, offset: float = 0.) -> None:
    root.mkdir()
    (root / "unused.npy").write_bytes(b"fixture-not-read")
    camera = Camera(640, 480, 525., 525., 319.5, 239.5).scaled(downsample)
    frames = [{"t": float(i) + offset,
               "cameraToWorld": np.eye(4).flatten(order="F").tolist(),
               "depth": "unused.npy"} for i in range(60)]
    (root / "recording.json").write_text(json.dumps({
        "version": "echo-dataset-1", "source": "recorded", "units": "meters",
        "name": "fixture", "duration": 60., "depth_scale": 1,
        "intrinsics": camera.to_dict(), "frames": frames}), encoding="utf8")
    (root / "association.json").write_text(json.dumps({
        "frames": [{"index": i} for i in range(60)]}), encoding="utf8")


def test_fixed_resolutions_and_holdout(tmp_path):
    low, high = tmp_path / "low", tmp_path / "high"
    source(low, 4)
    source(high, 2)
    assert verify_same_source(low, high)["same_poses_and_timestamps"]
    assert [CONFIGS[k]["downsample"] for k in CONFIGS] == [4, 2]
    assert VIEWS == (4, 24, 55)


def test_different_source_timestamps_rejected(tmp_path):
    low, high = tmp_path / "low", tmp_path / "high"
    source(low, 4)
    source(high, 2, .02)
    with pytest.raises(DataError, match="timestamp or camera pose"):
        verify_same_source(low, high)


def test_contact_sheet_keeps_unknown_black(tmp_path):
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    contact_sheet(None, blank, blank, tmp_path / "valid.png", "Free view")
    from PIL import Image
    with Image.open(tmp_path / "valid.png") as im:
        assert im.size == (1920, 514)
        assert im.getpixel((1000, 200)) == (0, 0, 0)
    with pytest.raises(DataError, match="320x240"):
        contact_sheet(None, blank[:100], blank, tmp_path / "invalid.png", "Invalid")
