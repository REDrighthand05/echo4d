"""Small guards for second real room depth-cutoff experiment (not visual acceptance)."""
import numpy as np
import pytest
from PIL import Image

from redwood_bedroom_depth_cutoff_ab import (
    BASELINE_DEPTH_M, CANDIDATE_DEPTH_M, make_rgbd, panel,
)


class _RGBD:
    def __init__(self):
        self.args = None

    def create_from_color_and_depth(self, *args, **kwargs):
        self.args = (args, kwargs)
        return "measured-frame"


class _FakeOpen3D:
    def __init__(self):
        self.rgbd = _RGBD()
        self.geometry = type("Geometry", (), {"RGBDImage": self.rgbd})()


def test_real_depth_truncation_is_explicit_and_no_upsampling():
    fake = _FakeOpen3D()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640), dtype=np.uint16)
    k = (640, 480, 525., 525., 319.5, 239.5)
    assert make_rgbd(fake, rgb, depth, k, CANDIDATE_DEPTH_M) == "measured-frame"
    assert fake.rgbd.args[1]["depth_trunc"] == 3.0
    assert fake.rgbd.args[1]["depth_scale"] == 1000.0
    assert BASELINE_DEPTH_M == 4.5
    with pytest.raises(ValueError, match="RGB dimensions"):
        make_rgbd(fake, rgb[::2], depth, k, CANDIDATE_DEPTH_M)
    with pytest.raises(ValueError, match="16-bit"):
        make_rgbd(fake, rgb, depth.astype(np.float32), k, CANDIDATE_DEPTH_M)
    with pytest.raises(ValueError, match="depth cutoff"):
        make_rgbd(fake, rgb, depth, k, float("nan"))


def test_five_view_panel_preserves_photo_and_both_candidates(tmp_path):
    old = np.zeros((240, 320, 3), dtype=np.uint8)
    new = np.ones_like(old) * 128
    photo = Image.new("RGB", (640, 480), "white")
    path = tmp_path / "panel.png"
    image = panel(path, photo, old, new, "REAL / 4.5 / 3")
    assert image.size == (960, 268)
    with Image.open(path) as saved:
        assert saved.getpixel((160, 120)) == (255, 255, 255)
        assert saved.getpixel((480, 120)) == (0, 0, 0)
        assert saved.getpixel((800, 120)) == (128, 128, 128)
