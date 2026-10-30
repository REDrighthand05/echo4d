"""Small geometric regressions for real Fountain RGB-only depth filtering."""
import numpy as np
import pytest

from echo4d.model import Camera, Frame
from fountain_rgb_visibility_guard import multiview_guard


def scene():
    h, w = 60, 80
    camera = Camera(w, h, 80., 80., (w-1)/2., (h-1)/2.)
    rgb = np.full((h, w, 3), 105, np.uint8)
    depth = np.full((h, w), 2., np.float32)
    pose0 = np.eye(4)
    pose1 = np.eye(4)
    pose1[0, 3] = .04
    frames = [
        Frame(0., pose0, depth.copy(), rgb.copy()).validate(camera),
        Frame(1., pose1, depth.copy(), rgb.copy()).validate(camera),
    ]
    return camera, rgb, frames


def test_sensor_supported_wall_is_preserved_and_snapped():
    camera, rgb, frames = scene()
    pred = np.full((60, 80), 2.04, np.float32)
    filtered, details = multiview_guard(pred, rgb, np.eye(4), frames, camera)
    assert details["multi_view_confirmed_pixels"] > 3000
    assert details["sensor_snapped_pixels"] > 1500
    assert details["multi_view_contradicted_pixels"] == 0
    assert np.median(filtered[filtered > 0]) < 2.02


def test_false_foreground_contradicted_by_two_measured_views():
    camera, _, frames = scene()
    pred = np.full((60, 80), 1., np.float32)
    wrong_color = np.full((60, 80, 3), 245, np.uint8)
    filtered, details = multiview_guard(pred, wrong_color, np.eye(4),
                                        frames, camera)
    assert details["multi_view_contradicted_pixels"] > 3000
    assert np.count_nonzero(filtered) < 1000


def test_occluded_or_unobserved_surface_not_called_contradiction():
    camera, _, frames = scene()
    pred = np.full((60, 80), 3., np.float32)
    other_color = np.full((60, 80, 3), 245, np.uint8)
    filtered, details = multiview_guard(pred, other_color, np.eye(4),
                                        frames, camera)
    assert details["multi_view_contradicted_pixels"] == 0
    assert np.count_nonzero(filtered) == pred.size


def test_invalid_target_rgb_rejected():
    camera, _, frames = scene()
    with pytest.raises(ValueError):
        multiview_guard(np.ones((60,80),np.float32),
                        np.zeros((30,40,3),np.uint8),np.eye(4),frames,camera)
