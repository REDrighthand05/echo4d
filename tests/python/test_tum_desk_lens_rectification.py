"""Synthetic remap tests only; real TUM visual verdict requires workflow output."""
import cv2
import numpy as np
import pytest

from echo4d.model import Camera, Frame
from tum_desk_lens_rectification import FR1_DISTORTION, lens_map, rectify_pair, rectify_frame


def test_zero_distortion_is_identity_for_registered_rgb_and_depth():
    cam = Camera(320, 240, 258.65, 258.25, 159.3, 127.65)
    yy, xx = np.indices((240, 320))
    depth = (1 + xx / 1000 + yy / 10000).astype(np.float32)
    depth[20:50, 30:60] = 0
    color = np.stack((xx % 256, yy % 256, (xx + yy) % 256), -1).astype(np.uint8)
    d, c, _ = rectify_pair(depth, color, lens_map(cam, (0, 0, 0, 0, 0)))
    np.testing.assert_array_equal(d, depth)
    np.testing.assert_array_equal(c, color)


def test_remap_matches_independent_brown_conrady_projection():
    cam = Camera(320, 240, 258.65, 258.25, 159.3, 127.65)
    mx, my = lens_map(cam)
    k = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], np.float64)
    for x, y in ((0, 0), (319, 239), (50, 100), (159, 127)):
        ray = np.array([[(x-cam.cx)/cam.fx, (y-cam.cy)/cam.fy, 1]], np.float64)
        projected, _ = cv2.projectPoints(ray, np.zeros(3), np.zeros(3),
                                         k, np.array(FR1_DISTORTION))
        np.testing.assert_allclose([mx[y, x], my[y, x]], projected[0, 0], atol=2e-5)


def test_no_interpolated_fake_depth_or_mutation_and_exclusion_preserved():
    cam = Camera(320, 240, 258.65, 258.25, 159.3, 127.65)
    d = np.zeros((240, 320), np.float32)
    d[50:150, 70:200] = 1.5
    d[100:150, 100:200] = 2.5
    c = np.zeros((240, 320, 3), np.uint8)
    c[:, :, 0] = 200
    ex = np.zeros_like(d, bool)
    ex[80:100, 80:100] = True
    frame = Frame(1.0, np.eye(4), d, c, ex)
    corrected = rectify_frame(frame, lens_map(cam))
    assert corrected is not frame
    assert corrected.depth.shape == d.shape
    assert corrected.color.shape == c.shape
    assert corrected.excluded.dtype == bool
    assert set(np.unique(corrected.depth)).issubset({0., 1.5, 2.5})
    np.testing.assert_array_equal(frame.depth, d)
    np.testing.assert_array_equal(frame.color, c)
    corrected.validate(cam)


def test_bad_maps_rejected():
    with pytest.raises(ValueError):
        lens_map(Camera(320, 240, 258.65, 258.25, 159.3, 127.65), (0, 1))
