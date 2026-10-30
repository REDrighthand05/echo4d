"""Lightweight camera geometry regression for real measured TUM 320 fusion."""
import numpy as np
import pytest
from echo4d.model import Camera
from tum_xyz_open3d_320 import camera_matrix, free_view, half_resolution


def test_pixel_centres_and_intrinsics_match_2x_decimation():
    camera = Camera(320, 240, 262.5, 262.5, 159.75, 119.75)
    reduced = camera.scaled(2)
    depth = np.arange(320 * 240, dtype=np.float32).reshape(240, 320)
    color = np.zeros((240, 320, 3), np.uint8)
    d, rgb = half_resolution(depth, color)
    assert d.shape == (120, 160) and rgb.shape == (120, 160, 3)
    assert d[23, 17] == depth[46, 34]
    assert reduced.fx == 131.25 and reduced.cx == 79.875
    ray_320 = np.linalg.inv(camera_matrix(camera)) @ np.array([34., 46., 1.])
    ray_160 = np.linalg.inv(camera_matrix(reduced)) @ np.array([17., 23., 1.])
    np.testing.assert_allclose(ray_320, ray_160, atol=1e-12)


def test_free_view_is_rigid_and_changes_observation():
    base = np.eye(4)
    left = free_view(base, -1.)
    right = free_view(base, 1.)
    np.testing.assert_allclose(left[:3, :3].T @ left[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(right[:3, :3].T @ right[:3, :3], np.eye(3), atol=1e-12)
    assert left[0, 3] == pytest.approx(-.22)
    assert right[0, 3] == pytest.approx(.22)
    assert not np.allclose(left[:3, :3], right[:3, :3])


def test_invalid_depth_pair_and_free_view_rejected():
    with pytest.raises(ValueError):
        half_resolution(np.zeros((4, 4)), np.zeros((4, 3, 3)))
    with pytest.raises(ValueError):
        free_view(np.eye(4), 0.)
