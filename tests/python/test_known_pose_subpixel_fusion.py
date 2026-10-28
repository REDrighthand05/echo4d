"""Conservative subpixel TSDF sampling: no sensor depth and no fake holes."""
import numpy as np
import pytest
from echo4d.model import Camera, DataError
from echo4d.known_pose_subpixel_fusion import (
    upsampled_camera, upsample_observed_depth, upsample_observed_rgb)


def test_original_rays_and_depths_remain_exact():
    k = Camera(4, 3, 60., 62., 1.5, 1.)
    z = np.array([[2., 2.02, 2.04, 2.],
                  [2., 2.02, 2.04, 2.],
                  [2., 2.02, 2.04, 2.]], np.float32)
    out, report = upsample_observed_depth(z)
    new_k = upsampled_camera(k)
    assert out.shape == (5, 7)
    np.testing.assert_array_equal(out[::2, ::2], z)
    assert (new_k.width, new_k.height) == (7, 5)
    for u, v in [(0, 0), (1, 1), (3, 2)]:
        x0, y0 = (u - k.cx) / k.fx, (v - k.cy) / k.fy
        x1, y1 = (2*u - new_k.cx) / new_k.fx, (2*v - new_k.cy) / new_k.fy
        assert x0 == pytest.approx(x1)
        assert y0 == pytest.approx(y1)
    assert report['interpolated_pixels_not_independent_measurements'] > 0


def test_unknown_and_depth_edges_are_not_bridged():
    z = np.array([[2., 0., 3.], [2., 0., 3.],
                  [2., 0., 3.]], np.float32)
    out, report = upsample_observed_depth(z)
    assert not np.any(out[:, 1:4])
    assert np.all(out[::2, 0] == 2)
    assert np.all(out[::2, 4] == 3)
    assert report['unknown_remains_zero']
    step = np.array([[1., 3.], [1., 3.]], np.float32)
    out, _ = upsample_observed_depth(step)
    assert np.all(out[:, 1] == 0)
    assert out[1, 0] == 1 and out[1, 2] == 3


def test_rgb_original_samples_are_not_changed():
    rgb = np.arange(3*4*3, dtype=np.uint8).reshape(3, 4, 3)
    out = upsample_observed_rgb(rgb)
    assert out.shape == (5, 7, 3)
    np.testing.assert_array_equal(out[::2, ::2], rgb)


def test_reject_invalid_inferred_depth_and_rgb():
    for z in [np.ones((2, 2), np.float64),
              np.array([[1., np.nan], [1., 1.]], np.float32),
              np.array([[1., -1.], [1., 1.]], np.float32)]:
        with pytest.raises(DataError):
            upsample_observed_depth(z)
    with pytest.raises(DataError):
        upsample_observed_rgb(np.ones((3, 4, 3), np.float32))
