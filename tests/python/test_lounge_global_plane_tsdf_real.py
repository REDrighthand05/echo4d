"""Measured plane regularization must respect missing and occluding depth."""
import numpy as np
import pytest

from echo4d.model import Camera
from lounge_global_plane_tsdf_real import regularize_depth


def camera():
    return Camera(40, 30, 40., 40., 19.5, 14.5)


def test_nearby_measured_wall_is_regularized():
    rng = np.random.default_rng(5)
    depth = (2.0 + rng.uniform(-.012, .012, (30, 40))).astype(np.float32)
    corrected, stats = regularize_depth(
        depth, camera(), np.eye(4), [np.array([0., 0., 1., -2.])])
    assert stats["changed_pixels"] > 700
    assert np.std(corrected[3:-3, 3:-3]) < .001
    assert np.count_nonzero(depth != corrected) > 700


def test_foreground_invalid_and_silhouette_are_preserved():
    depth = np.full((30, 40), 2.012, np.float32)
    depth[8:22, 12:28] = 1.0
    depth[3:7, 3:7] = 0.0
    original = depth.copy()
    corrected, _ = regularize_depth(
        depth, camera(), np.eye(4), [np.array([0., 0., 1., -2.])])
    np.testing.assert_array_equal(corrected[8:22, 12:28], original[8:22, 12:28])
    np.testing.assert_array_equal(corrected[3:7, 3:7], original[3:7, 3:7])
    assert corrected[9, 9] == pytest.approx(2.0, abs=1e-5)
    assert corrected[7, 12] == original[7, 12]


def test_no_plane_keeps_measured_depth_unchanged():
    depth = np.full((30, 40), 1.7, np.float32)
    corrected, stats = regularize_depth(depth, camera(), np.eye(4), [])
    np.testing.assert_array_equal(corrected, depth)
    assert stats["changed_pixels"] == 0


def test_bad_intrinsics_shape_rejected():
    with pytest.raises(ValueError):
        regularize_depth(np.ones((4, 4)), camera(), np.eye(4), [])
