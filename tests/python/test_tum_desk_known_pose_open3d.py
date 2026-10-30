"""Focused second-scene geometry checks; synthetic only, not visual acceptance."""
import numpy as np
import pytest

from echo4d.model import Camera, DataError
from tum_desk_known_pose_open3d import (
    MAX_IMPORTED, RECORDED, FREE, FR1_INTRINSICS, split_indices,
)
from tum_xyz_open3d_320 import camera_matrix, half_resolution, free_view


def test_desk_training_and_heldout_are_disjoint_and_recorded_cameras_are_measured():
    train, held = split_indices(MAX_IMPORTED)
    assert len(train) + len(held) == MAX_IMPORTED
    assert not set(train) & set(held)
    assert set(RECORDED).issubset(train)
    assert all(index < MAX_IMPORTED for _, index in FREE)


def test_import_may_skip_unsynchronized_desk_frames_without_reusing_holdouts():
    train, held = split_indices(54)
    assert len(train) + len(held) == 54
    assert set(RECORDED).issubset(train)
    with pytest.raises(DataError):
        split_indices(53)
    with pytest.raises(DataError):
        split_indices(MAX_IMPORTED + 1)


def test_freiburg1_intrinsics_are_scaled_with_actual_sampled_pixel_centres():
    cam = Camera(640, 480, *FR1_INTRINSICS)
    small = cam.scaled(2)
    assert (small.width, small.height) == (320, 240)
    assert small.fx == pytest.approx(258.65)
    assert small.fy == pytest.approx(258.25)
    assert small.cx == pytest.approx(159.3)
    assert small.cy == pytest.approx(127.65)
    sample = np.array([128., 160., 1.])
    np.testing.assert_allclose(
        np.linalg.inv(camera_matrix(cam)) @ sample,
        np.linalg.inv(camera_matrix(small)) @ np.array([64., 80., 1.]),
        atol=1e-12,
    )
    d = np.arange(240 * 320, dtype=np.float32).reshape(240, 320)
    color = np.zeros((240, 320, 3), np.uint8)
    depth_160, rgb_160 = half_resolution(d, color)
    assert depth_160.shape == (120, 160)
    assert rgb_160.shape == (120, 160, 3)


def test_free_cameras_are_not_source_cameras():
    for side, index in FREE:
        assert index >= 0
        moved = free_view(np.eye(4), side)
        assert not np.allclose(moved, np.eye(4))
        np.testing.assert_allclose(moved[:3, :3].T @ moved[:3, :3],
                                   np.eye(3), atol=1e-12)
