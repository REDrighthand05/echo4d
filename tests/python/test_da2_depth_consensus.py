"""Synthetic geometry regression; real DA2 CPU inference runs in GitHub Actions."""
import numpy as np
import pytest

from da2_depth_consensus import reproject_z_color, reconcile_depths, neighbors_for


def camera(x):
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose


def test_identity_camera_z_and_rgb():
    z = np.zeros((24, 36), np.float32)
    z[4:20, 4:32] = 2.
    rgb = np.full((24, 36, 3), 73, np.uint8)
    K = np.array([[40., 0, 18.], [0, 40., 12.], [0, 0, 1.]])
    new_z, new_rgb = reproject_z_color(z, rgb, camera(0), camera(0), K)
    np.testing.assert_allclose(new_z, z)
    np.testing.assert_array_equal(new_rgb[z > 0], rgb[z > 0])


def test_two_known_pose_rgb_predictions_fill_missing_center():
    shape = (40, 60)
    z = [np.full(shape, 2., np.float32) for _ in range(3)]
    z[1][12:28, 18:42] = 0
    rgb = [np.full((*shape, 3), 100, np.uint8) for _ in range(3)]
    poses = [camera(-.05), camera(0), camera(.05)]
    K = np.array([[80., 0, 30.], [0, 80., 20.], [0, 0, 1.]])
    revised, receipts = reconcile_depths(z, rgb, poses, K)
    assert receipts[1]["filled_rgb_supported_pixels"] > 100
    assert np.count_nonzero(revised[1]) > np.count_nonzero(z[1])
    assert abs(float(revised[1][20, 30]) - 2.) < .01
    np.testing.assert_array_equal(z[0], revised[0])


def test_inconsistent_predicted_depths_do_not_hallucinate_fill():
    shape = (40, 60)
    z = [np.full(shape, 2., np.float32),
         np.zeros(shape, np.float32),
         np.full(shape, 3., np.float32)]
    rgb = [np.full((*shape, 3), 100, np.uint8) for _ in range(3)]
    poses = [camera(-.05), camera(0), camera(.05)]
    K = np.array([[80., 0, 30.], [0, 80., 20.], [0, 0, 1.]])
    revised, receipts = reconcile_depths(z, rgb, poses, K)
    assert receipts[1]["filled_rgb_supported_pixels"] == 0
    assert not np.any(revised[1])


def test_color_mismatch_does_not_fill():
    shape = (40, 60)
    z = [np.full(shape, 2., np.float32),
         np.zeros(shape, np.float32),
         np.full(shape, 2., np.float32)]
    rgb = [np.full((*shape, 3), 255, np.uint8),
           np.zeros((*shape, 3), np.uint8),
           np.full((*shape, 3), 255, np.uint8)]
    poses = [camera(-.05), camera(0), camera(.05)]
    K = np.array([[80., 0, 30.], [0, 80., 20.], [0, 0, 1.]])
    revised, receipts = reconcile_depths(z, rgb, poses, K)
    assert receipts[1]["filled_rgb_supported_pixels"] == 0
    assert not np.any(revised[1])


def test_duplicate_camera_positions_not_independent():
    poses = [camera(0), camera(0), camera(0)]
    assert neighbors_for(poses, 1) == []


def test_reject_invalid_geometry():
    K = np.eye(3)
    with pytest.raises(ValueError):
        reproject_z_color(np.full((3, 3), -1., np.float32),
                          np.zeros((3, 3, 3), np.uint8), camera(0), camera(0), K)
