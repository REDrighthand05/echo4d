"""Bounded ICP geometry guards; no real-scene result is inferred from these tests."""
import numpy as np

from room_pose_icp import correction_limits


def test_small_camera_center_and_rotation_correction_is_allowed():
    initial = np.eye(4)
    initial[:3, 3] = [2.0, -3.0, 1.0]
    angle = np.deg2rad(0.4)
    candidate = initial.copy()
    candidate[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]]
    candidate[:3, 3] += [0.01, -0.005, 0]
    allowed, displacement, degrees = correction_limits(initial, candidate)
    assert allowed
    assert np.isclose(displacement, np.sqrt(0.01**2 + 0.005**2))
    assert np.isclose(degrees, 0.4, atol=1e-5)


def test_rejects_large_drift_and_invalid_pose():
    initial = np.eye(4)
    shifted = initial.copy()
    shifted[0, 3] = 0.04
    assert not correction_limits(initial, shifted)[0]
    rotated = initial.copy()
    angle = np.deg2rad(2)
    rotated[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]]
    assert not correction_limits(initial, rotated)[0]
    broken = initial.copy()
    broken[0, 0] = np.nan
    assert not correction_limits(initial, broken)[0]
