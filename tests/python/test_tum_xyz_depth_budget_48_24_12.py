"""Regression for nested real-depth budgets without requiring the TUM download."""
import numpy as np
import pytest

from tum_xyz_depth_budget_48_24_12 import diverse_pose_order


def entries_for_centers(xs):
    result = []
    for x in xs:
        pose = np.eye(4)
        pose[0, 3] = x
        result.append({"cameraToWorld": pose.reshape(-1, order="F").tolist()})
    return result


def test_pose_selection_is_deterministic_nested_and_spatially_spread():
    entries = entries_for_centers([0., .01, .02, .5, .51, 1., 1.01, 2.])
    order = diverse_pose_order(entries, tuple(range(len(entries))))
    assert order == diverse_pose_order(entries, tuple(range(len(entries))))
    assert sorted(order) == list(range(len(entries)))
    assert {0, 7}.issubset(set(order[:4]))
    assert set(order[:2]).issubset(order[:4])
    assert set(order[:4]).issubset(order[:8])


def test_selection_uses_pose_not_depth_or_rgb():
    entries = entries_for_centers([0., 1., 2., 3.])
    order = diverse_pose_order(entries, [0, 1, 2, 3])
    assert len(order) == 4
    assert set(order) == {0, 1, 2, 3}


def test_invalid_selection_rejected():
    entries = entries_for_centers([0., 1.])
    with pytest.raises(ValueError):
        diverse_pose_order(entries, [])
    with pytest.raises(ValueError):
        diverse_pose_order(entries, [0, 0])
    with pytest.raises(ValueError):
        diverse_pose_order(entries, [0, 1], orientation_m=-1)
    entries[1]["cameraToWorld"][12] = float("nan")
    with pytest.raises(ValueError):
        diverse_pose_order(entries, [0, 1])
