import numpy as np
import pytest

from redwood_measured_pose_refusion import (
    consensus_delta, corrected_world_to_camera, loop_targets,
    rotation_degrees, validate_delta,
)


def delta(tx=0., rz_deg=0.):
    angle = np.deg2rad(rz_deg)
    d = np.eye(4)
    d[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    d[0, 3] = tx
    return d


def test_small_consensus_and_pose_convention():
    a = delta(.014, .5)
    b = delta(.016, .7)
    c = consensus_delta(a, b)
    assert c is not None
    assert validate_delta(c)
    original = np.eye(4)
    corrected = corrected_world_to_camera(original, c)
    point_in_old_world = np.array([.1, .2, 1., 1.])
    # A camera coordinate point mapped through new C2W equals delta @ old C2W.
    assert np.allclose(np.linalg.inv(corrected) @ point_in_old_world,
                       c @ point_in_old_world)
    assert np.isclose(rotation_degrees(c[:3, :3]), .6, atol=1e-3)


def test_large_or_disagreeing_corrections_rejected():
    assert not validate_delta(delta(.04))
    assert not validate_delta(delta(.01, 2.))
    assert consensus_delta(delta(.02), delta(-.02)) is None
    assert consensus_delta(delta(.005, 1.), delta(.005, -1.)) is None
    with pytest.raises(ValueError):
        corrected_world_to_camera(np.eye(4), delta(.04))


def test_loop_targets_need_independent_real_observations():
    selected = list(range(0, 900, 50))
    poses = []
    for _ in range(max(selected) + 1):
        p = np.eye(4)
        poses.append(p)
    result = loop_targets(selected, poses)
    assert result
    for j, targets in result.items():
        assert len(targets) == 2
        assert all(abs(t - selected[j]) >= 50 for t in targets)
        assert abs(targets[0] - targets[1]) >= 50
    far = []
    for i in range(max(selected) + 1):
        p = np.eye(4)
        p[0, 3] = i * 2
        far.append(p)
    assert not loop_targets(selected, far)


def test_invalid_delta_not_accepted():
    d = np.eye(4)
    d[3, 0] = 1
    assert not validate_delta(d)
    d = np.eye(4)
    d[0, 0] = 2
    assert not validate_delta(d)
