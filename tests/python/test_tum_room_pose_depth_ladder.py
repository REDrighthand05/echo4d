"""Small deterministic checks for camera-coverage depth-frame reduction."""
import numpy as np
import pytest

from tum_room_pose_depth_ladder import select_pose_spread, select_uniform


def make_poses(n=145):
    poses = {}
    for i in range(n):
        t = 2*np.pi*i/(n-1)
        p = np.eye(4)
        p[:3, 3] = [np.cos(t), np.sin(t), .15*np.sin(2*t)]
        c, s = np.cos(t), np.sin(t)
        p[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        poses[i*2] = p
    return poses


def test_pose_selection_reduces_sensor_frames_and_spans_loop():
    poses = make_poses()
    ids = tuple(poses)
    chosen = select_pose_spread(ids, poses, 72)
    assert len(chosen) == len(set(chosen)) == 72
    assert set(chosen).issubset(ids)
    assert chosen[0] == ids[0] and chosen[-1] == ids[-1]
    assert all(any(abs(j - i) <= 8 for j in chosen) for i in ids)


def test_uniform_is_distinct_and_endpoints():
    ids = tuple(range(0, 290, 2))
    chosen = select_uniform(ids, 72)
    assert len(set(chosen)) == 72
    assert chosen[0] == 0 and chosen[-1] == 288


def test_invalid_pose_and_count_rejected():
    poses = make_poses(20)
    ids = tuple(poses)
    with pytest.raises(ValueError, match="Invalid"):
        select_pose_spread(ids, poses, 9)
    bad = {k: v.copy() for k, v in poses.items()}
    bad[ids[0]][0, 3] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        select_pose_spread(ids, bad, 10)
    with pytest.raises(ValueError, match="sorted"):
        select_pose_spread(ids[::-1], poses, 10)
