"""CPU-only frustum neighbor tests; no model weights or sensor depth."""
import numpy as np
import pytest
from tum_room_pose_overlap_neighbors import projected_overlap, choose_overlap_neighbors


def pose(x=0., y=0., z=0.):
    p = np.eye(4)
    p[:3, 3] = [x, y, z]
    return p


def test_overlap_is_not_a_depth_measurement():
    assert projected_overlap(pose(), pose()) > .99
    assert projected_overlap(pose(), pose(20.)) == 0.


def test_only_distinct_translated_rgb_witnesses():
    cameras = [pose(), pose(.09), pose(-.12), pose(.09), pose(8.)]
    selected = choose_overlap_neighbors(cameras, 0)
    assert len(selected) == 2
    assert set(selected) in ({1, 2}, {2, 3})
    assert 4 not in selected


def test_unknown_when_fewer_than_two_real_sources():
    assert len(choose_overlap_neighbors([pose(), pose(.11), pose(6.)], 0)) == 1


def test_invalid_poses_rejected():
    with pytest.raises(ValueError):
        choose_overlap_neighbors([np.full((4, 4), np.nan)], 0)
    with pytest.raises(ValueError):
        choose_overlap_neighbors([pose()], 2)
