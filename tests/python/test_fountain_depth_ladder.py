"""Small regression guards for actual 33/16/8 Fountain sensor ablation."""
import numpy as np
import pytest
from fountain_depth_ladder import choose_fountain_indices


def make_poses():
    poses = np.repeat(np.eye(4)[None], 33, axis=0)
    poses[:, 0, 3] = np.linspace(0., 1.4, 33)
    poses[:, 2, 3] = np.linspace(0., .2, 33)
    for i in range(33):
        theta = .5 * np.sin(i / 7.)
        c, s = np.cos(theta), np.sin(theta)
        poses[i, :3, :3] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
    return poses


def test_nested_pose_only_selection():
    p = make_poses()
    eight, sixteen = choose_fountain_indices(p)
    assert len(eight) == 8 and len(sixteen) == 16
    assert eight == tuple(sorted(eight))
    assert sixteen == tuple(sorted(sixteen))
    assert set(eight).issubset(sixteen)
    assert (eight[0], eight[-1]) == (0, 32)
    assert choose_fountain_indices(p) == (eight, sixteen)


def test_reject_corrupt_reference_poses():
    p = make_poses()
    p[2, 0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        choose_fountain_indices(p)
    p = make_poses()
    p[4, 3, 3] = 2.
    with pytest.raises(ValueError, match="homogeneous"):
        choose_fountain_indices(p)
