"""Synthetic geometry regressions; real TUM RGB replay runs in Actions."""
import numpy as np
import pytest
from da2_global_scale import synchronize_scales


def make_scene(n=3):
    h, w = 96, 128
    K = np.array([[110., 0., w/2], [0., 110., h/2], [0., 0., 1.]])
    poses, depths, rgbs = [], [], []
    yy, xx = np.mgrid[:h, :w]
    for i in range(n):
        pose = np.eye(4)
        pose[0, 3] = .12*i
        wx = (xx-K[0, 2])*2/K[0, 0]+pose[0, 3]
        wy = (yy-K[1, 2])*2/K[1, 1]
        rgb = np.stack((127+70*np.sin(wx*9), 127+65*np.cos(wy*7),
                        127+65*np.sin(wx*5+wy*4)), axis=-1)
        rgbs.append(np.clip(rgb, 0, 255).astype(np.uint8))
        z = np.full((h, w), 2., np.float32)
        if i == 1:
            z *= 1.10
        depths.append(z)
        poses.append(pose)
    return depths, rgbs, poses, K


def test_scale_drift_reduced_on_unfitted_rgb_correspondences():
    d, c, p, k = make_scene()
    out, r = synchronize_scales(d, c, p, k, min_pair_points=60, prior_strength=1.)
    assert r["pairs"] >= 1
    assert r["applied"]
    assert r["heldout_pair_rmse_after_m"] < r["heldout_pair_rmse_before_m"]
    assert r["scales"][1] < 1.
    assert all(np.isfinite(x).all() for x in out)


def test_no_neighbor_keeps_depth_unchanged():
    d, c, p, k = make_scene(2)
    p[1][0, 3] = 5.
    out, r = synchronize_scales(d, c, p, k)
    assert r["status"] == "no_supported_RGB_pairs"
    assert all(np.array_equal(a, b) for a, b in zip(d, out))


def test_unknown_depth_not_filled():
    d, c, p, k = make_scene()
    d[0][10:20, 15:40] = 0
    out, _ = synchronize_scales(d, c, p, k, min_pair_points=60)
    assert np.all(out[0][10:20, 15:40] == 0)


def test_negative_depth_rejected():
    d, c, p, k = make_scene()
    d[0][0, 0] = -1
    with pytest.raises(ValueError, match="Nonfinite or negative"):
        synchronize_scales(d, c, p, k)
