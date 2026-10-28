"""Cross-view RGB-supported depth transfer tests; no sensor depth or GPU."""
import numpy as np
import pytest
from echo4d.known_pose_depth_fusion import fuse_reprojected_depths
from echo4d.model import Camera


def scene(n=4):
    k = Camera(width=40, height=30, fx=50, fy=50, cx=19.5, cy=14.5)
    poses, depths, rgbs = [], [], []
    for i in range(n):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = i * .08
        poses.append(pose)
        depths.append(np.full((30, 40), 2, np.float32))
        rgbs.append(np.full((30, 40, 3), 100, np.uint8))
    return k, depths, poses, rgbs


def test_two_distinct_views_transfer_observed_surface_to_hole():
    k, d, p, rgb = scene()
    d[1][10:15, 15:22] = 0
    fused, receipts = fuse_reprojected_depths(d, p, rgb, k)
    assert np.count_nonzero(fused[1][10:15, 15:22]) >= 20
    assert receipts[1]['added_pixels'] >= 20
    assert all(np.all(np.isfinite(x)) for x in fused)


def test_one_view_cannot_fill_unknown():
    k, d, p, rgb = scene(3)
    d[1][:] = 0
    d[2][:] = 0
    fused, receipts = fuse_reprojected_depths(d, p, rgb, k)
    assert not fused[1].any()
    assert receipts[1]['added_pixels'] == 0


def test_inconsistent_depth_or_rgb_keeps_unknown():
    k, d, p, rgb = scene()
    d[1][:] = 0
    d[2][:] = 3
    d[3][:] = 4
    fused, _ = fuse_reprojected_depths(d, p, rgb, k)
    assert not fused[1].any()
    k, d, p, rgb = scene()
    d[1][:] = 0
    rgb[1][:] = 250
    fused, _ = fuse_reprojected_depths(d, p, rgb, k)
    assert not fused[1].any()


def test_zero_baseline_does_not_claim_multiple_observations():
    k, d, p, rgb = scene()
    p = [np.eye(4) for _ in p]
    fused, receipts = fuse_reprojected_depths(d, p, rgb, k)
    assert np.array_equal(fused[0], d[0])
    assert receipts[0]['added_pixels'] == 0


def test_invalid_estimated_depth_rejected():
    k, d, p, rgb = scene()
    d[0][0, 0] = np.nan
    with pytest.raises(ValueError):
        fuse_reprojected_depths(d, p, rgb, k)
