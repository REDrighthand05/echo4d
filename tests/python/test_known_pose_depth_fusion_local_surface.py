"""Local RGB-inferred Z support should reject isolated discontinuity candidates."""
import numpy as np
from echo4d.known_pose_depth_fusion import _supported_local_surface, fuse_reprojected_depths
from echo4d.model import Camera


def test_local_surface_support_preserves_planar_patch_and_rejects_spike():
    z = np.full((30, 40), 2., dtype=np.float32)
    z[12, 15] = 3.0
    mask = _supported_local_surface(z)
    assert not mask[12, 15]
    assert mask[12, 16]
    assert mask[4, 4]


def test_opt_in_source_support_changes_transfer_not_existing_inference():
    k = Camera(40, 30, 50, 50, 19.5, 14.5)
    depths = [np.full((30, 40), 2., np.float32) for _ in range(4)]
    rgbs = [np.full((30, 40, 3), 100, np.uint8) for _ in range(4)]
    poses = []
    for i in range(4):
        p = np.eye(4, dtype=np.float64)
        p[0, 3] = .08 * i
        poses.append(p)
    depths[1][10:15, 15:22] = 0
    old, _ = fuse_reprojected_depths(depths, poses, rgbs, k)
    new, receipts = fuse_reprojected_depths(depths, poses, rgbs, k,
                                            local_surface_support=True)
    assert np.count_nonzero(new[1][10:15, 15:22]) > 0
    assert np.count_nonzero(new[1]) <= np.count_nonzero(old[1])
    assert receipts[1]['local_surface_support'] is True
    assert np.array_equal(new[0][4:8, 4:8], depths[0][4:8, 4:8])


def test_unsupported_foreground_still_occludes_supported_background():
    k = Camera(40, 30, 50, 50, 19.5, 14.5)
    depths = [np.zeros((30, 40), np.float32),
              np.full((30, 40), 2., np.float32),
              np.full((30, 40), 2., np.float32)]
    poses = [np.eye(4) for _ in range(3)]
    rgbs = [np.full((30, 40, 3), 100, np.uint8) for _ in range(3)]
    for j, x, near_u in [(1, -.08, 24), (2, .08, 16)]:
        poses[j][0, 3] = x
        depths[j][15, near_u] = 1
        rgbs[j][15, near_u] = 200
    fused, _ = fuse_reprojected_depths(depths, poses, rgbs, k, local_surface_support=True)
    assert fused[0][15, 20] == 0
