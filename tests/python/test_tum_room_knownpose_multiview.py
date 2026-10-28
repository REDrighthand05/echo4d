"""CPU synthetic geometry checks only; real TUM room evidence comes from Actions."""
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV is installed by real room workflow")
from echo4d.model import DataError
from tum_room_knownpose_multiview import choose_neighbors, sweep_depth


def test_multiview_metric_plane_from_known_pose_without_sensor_depth():
    h, w = 160, 240
    rng = np.random.default_rng(3)
    reference = cv2.GaussianBlur(
        rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (3, 3), .5)
    K = np.array([[150., 0, w/2], [0, 150., h/2], [0, 0, 1.]])
    pose = np.eye(4)
    sources, source_poses = [], []
    for dx in (.08, -.10, .16):
        other = np.eye(4)
        other[0, 3] = dx
        H = K @ np.array([[1., 0., -dx/1.3], [0., 1., 0.], [0., 0., 1.]]) @ np.linalg.inv(K)
        sources.append(cv2.warpPerspective(reference, H, (w, h)))
        source_poses.append(other)
    depth, receipt = sweep_depth(reference, sources, K, pose, source_poses)
    valid = depth > 0
    assert receipt["source_cameras"] == 3
    assert receipt["valid_pixels"] > 20000
    assert abs(float(np.median(depth[valid])) - 1.3) < .15
    assert float(np.mean(np.abs(depth[valid] - 1.3) < .2)) > .9


def test_flat_rgb_remains_unknown_not_invented_plane():
    img = np.full((120, 200, 3), 120, dtype=np.uint8)
    K = np.array([[120., 0, 100.], [0, 120., 60.], [0, 0, 1.]])
    pose = np.eye(4)
    a, b = np.eye(4), np.eye(4)
    a[0, 3], b[0, 3] = .08, -.10
    depth, receipt = sweep_depth(img, [img, img], K, pose, [a, b])
    assert not np.any(depth)
    assert receipt["valid_pixels"] == 0


def test_requires_two_independent_source_cameras():
    img = np.zeros((100, 160, 3), dtype=np.uint8)
    K = np.eye(3)
    with pytest.raises(DataError, match=">=2 sources"):
        sweep_depth(img, [img], K, np.eye(4), [np.eye(4)])


def test_neighbor_selection_excludes_same_pose_and_forward_only_motion():
    poses = [np.eye(4) for _ in range(5)]
    poses[1][0, 3] = .12
    poses[2][0, 3] = -.12
    poses[3][2, 3] = .12
    poses[4][0, 3] = 1.0
    assert choose_neighbors(poses, 0) == [1, 2]


def test_duplicate_camera_cannot_supply_two_photometric_votes():
    h, w = 120, 200
    rng = np.random.default_rng(3)
    reference = cv2.GaussianBlur(rng.integers(0,256,(h,w,3),dtype=np.uint8),(3,3),.5)
    K = np.array([[120.,0,w/2],[0,120.,h/2],[0,0,1.]])
    pose = np.eye(4)
    pose[0,3] = .1
    H = K @ np.array([[1.,0,-.1/1.3],[0,1.,0],[0,0,1.]]) @ np.linalg.inv(K)
    source = cv2.warpPerspective(reference,H,(w,h))
    with pytest.raises(DataError, match='distinct camera positions'):
        sweep_depth(reference,[source,source.copy()],K,np.eye(4),[pose,pose.copy()])
    # Rotation at the same optical center does not create another witness.
    rotated = pose.copy()
    rotated[:3,:3] = np.diag([-1.,-1.,1.])
    with pytest.raises(DataError, match='distinct camera positions'):
        sweep_depth(reference,[source,source.copy()],K,np.eye(4),[pose,rotated])
    with pytest.raises(DataError, match='distinct camera positions'):
        sweep_depth(reference,[reference,source],K,np.eye(4),[np.eye(4),pose])


def test_duplicate_candidates_do_not_crowd_out_distinct_neighbors():
    poses = [np.eye(4) for _ in range(5)]
    for p in poses[1:4]:
        p[0,3] = .14
    poses[4][0,3] = -.15
    assert choose_neighbors(poses,0,limit=2) == [1,4]
