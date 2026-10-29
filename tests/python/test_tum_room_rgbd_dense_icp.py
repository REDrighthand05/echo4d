"""CPU regressions for measured-depth ICP edge acceptance and camera graph."""
import numpy as np
import pytest

o3d = pytest.importorskip("open3d")
pytest.importorskip("cv2")
from tum_room_rgbd_dense_icp import accept_icp, cloud_from_measured_depth, refine_poses


class Result:
    def __init__(self, transform, fitness, rmse):
        self.transformation = transform
        self.fitness = fitness
        self.inlier_rmse = rmse


def test_dense_cloud_uses_only_observed_depth_and_metric_intrinsics():
    depth = np.zeros((30, 30), np.float32)
    depth[6:24, 6:24] = 2.0
    cloud = cloud_from_measured_depth(
        depth, {"fx": 100, "fy": 100, "cx": 15, "cy": 15},
        stride=1, voxel=0.01)
    pts = np.asarray(cloud.points)
    assert len(pts) > 30
    assert np.allclose(pts[:, 2], 2.0)
    assert np.isfinite(pts).all()


def test_icp_rejects_bad_overlap_and_large_pose_jump():
    eye = np.eye(4)
    assert accept_icp(eye, Result(eye, .55, .015))
    assert not accept_icp(eye, Result(eye, .05, .015))
    assert not accept_icp(eye, Result(eye, .55, .10))
    jump = eye.copy()
    jump[0, 3] = .4
    assert not accept_icp(eye, Result(jump, .55, .015))


def test_graph_with_two_observed_clouds_keeps_anchor_and_finite_poses():
    intr = {"fx": 100, "fy": 100, "cx": 20, "cy": 20}
    depth = np.full((42, 42), 1.8, np.float32)
    # Vary depth to make an actual 3D surface rather than a planar toy.
    y, x = np.indices(depth.shape)
    depth += .08 * np.sin(x / 4) * np.cos(y / 6)
    a = cloud_from_measured_depth(depth, intr, stride=1, voxel=.025)
    initial = [np.eye(4), np.eye(4)]
    T = np.eye(4)
    graph = [[(1, T, 1.0)], [(0, T, 1.0)]]
    poses, report = refine_poses(graph, initial, [a, a])
    assert len(poses) == 2
    assert np.array_equal(poses[0], np.eye(4))
    assert np.isfinite(poses[1]).all()
    assert report["rgbd_edges_attempted"] == 1
    assert report["dense_icp_edges_accepted"] == 1
