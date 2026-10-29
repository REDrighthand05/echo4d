"""Synthetic CPU regressions for robust measured-RGB-D pose graph solver."""
import numpy as np
import pytest

pytest.importorskip("open3d")
from tum_room_rgbd_robust_graph import robust_refine_poses
from echo4d.model import DataError


def pose(x, y=0.0):
    p = np.eye(4)
    p[:3, 3] = [x, y, 0.0]
    return p


def make_graph(poses):
    graph = [[] for _ in poses]
    for i in range(len(poses) - 1):
        j = i + 1
        T = np.linalg.inv(poses[j]) @ poses[i]
        graph[i].append((j, T, 1.0))
        graph[j].append((i, np.linalg.inv(T), 1.0))
    # Deliberately contradictory long-range loop, with low confidence.
    bad = np.eye(4)
    bad[0, 3] = 0.7
    graph[0].append((len(poses) - 1, bad, 50.0))
    graph[-1].append((0, np.linalg.inv(bad), 50.0))
    return graph


def test_robust_graph_reduces_initial_camera_drift_without_gt_in_solver():
    true = [pose(i * 0.12) for i in range(5)]
    initial = [p.copy() for p in true]
    for i in range(1, 5):
        initial[i][1, 3] += 0.04 * i
    result, report = robust_refine_poses(make_graph(true), initial)
    before = np.mean([np.linalg.norm(initial[i][:3, 3] - true[i][:3, 3])
                      for i in range(1, 5)])
    after = np.mean([np.linalg.norm(result[i][:3, 3] - true[i][:3, 3])
                     for i in range(1, 5)])
    assert np.array_equal(result[0], initial[0])
    assert after < before
    assert report["measured_rgbd_pnp_edges"] == 5
    assert report["no_dense_icp"]
    assert report["no_gt_pose_or_prior_mesh_in_solver"]


def test_reject_missing_graph_edges():
    with pytest.raises(DataError):
        robust_refine_poses([[], [], []], [pose(0), pose(1), pose(2)])


def test_reject_uninitialized_camera():
    with pytest.raises(DataError):
        robust_refine_poses([[], []], [pose(0), None])
