"""CPU regression for actual RGB-D pose removal, no sensor archive needed."""
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("open3d")
from tum_room_rgbd_pose_ladder import (
    anchor_indices, build_pose_graph, estimate_poses, camera_matrix)


def test_nested_pose_anchor_count_and_endpoints():
    assert anchor_indices(1, 96) == [0]
    assert len(anchor_indices(13, 96)) == 13
    assert anchor_indices(13, 96)[-1] == 95


def test_unknown_ground_truth_cannot_enter_pose_propagation():
    poses = [np.eye(4) for _ in range(4)]
    poses[1][0, 3] = 100
    poses[2][0, 3] = -100
    poses[3][0, 3] = 0.3
    graph = [[] for _ in range(4)]
    for i in range(3):
        T = np.eye(4)
        T[0, 3] = -0.1
        graph[i].append((i + 1, T, 1.0))
        graph[i + 1].append((i, np.linalg.inv(T), 1.0))
    estimated, provenance = estimate_poses(graph, poses, [0])
    assert all(p is not None for p in estimated)
    assert np.allclose([p[0, 3] for p in estimated], [0, .1, .2, .3])
    assert all(p["kind"] == "rgbd_sift_pnp" for p in provenance[1:])


def test_synthetic_metric_rgbd_pnp_camera_motion():
    import cv2
    rng = np.random.default_rng(21)
    image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    other = np.zeros_like(image)
    other[:, :-16] = image[:, 16:]
    z = np.full((240, 320), 2.0, dtype=np.float32)
    K = camera_matrix({"fx": 160, "fy": 160, "cx": 159.5, "cy": 119.5})
    graph, report = build_pose_graph([image, other], [z, z], K, max_gap=1)
    assert report["tested_pairs"] == 1
    assert report["successful_pairs"] == 1
    known = [np.eye(4), np.eye(4)]
    estimated, provenance = estimate_poses(graph, known, [0])
    assert provenance[1]["kind"] == "rgbd_sift_pnp"
    assert abs(estimated[1][0, 3] - 0.2) < 0.08
