"""Regression: short supported topology bridges cannot grow dangling sheets."""
import numpy as np
import pytest

from echo4d.known_pose_edge_bridges import select_short_bridges, filter_bridge_faces
from echo4d.model import Camera


def test_single_weak_face_connects_two_strong_islands():
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [6, 7, 8]])
    baseline = np.array([1, 0, 0, 0, 0], dtype=bool)
    strong = np.array([1, 1, 0, 1, 0], dtype=bool)
    weak = np.array([1, 1, 1, 1, 1], dtype=bool)
    selected, stats = select_short_bridges(faces, baseline, strong, weak)
    assert selected.tolist() == [True, True, True, True, False]
    assert stats["bridge_faces_retained"] == 1


def test_two_face_bridge_only_when_allowed():
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    baseline = np.array([1, 0, 0, 0, 0], dtype=bool)
    strong = np.array([1, 1, 0, 0, 1], dtype=bool)
    weak = np.ones(5, dtype=bool)
    selected, _ = select_short_bridges(faces, baseline, strong, weak)
    assert selected.tolist() == [True] * 5
    selected, _ = select_short_bridges(faces, baseline, strong, weak, max_weak_chain=1)
    assert selected.tolist() == [True, True, False, False, False]


def test_no_bridge_to_single_component_or_detached_island():
    faces = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5], [5, 6, 7]])
    baseline = np.array([1, 0, 0, 0], dtype=bool)
    strong = np.array([1, 0, 0, 0], dtype=bool)
    weak = np.ones(4, dtype=bool)
    selected, _ = select_short_bridges(faces, baseline, strong, weak)
    assert selected.tolist() == [True, False, False, False]


def test_baseline_cannot_be_weak_or_removed():
    faces = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="Invalid face support"):
        select_short_bridges(faces, np.array([True]), np.array([False]), np.array([True]))


def test_two_metric_views_support_edge_and_never_use_sensor_depth():
    camera = Camera(width=64, height=48, fx=40., fy=40., cx=32., cy=24.)
    vertices = np.array([[0., 0., 1.], [.1, 0., 1.],
                         [0., .1, 1.], [.1, .1, 1.]], dtype=np.float32)
    old = np.array([[0, 1, 2]], dtype=np.int32)
    candidate = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    depths = [np.ones((48, 64), dtype=np.float32) for _ in range(2)]
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = .06
    selected, stats = filter_bridge_faces(vertices, old, vertices, candidate,
                                          depths, poses, camera)
    assert selected.tolist() == [True, True]
    assert stats["baseline_faces_preserved"] == 1
    assert stats["sensor_depth_used"] is False
    poses[1][0, 3] = 0
    selected, _ = filter_bridge_faces(vertices, old, vertices, candidate,
                                      depths, poses, camera)
    assert selected.tolist() == [True, False]
