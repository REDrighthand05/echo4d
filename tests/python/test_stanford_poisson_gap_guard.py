"""Geometric safety checks for real Stanford Poisson gap candidate experiment."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from stanford_poisson_gap_guard import boundary_vertices, depth_conflict_mask


def test_boundary_vertices_square_and_shared_edge():
    vertices = np.zeros((4, 3))
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    assert set(boundary_vertices(vertices, triangles)) == {0, 1, 2, 3}
    assert len(boundary_vertices(vertices, np.array([[0, 1, 2]]))) == 3


def test_depth_veto_only_rejects_patch_in_front_of_measured_surface():
    k = [4, 4, 2., 2., 1.5, 1.5]
    pose = np.eye(4)
    point = np.array([[0., 0., 1.]])
    measured_behind = np.full((4, 4), 2000, dtype=np.uint16)
    measured_in_front = np.full((4, 4), 900, dtype=np.uint16)
    missing = np.zeros((4, 4), dtype=np.uint16)
    assert depth_conflict_mask(point, [pose], k, [measured_behind])[0]
    assert not depth_conflict_mask(point, [pose], k, [measured_in_front])[0]
    assert not depth_conflict_mask(point, [pose], k, [missing])[0]


def test_depth_veto_across_independent_cameras_and_behind_camera():
    k = [4, 4, 2., 2., 1.5, 1.5]
    pose = np.eye(4)
    empty = np.zeros((4, 4), dtype=np.uint16)
    behind = np.full((4, 4), 1500, dtype=np.uint16)
    points = np.array([[0., 0., 1.], [0., 0., -1.]])
    mask = depth_conflict_mask(points, [pose, pose], k, [empty, behind])
    assert mask.tolist() == [True, False]
