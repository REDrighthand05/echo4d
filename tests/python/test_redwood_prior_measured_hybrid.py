import numpy as np
import pytest
from redwood_prior_measured_hybrid import select_measured_detail_faces


def test_only_measured_detail_missing_from_prior():
    vertices = np.array([[0, 0, 0], [.02, 0, 0], [0, .02, 0],
                         [1, 0, 0], [1.02, 0, 0], [1, .02, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    distances = np.array([.005, .006, .007, .1, .1, .1])
    selected = select_measured_detail_faces(
        vertices, faces, distances, min_component_faces=1)
    assert selected.tolist() == [[3, 4, 5]]


def test_rejects_long_bridges_and_far_floaters():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, .02, 0],
                         [2, 0, 0], [2.02, 0, 0], [2, .02, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    distances = np.array([.1, .1, .1, .6, .6, .6])
    selected = select_measured_detail_faces(
        vertices, faces, distances, min_component_faces=1)
    assert selected.shape == (0, 3)


def test_isolated_fragments_need_connected_support():
    vertices = np.array([[0, 0, 0], [.02, 0, 0], [0, .02, 0],
                         [1, 0, 0], [1.02, 0, 0], [1, .02, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    assert select_measured_detail_faces(
        vertices, faces, np.full(6, .1), min_component_faces=2).shape == (0, 3)


def test_invalid_geometry_fails_explicitly():
    with pytest.raises(ValueError):
        select_measured_detail_faces(
            np.zeros((3, 3)), np.array([[0, 1, 5]]), np.ones(3))
    with pytest.raises(ValueError):
        select_measured_detail_faces(
            np.zeros((3, 3)), np.array([[0, 1, 2]]),
            np.array([.1, np.nan, .1]))
