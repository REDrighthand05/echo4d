import numpy as np
import pytest

from redwood_observed_component_cleanup import component_face_mask


def test_remove_only_tiny_disconnected_observed_component():
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int32)
    keep, sizes = component_face_mask(faces, 7, min_faces=2)
    assert keep.tolist() == [True, True, False]
    assert sorted(sizes.tolist()) == [1, 2]


def test_vertex_connected_fragments_stay_together():
    faces = np.array([[0, 1, 2], [0, 3, 4], [5, 6, 7]], dtype=np.int32)
    keep, _ = component_face_mask(faces, 8, min_faces=2)
    assert keep.tolist() == [True, True, False]


def test_reject_invalid_topology_and_threshold():
    with pytest.raises(ValueError):
        component_face_mask(np.array([[0, 1, 99]]), 8)
    with pytest.raises(ValueError):
        component_face_mask(np.array([[0, 1, 2]]), 3, min_faces=0)
    with pytest.raises(ValueError):
        component_face_mask(np.empty((0, 3), dtype=int), 3)
