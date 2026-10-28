"""Pure geometry guards; synthetic tests do not claim a real-scene result."""
import numpy as np
import pytest
from tum_room_microprune import micro_island_mask


def test_only_tiny_compact_disconnected_island_is_removed():
    vertices = np.array([
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        [2, 0, 1], [2.02, 0, 1], [2.02, .02, 1], [2, .02, 1],
        [3, 0, 1], [3.3, 0, 1], [3, .01, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3],
                      [4, 5, 6], [4, 6, 7], [8, 9, 10]], dtype=np.int32)
    mask, report = micro_island_mask(vertices, faces, max_faces=2)
    assert mask.tolist() == [False, False, True, True, False]
    assert report["removed_faces"] == 2
    assert report["removed_components"] == 1
    assert report["unknown_surface_fill"] is False


def test_reject_bad_mesh_and_invalid_thresholds():
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    f = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="Invalid observed mesh"):
        micro_island_mask(v, np.array([[0, 1, 9]]))
    for bad in (0, -1, True, 101):
        with pytest.raises(ValueError, match="thresholds"):
            micro_island_mask(v, f, max_faces=bad)
