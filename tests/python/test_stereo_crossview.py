"""Synthetic geometric oracle and counterexamples; not real-scene evidence."""
import numpy as np
import pytest
from echo4d.model import Camera
from echo4d.stereo_crossview import reject_crossview_foreground


def setup_views():
    cam = Camera(32, 24, 24., 24., 15.5, 11.5)
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = .1
    return cam, a, b


def test_agreeing_plane_is_preserved():
    cam, a, b = setup_views()
    depth = np.full((24, 32), 2., np.float32)
    result, info = reject_crossview_foreground([depth, depth], [a, b], cam)
    assert np.array_equal(result[0], depth)
    assert np.array_equal(result[1], depth)
    assert info[0]['supported_pixels'] > 600
    assert info[0]['rejected_pixels'] == 0


def test_false_foreground_rejected_but_occluded_background_not_rejected():
    cam, a, b = setup_views()
    first = np.full((24, 32), 2., np.float32)
    second = np.full((24, 32), 2., np.float32)
    first[10, 16] = .8  # false foreground, independently farther observation
    first[12, 16] = 3.0  # behind a real occluder: not contradictory
    result, info = reject_crossview_foreground([first, second], [a, b], cam)
    assert result[0][10, 16] == 0
    assert result[0][12, 16] == 3
    assert info[0]['rejected_pixels'] >= 1


def test_unknown_neighbor_does_not_create_or_remove_surface():
    cam, a, b = setup_views()
    first = np.full((24, 32), 2., np.float32)
    second = np.zeros_like(first)
    result, info = reject_crossview_foreground([first, second], [a, b], cam)
    assert np.array_equal(result[0], first)
    assert np.count_nonzero(result[1]) == 0
    assert info[0]['rejected_pixels'] == 0


def test_bad_units_shapes_poses_and_budget_rejected():
    cam, a, b = setup_views()
    d = np.full((24, 32), 2., np.float32)
    for maps, poses, opts in [
        ([d], [a], {}), ([d, d[:, :-1]], [a, b], {}),
        ([d, d], [a, b * 2], {}),
        ([d, d], [a, b], {'neighbor_radius': 20}),
        ([d, d], [a, b], {'tolerance_m': -1}),
        ([d, d.astype(np.int32)], [a, b], {}),
    ]:
        with pytest.raises(ValueError):
            reject_crossview_foreground(maps, poses, cam, **opts)
