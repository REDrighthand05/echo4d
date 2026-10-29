"""CPU geometric behavior tests; synthetic metric Z, no model or sensor data."""
import numpy as np
import pytest
from tum_room_depth_support import filter_reference, filter_all

K = np.array([[80., 0., 40.], [0., 80., 30.], [0., 0., 1.]])


def pose(x=0):
    p = np.eye(4)
    p[0, 3] = x
    return p


def test_consistent_plane_survives():
    z = np.full((60, 80), 2., dtype=np.float32)
    result, r = filter_reference(z, pose(), [(z, pose(.1)), (z, pose(-.1))], K)
    assert np.array_equal(result, z)
    assert r["supported_pixels"] > 3000
    assert r["removed"] == 0


def test_ghost_in_front_of_two_visible_surfaces_removed():
    z = np.full((60, 80), 2., dtype=np.float32)
    bad = z.copy()
    bad[20:40, 25:55] = 1.2
    result, r = filter_reference(bad, pose(), [(z, pose(.1)), (z, pose(-.1))], K)
    assert result[30, 40] == 0
    assert result[5, 40] == 2
    assert r["removed"] > 100


def test_closer_neighbor_is_occlusion_not_contradiction():
    ref = np.full((60, 80), 2., dtype=np.float32)
    closer = np.full((60, 80), 1.2, dtype=np.float32)
    result, r = filter_reference(ref, pose(), [(closer, pose(.1)), (closer, pose(-.1))], K)
    assert np.array_equal(result, ref)
    assert r["occluded_observations"] > 0


def test_unknown_neighbors_never_invent_surface_or_reject():
    z = np.full((60, 80), 2., dtype=np.float32)
    unknown = np.zeros_like(z)
    result, r = filter_reference(z, pose(), [(unknown, pose(.1)), (unknown, pose(-.1))], K)
    assert np.array_equal(result, z)
    assert r["comparable_observations"] == 0


def test_two_contradictions_required():
    z = np.full((60, 80), 1.2, dtype=np.float32)
    farther = np.full((60, 80), 2., dtype=np.float32)
    result, r = filter_reference(z, pose(), [(farther, pose(.1))], K)
    assert np.array_equal(result, z)
    assert r["front_conflicted_pixels"] > 0


def test_original_maps_not_mutated_and_empty_preserved():
    z = np.full((60, 80), 2., dtype=np.float32)
    bad = z.copy()
    bad[20:40, 25:55] = 1.2
    empty = np.zeros_like(z)
    maps = [bad, z, z, empty]
    poses = [pose(), pose(.1), pose(-.1), pose(.2)]
    copies = [x.copy() for x in maps]
    out, r = filter_all(maps, poses, K, lambda poses, i, limit: [j for j in range(4) if j != i])
    assert all(np.array_equal(x, y) for x, y in zip(maps, copies))
    assert out[0][30, 40] == 0
    assert not np.any(out[3])


def test_invalid_negative_depth_rejected():
    z = np.ones((60, 80), dtype=np.float32)
    z[0, 0] = -1
    with pytest.raises(ValueError):
        filter_reference(z, pose(), [], K)
