"""Geometry tests: new faces need two independent inferred-Z observations."""
import numpy as np
import pytest

from echo4d.model import Camera
from echo4d.known_pose_edge_crossview import filter_crossview_faces


def _case(z=1., offset=.06, observed=True):
    vertices = np.array([[0., 0., 1.], [.10, 0., 1.],
                         [0., .10, 1.], [.10, .10, z]], dtype=np.float32)
    original = np.array([[0, 1, 2]], dtype=np.int32)
    candidate = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    camera = Camera(width=64, height=48, fx=40., fy=40., cx=32., cy=24.)
    depth = np.full((48, 64), 1. if observed else 0., dtype=np.float32)
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = offset
    return vertices, original, candidate, [depth, depth.copy()], poses, camera


def _run(z=1., offset=.06, observed=True):
    v, old, new, depths, poses, k = _case(z, offset, observed)
    return filter_crossview_faces(v, old, v, new, depths, poses, k)


def test_two_distinct_matching_inferred_views_extend_original_surface():
    keep, stats = _run()
    assert keep.tolist() == [True, True]
    assert stats["baseline_faces_preserved"] == 1
    assert stats["extra_faces_retained_after_connectivity"] == 1
    assert not stats["sensor_depth_used"]


def test_same_camera_center_cannot_vote_twice():
    keep, stats = _run(offset=0.)
    assert keep.tolist() == [True, False]
    assert stats["extra_faces_with_two_independent_views"] == 0


def test_unsupported_or_wrong_depth_preserves_original_only():
    keep, stats = _run(observed=False)
    assert keep.tolist() == [True, False]
    assert stats["extra_faces_rejected"] == 1
    keep, _ = _run(z=1.8)
    assert keep.tolist() == [True, False]


def test_missing_baseline_face_is_rejected():
    v, old, new, depths, poses, k = _case()
    with pytest.raises(ValueError, match="lost original"):
        filter_crossview_faces(v, old, v, new[1:], depths, poses, k)


def test_invalid_inferred_depth_is_rejected():
    v, old, new, depths, poses, k = _case()
    depths[0][1, 1] = np.nan
    with pytest.raises(ValueError, match="finite inferred Z"):
        filter_crossview_faces(v, old, v, new, depths, poses, k)
