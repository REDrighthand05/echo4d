"""Frozen RGB-inferred local-normal gate: actual geometry, not sensor-Z tuning."""
import numpy as np
import pytest

from echo4d.known_pose_edge_orientation import filter_oriented_bridge_faces
from echo4d.model import Camera


def scene(raised_z=1.0):
    camera = Camera(width=64, height=48, fx=40., fy=40., cx=32., cy=24.)
    vertices = np.array([[0., 0., 1.], [.1, 0., 1.],
                         [0., .1, 1.], [.1, .1, raised_z]], dtype=np.float32)
    original = np.array([[0, 1, 2]], dtype=np.int32)
    candidate = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    depths = [np.ones((48, 64), dtype=np.float32) for _ in range(2)]
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = .06
    return vertices, original, candidate, depths, poses, camera


def test_flat_inferred_plane_preserves_supported_extension():
    vertices, original, candidate, depths, poses, camera = scene()
    keep, report = filter_oriented_bridge_faces(
        vertices, original, vertices, candidate, depths, poses, camera)
    assert keep.tolist() == [True, True]
    assert report["retained_extra_faces"] == 1
    assert report["sensor_depth_used"] is False


def test_steep_bridge_rejected_despite_matching_centroid_depth():
    vertices, original, candidate, depths, poses, camera = scene(1.18)
    keep, report = filter_oriented_bridge_faces(
        vertices, original, vertices, candidate, depths, poses, camera)
    assert keep.tolist() == [True, False]
    assert report["previous_extra_faces"] == 1
    assert report["rejected_by_orientation_or_attachment"] == 1


def test_same_camera_center_not_independent():
    vertices, original, candidate, depths, poses, camera = scene()
    poses[1][0, 3] = 0
    keep, _ = filter_oriented_bridge_faces(
        vertices, original, vertices, candidate, depths, poses, camera)
    assert keep.tolist() == [True, False]


def test_invalid_threshold_is_rejected():
    vertices, original, candidate, depths, poses, camera = scene()
    with pytest.raises(ValueError, match="threshold"):
        filter_oriented_bridge_faces(
            vertices, original, vertices, candidate, depths, poses, camera,
            normal_cosine=1.1)
