"""Independent ray/plane projection oracle and fail-closed pose regressions.

The oracle computes depth by intersecting camera rays with a geometric plane,
not by reusing the renderer's barycentric interpolation implementation.
Synthetic tests only: no claims about desk V2's real-world error causes.
"""
import math

import numpy as np
import pytest

from echo4d.holdout import rasterize_mesh_depth
from echo4d.model import Camera, DataError


CAMERA = Camera(32, 24, 24, 24, 15.5, 11.5)
TRIANGLE_CAMERA = np.array(
    [[-1.5, -1.0, 1.5], [1.5, -1.0, 3.0], [0.0, 1.5, 2.0]],
    dtype=np.float64,
)
FACES = np.array([[0, 1, 2]], dtype=np.int32)


def test_oblique_triangle_rotated_translated_camera_matches_ray_plane_oracle():
    yaw, pitch = 0.37, -0.21
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rotation = ry @ rx
    translation = np.array([0.4, -0.3, 1.1])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    world_vertices = TRIANGLE_CAMERA @ rotation.T + translation

    depth, _ = rasterize_mesh_depth(world_vertices, FACES, CAMERA, pose)
    hits = np.argwhere(depth > 0)
    assert len(hits) > 50
    a, b, c = TRIANGLE_CAMERA
    normal = np.cross(b - a, c - a)
    for y, x in hits:
        ray = np.array([(x - CAMERA.cx) / CAMERA.fx,
                        (y - CAMERA.cy) / CAMERA.fy, 1.0])
        expected_z = np.dot(normal, a) / np.dot(normal, ray)
        assert depth[y, x] == pytest.approx(expected_z, abs=2e-6)

    # Winding reversal cannot change the measured visible surface.
    reverse, _ = rasterize_mesh_depth(world_vertices, FACES[:, ::-1], CAMERA, pose)
    np.testing.assert_array_equal(depth, reverse)


@pytest.mark.parametrize("invalid_kind", ["scale", "reflection", "homogeneous_row"])
def test_rasterizer_rejects_nonrigid_camera_pose(invalid_kind):
    pose = np.eye(4)
    if invalid_kind == "scale":
        pose[0, 0] = 1.1
    elif invalid_kind == "reflection":
        pose[0, 0] = -1
    else:
        pose[3, 0] = 0.2
    with pytest.raises(DataError, match="right-handed rigid transform"):
        rasterize_mesh_depth(TRIANGLE_CAMERA, FACES, CAMERA, pose)
