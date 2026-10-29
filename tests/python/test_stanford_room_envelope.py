"""CPU regression guards for inferred room-shell geometry."""
import numpy as np
import pytest
from scipy.spatial import cKDTree
from stanford_room_envelope import fit_room_planes, extend_plane


def grid_points(n=100):
    a = np.arange(n, dtype=float) * .05
    x, y = np.meshgrid(a, a, indexing="ij")
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(n*n)))
    normals = np.tile([0., 0., 1.], (len(vertices), 1))
    return vertices, normals


def test_large_measured_plane_detected_without_prior():
    vertices, normals = grid_points()
    planes = fit_room_planes(vertices, normals, max_planes=2)
    assert len(planes) == 1
    assert len(planes[0][3]) > 4000
    assert abs(planes[0][1][2]) > .99


def test_extension_reaches_outside_measured_outline_but_is_bounded():
    vertices, normals = grid_points()
    rgb = np.tile([.5, .6, .7], (len(vertices), 1))
    center = vertices.mean(axis=0)
    plane = (center, np.array([0., 0., 1.]),
             np.array([[1., 0., 0.], [0., 1., 0.]]),
             np.arange(len(vertices)), np.array([4.95, 4.95]))
    out, faces, colors, cells = extend_plane(
        vertices, rgb, plane, cKDTree(vertices))
    assert cells > 0 and len(faces) == 2 * cells
    assert colors.shape == (len(out), 3)
    assert out[:, 0].min() < vertices[:, 0].min()
    assert out[:, 0].min() > vertices[:, 0].min() - .7
    assert out[:, 0].max() < vertices[:, 0].max() + .7
    assert np.max(np.abs(out[:, 2])) < 1e-6


def test_invalid_measured_geometry_rejected():
    with pytest.raises(ValueError):
        fit_room_planes(np.ones((5, 3)), np.ones((5, 3)))
    vertices, normals = grid_points()
    vertices[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_room_planes(vertices, normals)
