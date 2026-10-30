import numpy as np
import pytest
from redwood_conservative_hole_patch import infer_small_hole_faces, triangulate_polygon


def ring(inner_z=0):
    outer = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    inner = [[-.08, -.08, 0], [.08, -.08, 0],
             [.08, .08, inner_z], [-.08, .08, 0]]
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces.extend([[i, j, 4 + j], [i, 4 + j, 4 + i]])
    return np.array(outer + inner), np.array(faces)


def test_small_hole_capped_without_moving_original_geometry():
    vertices, faces = ring()
    original = faces.copy()
    patch, report = infer_small_hole_faces(vertices, faces, max_perimeter=1)
    assert report["inferred_small_holes"] == 1
    assert len(patch) == 2
    assert set(patch.flatten()) == {4, 5, 6, 7}
    assert np.array_equal(faces, original)


def test_nonplanar_and_large_holes_not_invented():
    vertices, faces = ring(.1)
    patch, _ = infer_small_hole_faces(vertices, faces, max_perimeter=1)
    assert len(patch) == 0
    vertices, faces = ring()
    patch, _ = infer_small_hole_faces(vertices, faces, max_perimeter=.3)
    assert len(patch) == 0


def test_concave_polygon_ear_clipping():
    polygon = np.array([[0, 0], [2, 0], [2, 2], [1, 1], [0, 2]])
    assert len(triangulate_polygon(polygon)) == 3


def test_invalid_mesh_rejected():
    vertices, _ = ring()
    with pytest.raises(ValueError):
        infer_small_hole_faces(vertices, np.array([[999, 1, 2]]))
