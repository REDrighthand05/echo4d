"""CPU-only sampling invariance; synthetic counterexamples, not real Hi3D evidence."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from hi3d_geometry_gate import score, surface_sample


def scene():
    x, y = np.meshgrid(np.linspace(-0.6, 0.6, 4),
                       np.linspace(-0.6, 0.6, 4))
    vertices = np.column_stack([x.ravel(), y.ravel(), np.ones(16)])
    faces = []
    for row in range(3):
        for col in range(3):
            i = row * 4 + col
            faces.extend(([i, i + 1, i + 4], [i + 1, i + 5, i + 4]))
    u, v = np.meshgrid(np.linspace(-0.59, 0.59, 32),
                       np.linspace(-0.59, 0.59, 32))
    measured = np.column_stack([u.ravel(), v.ravel(), np.ones(1024)])
    return vertices, np.asarray(faces), measured


def test_face_permutation_and_winding_do_not_change_samples_or_gate_metrics():
    vertices, faces, measured = scene()
    # Same indexed triangles, reversed face order AND vertex winding.
    changed = faces[::-1, ::-1].copy()
    np.testing.assert_array_equal(
        surface_sample(vertices, faces, 3000, seed=17),
        surface_sample(vertices, changed, 3000, seed=17))
    assert score((vertices, faces), measured, 3000, seed=17) == score(
        (vertices, changed), measured, 3000, seed=17)


def test_canonicalization_does_not_hide_a_large_unsupported_surface():
    vertices, faces, measured = scene()
    displaced = vertices.copy()
    displaced[:, 2] += 2.0
    extended_vertices = np.concatenate([vertices, displaced])
    extended_faces = np.concatenate([faces, faces + len(vertices)])
    original = score((vertices, faces), measured, 3000)
    added = score((extended_vertices, extended_faces), measured, 3000)
    reordered = score((extended_vertices, extended_faces[::-1, ::-1]),
                      measured, 3000)
    assert added == reordered
    assert added["surface_to_measurements_rmse_m"] > 1.0
    assert added["surface_to_measurements_rmse_m"] > (
        original["surface_to_measurements_rmse_m"] + 0.5)
