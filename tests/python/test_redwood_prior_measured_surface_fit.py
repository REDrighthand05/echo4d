import numpy as np
import pytest
import trimesh

from redwood_prior_measured_surface_fit import fit_prior_to_measured


def grid(z):
    points = np.array([(x * .01, y * .01, z)
                       for y in range(7) for x in range(7)])
    faces = []
    for y in range(6):
        for x in range(6):
            a = y * 7 + x
            faces.extend([[a, a + 1, a + 8], [a, a + 8, a + 7]])
    return trimesh.Trimesh(vertices=points, faces=faces, process=False)


def test_fit_moves_only_measured_supported_prior_vertices():
    prior, observed = grid(0), grid(.012)
    fitted, report = fit_prior_to_measured(
        prior, observed, np.ones(len(prior.vertices), dtype=bool))
    assert report["moved_prior_vertices"] > 0
    assert np.array_equal(fitted.faces, prior.faces)
    assert .004 < fitted.vertices[:, 2].max() <= .025


def test_unobserved_prior_is_unchanged():
    prior, observed = grid(0), grid(.012)
    fitted, report = fit_prior_to_measured(
        prior, observed, np.zeros(len(prior.vertices), dtype=bool))
    assert report["moved_prior_vertices"] == 0
    assert np.array_equal(fitted.vertices, prior.vertices)


def test_invalid_fit_radius_rejected():
    with pytest.raises(ValueError):
        fit_prior_to_measured(grid(0), grid(.01),
                              np.ones(49, dtype=bool), radius=.5)
