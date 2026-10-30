import numpy as np
import pytest
from redwood_prior_poisson_continuity import source_support_mask


def test_support_marks_inferred_without_rewriting_geometry():
    distances = np.array([0., .02, .06, .09, .01])
    densities = np.array([10., 9., 8., 7., .1])
    before = distances.copy()
    mask = source_support_mask(distances, densities, threshold_m=.05)
    assert mask.tolist() == [True, True, False, False, False]
    assert np.array_equal(distances, before)


def test_invalid_support_input_is_rejected():
    with pytest.raises(ValueError):
        source_support_mask([0, -1], [1, 2])
    with pytest.raises(ValueError):
        source_support_mask([0, 1], [1])
    with pytest.raises(ValueError):
        source_support_mask([0], [np.nan])
    with pytest.raises(ValueError):
        source_support_mask([0], [1], threshold_m=0)
