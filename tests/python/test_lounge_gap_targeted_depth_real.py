"""Fast depth-coverage regression for gap-directed real Lounge view selection."""
import numpy as np
import pytest

from lounge_gap_targeted_depth_real import gap_pixels


def test_gap_count_excludes_invalid_depth_and_matches_visible_surface():
    d = np.full((20, 20), 2.0, np.float32)
    z = np.full_like(d, 2.03)
    missing, valid = gap_pixels(d, z)
    assert missing == 0
    assert valid == 400
    z[4:16, 4:16] = np.inf
    missing, valid = gap_pixels(d, z)
    assert missing == 144 and valid == 400


def test_disagreement_and_occlusion_are_candidates_not_claimed_reconstruction():
    d = np.full((20, 20), 2.0, np.float32)
    z = np.full_like(d, 2.25)
    assert gap_pixels(d, z) == (400, 400)
    z[:] = 1.75
    assert gap_pixels(d, z) == (400, 400)


def test_invalid_and_depth_discontinuity_do_not_trigger_gap_selection():
    d = np.full((20, 20), 2.0, np.float32)
    d[:, 10:] = 3.0
    d[:3, :] = 0.0
    z = np.full_like(d, np.inf)
    missing, stable = gap_pixels(d, z)
    assert 0 < stable < 400
    assert missing == stable


def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        gap_pixels(np.ones((4, 4)), np.ones((4, 5)))
