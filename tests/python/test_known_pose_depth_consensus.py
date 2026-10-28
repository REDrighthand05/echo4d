"""Synthetic checks for observation-preserving RGB depth regularization."""
import numpy as np
import pytest
from echo4d.known_pose_depth_consensus import regularize_observed_depth


def test_flat_noisy_observed_surface_improves_without_filling():
    rng = np.random.default_rng(123)
    z = (2 + rng.normal(0, .02, (60, 80))).astype('float32')
    z[10:20, 10:20] = 0
    rgb = np.full((60, 80, 3), 128, dtype='uint8')
    result, stats = regularize_observed_depth(z, rgb)
    assert np.all(result[10:20, 10:20] == 0)
    assert np.mean(np.abs(result[25:45, 25:55] - 2)) < np.mean(np.abs(z[25:45, 25:55] - 2))
    assert stats['newly_filled_pixels'] == 0


def test_depth_and_color_edge_preserved():
    z = np.full((40, 60), 1.5, dtype='float32')
    z[:, 30:] = 3
    rgb = np.full((40, 60, 3), 40, dtype='uint8')
    rgb[:, 30:] = 200
    result, _ = regularize_observed_depth(z, rgb)
    assert np.array_equal(result, z)


def test_single_speckle_removed_and_unknown_stays_unknown():
    z = np.zeros((30, 30), dtype='float32')
    z[15, 15] = 2
    rgb = np.zeros((30, 30, 3), dtype='uint8')
    result, stats = regularize_observed_depth(z, rgb)
    assert not result.any() and stats['removed_isolated_pixels'] == 1


def test_invalid_depth_rejected():
    z = np.ones((20, 20), dtype='float32')
    z[2, 2] = np.nan
    with pytest.raises(ValueError):
        regularize_observed_depth(z, np.zeros((20, 20, 3), dtype='uint8'))
