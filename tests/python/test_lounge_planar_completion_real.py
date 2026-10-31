"""Metric hole completion must never fill open or large unobserved regions."""
import numpy as np
import pytest
from lounge_planar_completion_real import bounded_planar_holes


def test_small_closed_hole_only():
    a = np.zeros((60, 60), bool)
    a[5:50, 5:50] = True
    a[20:24, 20:24] = False
    a[30:46, 30:46] = False
    result = bounded_planar_holes(a, cell_m=.03, max_area_m2=.12)
    assert result[21, 21]
    assert not result[35, 35]
    assert not result[2, 2]


def test_open_unseen_region_not_filled():
    a = np.zeros((60, 60), bool)
    a[5:50, 5:50] = True
    a[15:20, 5:20] = False
    result = bounded_planar_holes(a)
    assert not result[17, 8]


def test_invalid_grid_rejected():
    with pytest.raises(ValueError):
        bounded_planar_holes(np.ones((5, 5, 3)))
