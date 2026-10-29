"""Small deterministic tests for far-side-only Stanford depth-edge filtering."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from stanford_depth_edge_farside import mask_far_side_depth_edges


def test_far_side_only_at_occlusion_boundary():
    depth = np.array([[1000, 1000, 1700, 1700, 1700],
                      [1000, 1000, 1700, 1700, 1700]], dtype=np.uint16)
    filtered, removed = mask_far_side_depth_edges(depth)
    assert np.all(filtered[:, :2] == 1000)
    assert np.all(filtered[:, 2] == 0)
    assert np.all(filtered[:, 3:] == 1700)
    assert int(removed.sum()) == 2


def test_invalid_depth_and_subthreshold_jitter_do_not_create_edges():
    depth = np.array([[0, 1000, 1040, 1080, 0],
                      [0, 1000, 1040, 1080, 0]], dtype=np.uint16)
    filtered, removed = mask_far_side_depth_edges(depth)
    np.testing.assert_array_equal(filtered, depth)
    assert not removed.any()


def test_relative_threshold_at_far_depth():
    depth = np.array([[3000, 3110, 3110]], dtype=np.uint16)
    # 110mm is smaller than 3.5% of 3110mm (108.85mm) only just larger;
    # at 3.8m the relative threshold dominates the 100mm floor.
    depth = np.array([[3700, 3820, 3820]], dtype=np.uint16)
    filtered, removed = mask_far_side_depth_edges(depth)
    np.testing.assert_array_equal(filtered, depth)
    assert not removed.any()


def test_nearer_side_preserved_and_edge_no_wraparound():
    depth = np.array([[1700, 1700, 1000, 1000]], dtype=np.uint16)
    filtered, removed = mask_far_side_depth_edges(depth)
    np.testing.assert_array_equal(filtered, np.array(
        [[1700, 0, 1000, 1000]], dtype=np.uint16))
    assert removed[0, 1] and not removed[0, 0]


def test_reject_invalid_depth_type_and_parameters():
    with pytest.raises(ValueError, match="uint16"):
        mask_far_side_depth_edges(np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="parameters"):
        mask_far_side_depth_edges(np.ones((4, 4), dtype=np.uint16),
                                  absolute_mm=-1)
