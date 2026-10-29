import numpy as np
import pytest

from redwood_depth_edge_guard import guard_depth_edges


def test_preserves_continuous_depth_and_input():
    raw = np.full((8, 9), 1200, dtype=np.uint16)
    result, removed = guard_depth_edges(raw)
    assert removed == 0
    assert np.array_equal(result, raw)
    assert np.array_equal(raw, np.full((8, 9), 1200, dtype=np.uint16))


def test_removes_only_depth_jump_neighbors():
    raw = np.full((7, 7), 1000, dtype=np.uint16)
    raw[3, 3] = 2000
    result, removed = guard_depth_edges(raw, threshold_mm=90)
    assert removed == 1
    assert result[3, 3] == 0
    assert result[3, 2] == 1000  # foreground silhouette survives
    assert result[0, 0] == 1000
    assert raw[3, 3] == 2000


def test_invalid_depth_not_counted_as_jump():
    raw = np.full((7, 7), 1000, dtype=np.uint16)
    raw[3, 3] = 0
    result, removed = guard_depth_edges(raw)
    assert removed == 0
    assert np.array_equal(result, raw)


def test_rejects_wrong_dtype_and_threshold():
    with pytest.raises(ValueError):
        guard_depth_edges(np.ones((3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        guard_depth_edges(np.ones((3, 3), dtype=np.uint16), threshold_mm=5)


def test_preserves_near_object_and_masks_far_neighbors():
    raw = np.full((7, 7), 2000, dtype=np.uint16)
    raw[3, 3] = 1000
    result, removed = guard_depth_edges(raw, threshold_mm=90)
    assert removed == 4
    assert result[3, 3] == 1000
    assert result[3, 2] == 0
    assert result[2, 3] == 0
    assert result[0, 0] == 2000


def test_horizontal_occlusion_masks_far_side_only():
    raw = np.array([[1000, 1000, 2000, 2000]], dtype=np.uint16)
    result, removed = guard_depth_edges(raw, threshold_mm=90)
    assert removed == 1
    assert np.array_equal(result, np.array([[1000, 1000, 0, 2000]], dtype=np.uint16))
