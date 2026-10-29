"""Synthetic geometry guards; the GitHub workflow runs the real Redwood data."""
import numpy as np
import pytest

from redwood_coverage_budget_fusion import (
    measured_surface_cells, select_coverage_frames,
)


def _k():
    return (32, 24, 20., 20., 15.5, 11.5)


def test_native_depth_world_cell_geometry_and_invalid_inputs():
    depth = np.full((24, 32), 2000, dtype=np.uint16)
    identity = np.eye(4)
    a = measured_surface_cells(depth, identity, _k(), step=8, cell_m=.10)
    assert len(a) >= 4
    translated = identity.copy()
    translated[0, 3] = 2.0
    b = measured_surface_cells(depth, translated, _k(), step=8, cell_m=.10)
    assert len(b) == len(a)
    assert a.isdisjoint(b)
    with pytest.raises(ValueError, match="Invalid original"):
        measured_surface_cells(depth.astype(np.float32), identity, _k())
    with pytest.raises(ValueError, match="Invalid original"):
        measured_surface_cells(depth[:, 1:], identity, _k())
    with pytest.raises(ValueError, match="Invalid original"):
        measured_surface_cells(depth, np.zeros((4, 4)), _k())


def test_fixed_budget_prioritizes_new_real_surface_cells():
    # 8 original frames, 3 mandatory evenly spaced anchors: 0, 4, 7.
    cells = [{1}, {1}, {10, 11}, {1}, {1}, {12, 13},
             {1}, {1}]
    chosen, info = select_coverage_frames(cells, budget=5, anchors=3)
    assert len(chosen) == len(set(chosen)) == 5
    assert {0, 4, 7}.issubset(chosen)
    assert {2, 5}.issubset(chosen)
    assert info["coverage_greedy_count"] == 2
    assert info["uniform_fallback_count"] == 0


def test_fallback_uses_only_real_distinct_frames():
    selected, info = select_coverage_frames([set() for _ in range(9)],
                                            budget=6, anchors=3)
    assert len(selected) == len(set(selected)) == 6
    assert all(0 <= i < 9 for i in selected)
    assert info["uniform_fallback_count"] == 3
    with pytest.raises(ValueError):
        select_coverage_frames([set() for _ in range(4)], budget=5, anchors=3)
