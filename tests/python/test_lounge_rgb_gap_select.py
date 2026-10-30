"""Gap-targeted RGB cameras: deterministic geometry-only selection checks."""
import numpy as np
import pytest

from lounge_rgb_gap_select import choose_gap_targets


def make_poses(n):
    poses = []
    for i in range(n):
        pose = np.eye(4)
        pose[:3, 3] = [(i % 8) * .25, (i // 8) * .30, 0.]
        poses.append(pose)
    return poses


def test_gap_selection_prefers_visible_holes_and_spreads_views():
    poses = make_poses(32)
    rows = [{"index": i, "measured_visible_fraction": .4,
             "near_surface_gap_fraction": .28 if i in (2, 10, 18, 26)
             else .015} for i in range(32)]
    a = choose_gap_targets(rows, poses, 4)
    b = choose_gap_targets(list(reversed(rows)), poses, 4)
    assert a == b
    assert a == [2, 10, 18, 26]
    assert len(set(a)) == 4


def test_gap_selection_needs_real_visible_candidates():
    poses = make_poses(4)
    with pytest.raises(ValueError, match="Too few"):
        choose_gap_targets([{"index": 1, "measured_visible_fraction": .5,
                             "near_surface_gap_fraction": .2}], poses, 2)
    with pytest.raises(ValueError, match="Fractions"):
        choose_gap_targets([{"index": 1, "measured_visible_fraction": 1.2,
                             "near_surface_gap_fraction": .2}], poses, 1)


def test_no_depth_or_rgb_needed_for_camera_choice():
    poses = make_poses(12)
    rows = [{"index": i, "measured_visible_fraction": .3,
             "near_surface_gap_fraction": .1 + i*.001} for i in range(12)]
    chosen = choose_gap_targets(rows, poses, 5)
    assert len(chosen) == 5
    assert all(i in range(12) for i in chosen)
