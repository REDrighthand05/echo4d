"""Focused regression: no held-out measured depth and nested real-frame cohorts."""
import numpy as np
import pytest

from tum_desk_native640_depth_ladder import nested_measured_indices
from tum_desk_known_pose_open3d import split_indices


def test_real_desk_72_frame_split_is_nested_and_excludes_holdouts():
    train, heldout = split_indices(72)
    cohorts = nested_measured_indices(train)
    assert len(train) == 57
    assert len(cohorts[12]) == 12
    assert len(cohorts[24]) == 24
    assert set(cohorts[12]) < set(cohorts[24]) < set(train)
    assert not (set(cohorts[24]) & set(heldout))
    assert cohorts[12][0] == train[0]
    assert cohorts[12][-1] == train[-1]


def test_deterministic_subsets_cover_recording_without_peeking():
    train = tuple(i for i in range(54) if i % 5)
    a = nested_measured_indices(train)
    b = nested_measured_indices(train)
    assert a == b
    assert max(np.diff(a[12])) <= 7
    assert all(i in train for i in a[24])


@pytest.mark.parametrize("training,budgets", [
    ((1, 2, 2, 3), (2, 3)),
    ((2, 1, 3, 4), (2, 3)),
    ((1, 2, 3, 4), (1, 2)),
    ((1, 2, 3, 4), (2, 5)),
    ((1, 2, 3, 4), (3, 2)),
])
def test_invalid_selection_rejected(training, budgets):
    with pytest.raises(ValueError):
        nested_measured_indices(training, budgets)
