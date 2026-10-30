import numpy as np
import pytest

from redwood_multiview_depth_consistency import (
    compare_depths, multiview_guard, depth_patch,
)


K = (11, 11, 100., 100., 5., 5.)
POSE = np.eye(4)


def measured(mm):
    return np.full((11, 11), mm, dtype=np.uint16)


def test_same_depth_is_supported_not_removed():
    a = measured(1000)
    yes, no = compare_depths(a, POSE, a, POSE, K)
    assert yes[5, 5] and not no[5, 5]
    clean, removed = multiview_guard(a, POSE, [(a, POSE)], [(a, POSE)], K)
    assert removed == 0 and np.array_equal(clean, a)


def test_two_independent_sides_must_disagree():
    source, farther, same = measured(1000), measured(1200), measured(1000)
    yes, no = compare_depths(source, POSE, farther, POSE, K)
    assert no[5, 5] and not yes[5, 5]
    clean, removed = multiview_guard(
        source, POSE, [(farther, POSE)], [(farther, POSE)], K)
    assert removed > 0 and clean[5, 5] == 0
    assert source[5, 5] == 1000  # input never mutated
    clean, removed = multiview_guard(
        source, POSE, [(farther, POSE)], [(same, POSE)], K)
    assert removed == 0 and clean[5, 5] == 1000


def test_occlusion_and_missing_depth_are_not_disproof():
    source = measured(1200)
    closer = measured(1000)
    _, no = compare_depths(source, POSE, closer, POSE, K)
    assert not no[5, 5]
    clean, removed = multiview_guard(
        source, POSE, [(closer, POSE)], [(closer, POSE)], K)
    assert removed == 0 and clean[5, 5] == 1200
    missing = measured(0)
    _, no = compare_depths(source, POSE, missing, POSE, K)
    assert not no.any()


def test_temporal_endpoints_are_kept():
    source = measured(1000)
    far = measured(1200)
    clean, removed = multiview_guard(source, POSE, [], [(far, POSE)], K)
    assert removed == 0 and np.array_equal(clean, source)


def test_patch_supports_small_projection_offset():
    source = measured(1000)
    target = measured(1000)
    shifted = np.eye(4)
    shifted[0, 3] = .01  # 1 px at 1m and fx=100
    yes, no = compare_depths(source, POSE, target, shifted, K)
    assert yes[5, 5] and not no[5, 5]


def test_rejects_wrong_dtype_and_invalid_tolerance():
    with pytest.raises(ValueError):
        depth_patch(measured(1000).astype(np.float32))
    with pytest.raises(ValueError):
        compare_depths(measured(1000), POSE, measured(1000), POSE, K, .001)
