"""Temporal measured-depth geometry oracle; not real-scene acceptance."""
import numpy as np
import pytest
from echo4d.model import Camera, DataError
from temporal_depth_consensus import refine_observed_depth


def setup():
    cam = Camera(15, 15, 14., 14., 7., 7.)
    pose, a, b = np.eye(4), np.eye(4), np.eye(4)
    a[0, 3], b[0, 3] = -.03, .03
    return cam, pose, a, b


def test_two_witnesses_correct_small_observed_outlier_without_mutation():
    cam, pose, a, b = setup()
    src = np.ones((15, 15), np.float32)
    src[7, 7] = 1.04
    observed = np.ones_like(src)
    revised, changed = refine_observed_depth(src, pose, [(observed, a), (observed, b)], cam, .06)
    assert changed > 0 and abs(float(revised[7, 7]) - 1.) < .01
    assert src[7, 7] > 1.03


def test_unknown_holes_and_occluded_foreground_are_not_filled():
    cam, pose, a, b = setup()
    src = np.ones((15, 15), np.float32)
    src[2, 2], src[7, 7] = 0., .7
    witness = np.ones_like(src)
    revised, _ = refine_observed_depth(src, pose, [(witness, a), (witness, b)], cam, .06)
    assert revised[2, 2] == 0 and revised[7, 7] == src[7, 7]


def test_insufficient_or_disagreeing_evidence_changes_nothing():
    cam, pose, a, b = setup()
    src = np.ones((15, 15), np.float32)
    src[7, 7] = 1.04
    agree, far = np.ones_like(src), np.full_like(src, 1.4)
    for witnesses in ([(agree, a)], [(agree, a), (far, b)]):
        revised, count = refine_observed_depth(src, pose, witnesses, cam, .06)
        assert count == 0 and np.array_equal(revised, src)


def test_depth_jump_does_not_bridge_measured_occlusion():
    cam, pose, a, b = setup()
    src = np.ones((15, 15), np.float32)
    src[:, 8:], src[:, 3] = 2., 0.
    revised, changed = refine_observed_depth(src, pose, [(src, a), (src, b)], cam, .06)
    assert changed == 0 and np.array_equal(revised, src)


def test_bad_type_pose_or_tolerance_fails_closed():
    cam, pose, a, b = setup()
    src = np.ones((15, 15), np.float32)
    with pytest.raises(DataError, match="resolution/type"):
        refine_observed_depth(src, pose, [(src[:5], a), (src, b)], cam)
    broken = pose.copy()
    broken[0, 0] = 2
    with pytest.raises(DataError, match="rigid"):
        refine_observed_depth(src, broken, [(src, a), (src, b)], cam)
    with pytest.raises(DataError, match="tolerance"):
        refine_observed_depth(src, pose, [], cam, float("nan"))
