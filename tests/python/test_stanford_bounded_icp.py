"""Deterministic pose correction guard tests; real Stanford data run is separate."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from stanford_bounded_icp import (
    pose_correction_accepted, corrected_world_to_camera,
    MAX_TRANSLATION_M, MAX_ROTATION_DEG,
)


def rotation_z(degrees):
    angle = np.radians(degrees)
    t = np.eye(4)
    t[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                 [np.sin(angle), np.cos(angle)]]
    return t


def test_accept_small_measured_correction():
    t = rotation_z(0.7)
    t[:3, 3] = [0.009, 0.005, 0.001]
    ok, reason, distance, angle = pose_correction_accepted(t, .7, .018)
    assert ok and reason == "accepted"
    assert distance < MAX_TRANSLATION_M and angle < MAX_ROTATION_DEG


def test_reject_large_or_weak_registration():
    t = np.eye(4)
    t[0, 3] = MAX_TRANSLATION_M + .001
    assert pose_correction_accepted(t, .9, .01)[1] == "correction_too_large"
    t = rotation_z(MAX_ROTATION_DEG + .1)
    assert pose_correction_accepted(t, .9, .01)[1] == "correction_too_large"
    assert pose_correction_accepted(np.eye(4), .2, .01)[1] == "weak_registration"
    assert pose_correction_accepted(np.eye(4), .9, .1)[1] == "weak_registration"


def test_reject_nonrigid_and_nonfinite():
    t = np.eye(4)
    t[0, 0] = 1.05
    assert pose_correction_accepted(t, .8, .01)[1] == "nonrigid_transform"
    t = np.eye(4)
    t[0, 0] = np.nan
    assert pose_correction_accepted(t, .8, .01)[1] == "invalid_transform"


def test_corrected_extrinsic_projects_same_new_world_point():
    e = rotation_z(15)
    e[:3, 3] = [.3, -.1, .7]
    correction = rotation_z(.8)
    correction[:3, 3] = [.01, .004, -.006]
    world_old = np.array([1.3, -.5, 2.0, 1.])
    world_new = correction @ world_old
    refined = corrected_world_to_camera(e, correction)
    np.testing.assert_allclose(refined @ world_new, e @ world_old, atol=1e-10)
