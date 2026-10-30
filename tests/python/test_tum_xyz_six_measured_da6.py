"""Regression for TUM six-sensor + six-RGB predicted-depth adapter."""
import numpy as np
import pytest

from echo4d.model import Camera, Frame
from tum_xyz_six_measured_da6 import (
    calibrate_from_six, measured_projection, test_calibration as synthetic_self_test,
)


def test_calibration_synthetic_metric_from_measured_projection():
    synthetic_self_test()


def test_target_sensor_depth_never_required_for_projection():
    camera = Camera(48, 36, 70., 70., 23.5, 17.5)
    depth = np.full((36, 48), 1.7, dtype=np.float32)
    measured = Frame(0., np.eye(4), depth, np.zeros((36, 48, 3), np.uint8))
    target = np.eye(4)
    target[0, 3] = .03
    projected = measured_projection([measured], camera, target, stride=2)
    assert np.isfinite(projected).sum() > 300
    assert np.isclose(np.median(projected[np.isfinite(projected)]), 1.7, atol=1e-5)


def test_fails_closed_without_six_anchor_support():
    with pytest.raises(RuntimeError, match="Insufficient projected"):
        calibrate_from_six(np.ones((32, 32), np.float32),
                           np.full((32, 32), np.inf, np.float32))


def test_does_not_fabricate_metric_scale_from_constant_relative_prediction():
    predicted = np.ones((80, 100), np.float32)
    observed = np.full((80, 100), np.inf, np.float32)
    observed[::3, ::3] = 1.5
    with pytest.raises(RuntimeError, match="cannot be calibrated"):
        calibrate_from_six(predicted, observed)
