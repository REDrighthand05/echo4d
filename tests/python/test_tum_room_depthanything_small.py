"""Synthetic-only calibration checks; model weights and real room run in Actions."""
import numpy as np
import pytest
pytest.importorskip('cv2', reason='OpenCV installed by real-room workflow')

from echo4d.model import DataError
from tum_room_depthanything_small import calibrate_relative_depth


def scene():
    h, w = 80, 140
    yy, xx = np.mgrid[:h, :w]
    truth = (.9 + 1.5 * xx / w + .4 * yy / h).astype(np.float32)
    relative = (5. + 3. / truth).astype(np.float32)
    return truth, relative


def test_metric_from_sparse_rgb_anchors_with_outliers():
    truth, relative = scene()
    rng = np.random.default_rng(6)
    seed = truth.copy()
    seed[rng.random(seed.shape) > .4] = 0
    corrupt = (rng.random(seed.shape) < .12) & (seed > 0)
    seed[corrupt] = 3.8
    metric, receipt = calibrate_relative_depth(relative, seed)
    assert receipt["status"] == "calibrated_from_known_pose_rgb_only"
    assert receipt["inverse_depth_slope"] > 0
    assert np.count_nonzero(metric) > 6000
    assert np.median(np.abs(metric[metric > 0] - truth[metric > 0])) < .10


def test_unknown_far_from_rgb_anchors_stays_unknown():
    truth, relative = scene()
    seed = np.zeros_like(truth)
    seed[:, :30] = truth[:, :30]
    metric, receipt = calibrate_relative_depth(relative, seed)
    assert receipt["status"] == "calibrated_from_known_pose_rgb_only"
    assert np.all(metric[:, 90:] == 0)


def test_flat_relative_depth_never_claims_metric():
    truth, _ = scene()
    metric, receipt = calibrate_relative_depth(
        np.full_like(truth, 5.), truth)
    assert receipt["status"] == "flat_relative_depth"
    assert not metric.any()


def test_too_few_known_pose_rgb_anchors_fails_closed():
    truth, relative = scene()
    seed = np.zeros_like(truth)
    seed[:5, :5] = truth[:5, :5]
    metric, receipt = calibrate_relative_depth(relative, seed)
    assert receipt["status"] == "insufficient_rgb_metric_anchors"
    assert not metric.any()


def test_bad_shapes_or_nan_rejected():
    truth, relative = scene()
    with pytest.raises(DataError):
        calibrate_relative_depth(relative[:-1], truth)
    relative[0, 0] = np.nan
    with pytest.raises(DataError):
        calibrate_relative_depth(relative, truth)
