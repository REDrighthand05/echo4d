"""Regression tests for real Fountain 16 sensor + 8 RGB-only experiment."""
import numpy as np
import pytest

from echo4d.model import Camera, Frame
from fountain_cpu_rgb_depth8 import (
    metric_from_relative, project_sensor_anchors, select_rgb_targets,
)
from fountain_depth_ladder import choose_fountain_indices


def synthetic_poses():
    poses = np.repeat(np.eye(4)[None], 33, axis=0)
    poses[:, 0, 3] = np.linspace(0, 1., 33)
    return poses


def test_16_measured_8_rgb_only_9_holdout():
    _, measured = choose_fountain_indices(synthetic_poses())
    targets = select_rgb_targets(measured)
    holdout = set(range(33)) - set(measured) - set(targets)
    assert len(measured) == 16 and len(targets) == 8 and len(holdout) == 9
    assert not set(measured) & set(targets)
    assert targets == select_rgb_targets(measured)


def test_reject_bad_cohorts():
    with pytest.raises(ValueError):
        select_rgb_targets((1,2,3))
    with pytest.raises(ValueError):
        select_rgb_targets(tuple(range(16)),count=18)


def test_sensor_projection_and_metric_depth():
    h, w = 90, 120
    yy, xx = np.mgrid[:h, :w]
    depth = (1.25 + xx*.002 + yy*.001).astype(np.float32)
    camera = Camera(w, h, 105., 105., (w-1)/2, (h-1)/2)
    measured = Frame(0., np.eye(4), depth,
                     np.zeros((h,w,3),np.uint8)).validate(camera)
    anchors = project_sensor_anchors([measured],camera,np.eye(4),stride=3)
    assert np.isfinite(anchors).sum() > 900
    np.testing.assert_allclose(anchors[::3,::3],depth[::3,::3],atol=1e-5)
    relative = ((1/depth-.15)/.35).astype(np.float32)
    metric, details = metric_from_relative(relative,anchors)
    valid = metric > 0
    assert valid.mean() > .7, details
    assert np.median(np.abs(metric[valid]-depth[valid])) < .03


def test_no_sensor_support_rejected():
    with pytest.raises(RuntimeError,match="Too few"):
        metric_from_relative(np.ones((40,60),np.float32),
                             np.full((40,60),np.inf,np.float32))
