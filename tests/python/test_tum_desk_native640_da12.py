"""Target cohort isolation and metric calibration tests for the real Desk experiment."""
import numpy as np
import pytest

from echo4d.model import Camera, Frame
from tum_desk_native640_da12 import select_rgb_targets, project_measured_depth, calibrate_relative_depth
from tum_desk_native640_depth_ladder import nested_measured_indices
from tum_desk_known_pose_open3d import split_indices


def test_desk_24_measured_12_rgb_targets_exclude_holdouts():
    training, holdout = split_indices(72)
    measured = nested_measured_indices(training)[24]
    target = select_rgb_targets(training, measured)
    assert len(training) == 57 and len(measured) == 24 and len(target) == 12
    assert len(set(target)) == 12
    assert not set(target) & set(measured)
    assert not (set(target) | set(measured)) & set(holdout)
    assert target == select_rgb_targets(training, measured)


@pytest.mark.parametrize("measured,count", [((1,2),2),((1,3),4),((1,4),2)])
def test_bad_rgb_target_selection_fails(measured,count):
    with pytest.raises(ValueError):
        select_rgb_targets((1,2,3),measured,count)


def test_measured_only_projection_and_metric_fit():
    h,w = 80,100
    y,x = np.indices((h,w))
    depth = (1.1+x*.004+y*.002).astype(np.float32)
    camera = Camera(w,h,100.,100.,49.5,39.5)
    frame = Frame(0.,np.eye(4),depth,np.zeros((h,w,3),np.uint8))
    projected = project_measured_depth([frame],camera,np.eye(4),stride=4)
    assert np.isfinite(projected).sum() > 400
    np.testing.assert_allclose(projected[::4,::4],depth[::4,::4],atol=1e-5)
    relative = ((1/depth-.09)/.2).astype(np.float32)
    metric,details = calibrate_relative_depth(relative,projected)
    valid = metric > 0
    assert valid.mean() > .8,details
    assert np.median(np.abs(metric[valid]-depth[valid])) < .02


def test_missing_sensor_anchors_rejected():
    with pytest.raises(RuntimeError,match="Too few"):
        calibrate_relative_depth(np.ones((30,40),np.float32),
            np.full((30,40),np.inf,np.float32))
