"""Deterministic selection and sensor-only metric calibration regressions."""
import numpy as np
from lounge_sparse_rgb_real import diverse_add
from lounge_rgb_depth_adapter import metric_from_relative


def test_pose_spread_is_unique_deterministic_and_uses_no_hidden_depth():
    poses = []
    for i in range(18):
        p = np.eye(4)
        p[:3, 3] = [float(i % 6) * .25, float(i // 6) * .4, 0.]
        poses.append(p)
    candidates = list(range(18))
    a = diverse_add(candidates, poses, 7, seed=[0, 8, 17])
    b = diverse_add(candidates, poses, 7, seed=[0, 8, 17])
    assert a == b and len(a) == 7 and len(set(a)) == 7
    assert not ({0, 8, 17} & set(a))
    remaining = sorted(set(candidates) - set(a) - {0, 8, 17})
    rgb = diverse_add(remaining, poses, 4, seed=a+[0, 8, 17])
    assert len(rgb) == 4 and not set(rgb) & set(a)


def test_inverse_depth_scale_from_retained_sensor_projection():
    yy, xx = np.mgrid[:40, :40]
    depth = (1.0 + .006 * xx + .004 * yy).astype(np.float32)
    relative = (1.7 / depth + .2).astype(np.float32)
    metric, stats = metric_from_relative(relative, depth)
    assert stats["sensor_projection_pixels"] == 1600
    assert stats["predicted_valid_fraction"] > .95
    assert np.median(np.abs(metric[metric > 0] - depth[metric > 0])) < .005
