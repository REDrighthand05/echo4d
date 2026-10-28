"""No sensor Z can be consumed by reconstruction: diagnostic-only arithmetic tests."""
import numpy as np
import pytest
from tools.tum_plane_sweep_depth_diagnostic import score_pair, error_image


def test_new_only_wrong_surface_is_counted_separately_from_coverage():
    measured = np.array([[1., 2., 3., 0.], [1., 2., 3., 0.]], np.float32)
    old = np.array([[1., 0., 3., 0.], [1., 2., 0., 0.]], np.float32)
    new = np.array([[1., 2.5, 3., 1.], [1., 2., 3.02, 0.]], np.float32)
    s = score_pair(old, new, measured)
    assert s['sensor_valid_pixels'] == 6
    assert s['gained_sensor_covered_pixels'] == 2
    assert s['new_only_sensor_agreement']['pixels'] == 2
    assert s['new_only_sensor_agreement']['over_30cm_fraction'] == .5
    assert s['sensor_unknown_new_only_pixels'] == 1
    assert s['common_old_error']['pixels'] == s['common_new_error']['pixels'] == 4
    assert s['common_old_error']['mae_m'] == s['common_new_error']['mae_m']
    assert error_image(old, new, measured).size == (4, 2)


def test_newly_missing_surface_does_not_look_like_better_mae():
    truth = np.ones((2, 3), np.float32)
    old = truth.copy()
    new = truth.copy()
    new[0, 0] = 0
    s = score_pair(old, new, truth)
    assert s['lost_sensor_covered_pixels'] == 1
    assert s['new_all_error']['pixels'] == 5
    assert s['new_only_sensor_agreement']['mae_m'] is None


def test_invalid_inputs_rejected():
    a = np.ones((2, 2), np.float32)
    with pytest.raises(ValueError):
        score_pair(a, a, -a)
    with pytest.raises(ValueError):
        score_pair(a, a[:, :1], a)
