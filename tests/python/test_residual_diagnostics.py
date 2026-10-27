import numpy as np
import pytest
from echo4d.residual_diagnostics import diagnose_frames, residual_map


def sample(pred, measured, index=0):
    return {'index': index, 'prediction': pred.astype(np.float32),
            'measured': measured.astype(np.float32)}


def test_perfect_surface_zero_error_and_exclusive_regions():
    m = np.full((24, 32), 1.2, np.float32)
    out = diagnose_frames([sample(m, m)])
    assert out['coverage'] == 1
    assert out['rmse_m'] == 0
    assert out['overlap_pixels'] == 24*32
    assert sum(v['valid_sensor_pixels'] for v in out['regions'].values()) == 24*32
    assert all(v['squared_error_share'] is None for v in out['regions'].values())


def test_tiny_fraction_can_dominate_global_squared_error():
    m = np.full((20, 20), 1.0, np.float32)
    p = m.copy()
    p[0, 0] = 2.0
    d = diagnose_frames([sample(p, m)])
    assert d['tails']['over_50cm']['pixels'] == 1
    assert d['tails']['over_50cm']['squared_error_share'] == pytest.approx(1)
    assert d['mae_m'] == pytest.approx(1/400)
    assert d['rmse_m'] == pytest.approx(1/20)


def test_depth_discontinuity_band_encloses_real_step_and_is_not_assumed_cause():
    m = np.full((30, 40), 1.0, np.float32)
    m[:, 20:] = 2.0
    p = m.copy()
    p[:, 19:21] += .20
    d = diagnose_frames([sample(p, m)], step_m=.1, band_pixels=2)
    assert d['regions']['depth_step_band']['overlap_pixels'] > 0
    assert d['regions']['depth_step_band']['squared_error_share'] == pytest.approx(1)
    assert d['regions']['interior']['rmse_m'] == 0


def test_missing_prediction_counts_against_coverage_not_error():
    m = np.full((10, 10), 2.0, np.float32)
    p = m.copy()
    p[:, :5] = 0
    d = diagnose_frames([sample(p, m)])
    assert d['coverage'] == .5
    assert d['overlap_pixels'] == 50
    assert d['missing_surface_pixels'] == 50
    assert d['rmse_m'] == 0


def test_invalid_sensor_and_exclusion_mask_are_not_false_failures():
    m = np.full((10, 10), 1.0, np.float32)
    p = m.copy()
    m[:, 0] = 0
    p[:, 0] = 3
    mask = np.zeros((10, 10), bool)
    mask[:, -1] = True
    p[:, -1] = 3
    d = diagnose_frames([{'index': 0,'prediction': p,'measured': m,'excluded': mask}])
    assert d['sensor_valid_pixels'] == 80
    assert d['rmse_m'] == 0


def test_weighted_not_average_of_frame_errors():
    big = sample(np.ones((20,20)), np.ones((20,20)), 0)
    small_m = np.zeros((20,20), np.float32)
    small_m[0,0] = 1
    small_p = np.zeros((20,20), np.float32)
    small_p[0,0] = 2
    d = diagnose_frames([big,sample(small_p,small_m,1)])
    assert d['overlap_pixels'] == 401
    assert d['mae_m'] == pytest.approx(1/401)
    assert d['rmse_m'] == pytest.approx(np.sqrt(1/401))


def test_observably_wrong_depth_is_detected():
    m = np.full((20,20), 1.5, np.float32)
    d = diagnose_frames([sample(m + 0.10,m)])
    assert .099 < d['rmse_m'] < .101
    assert d['bias_m'] > .099


@pytest.mark.parametrize('meas,pred',[(np.ones((3,4)),np.ones((4,3))),
                                         (np.full((3,4),np.nan),np.ones((3,4))),
                                         (np.zeros((3,4)),np.ones((3,4)))])
def test_invalid_inputs_fail(meas,pred):
    with pytest.raises(ValueError):
        diagnose_frames([{'index':0,'measured':meas,'prediction':pred}])