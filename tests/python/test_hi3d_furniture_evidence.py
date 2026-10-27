"""Tests for true source evidence, fixed split and fail-closed object scoring."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from hi3d_furniture_evidence import inventory, validate_ledger, raw_archive_checks
from hi3d_furniture_compare import numeric_score, dimensions, evaluate
from echo4d.model import DataError

LEDGER = Path(__file__).resolve().parents[2] / "docs/evidence/hi3d-furniture-d/xyz-frozen-source-ledger.json"


def source():
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def test_real_frozen_ledger_is_60_distinct_original_sensor_pairs():
    ledger = source()
    rows = validate_ledger(ledger)
    assert len(rows) == 60 and len(set(f["name"] for f in rows)) == 60
    hold = [f for f in rows if f["role"] == "holdout_no_model_access"]
    train = [f for f in rows if f["role"] == "train_input"]
    assert len(train) == 48 and len(hold) == 12
    assert set(f["raw_depth_path"] for f in train).isdisjoint(
        set(f["raw_depth_path"] for f in hold))
    assert all(f["raw_depth_sha256"] is not None for f in hold)
    assert all(f["processed_train_depth_sha256"] is not None for f in train)
    report = inventory(LEDGER)
    assert report["status"] == "BLOCKED_NO_VERIFIED_SINGLE_FURNITURE"
    assert report["independent_furniture_dimensions_m"] is None
    assert report["real_hi3d_gpu_inference"] is False


def test_holdout_swap_is_denied():
    ledger = source()
    ledger["preregistered_protocol"]["holdout_names"][0] = "frame_00000.png"
    with pytest.raises(DataError, match="preregistration"):
        validate_ledger(ledger)


def test_appearing_to_have_object_mask_is_not_verified():
    ledger = source()
    ledger["frames"][4]["object_mask_status"] = "claimed"
    with pytest.raises(DataError, match="not verified object"):
        validate_ledger(ledger)


def test_fake_metric_observation_hash_rejected():
    ledger = source()
    ledger["frames"][4]["raw_depth_sha256"] = "NOT_A_SHA"
    with pytest.raises(DataError, match="hash"):
        validate_ledger(ledger)


def test_original_source_time_is_not_resampled_video_time():
    ledger = source()
    ledger["frames"][4]["original_rgb_t_s"] = 1.0
    with pytest.raises(DataError, match="clock"):
        validate_ledger(ledger)


def test_train_depth_overlap_fails_original_roles():
    ledger = source()
    ledger["frames"][0]["role"] = "holdout_no_model_access"
    with pytest.raises(DataError, match="not verified object"):
        validate_ledger(ledger)


def test_archive_checksum_is_required_before_parsing(tmp_path):
    bad = tmp_path / "bad.tgz"
    bad.write_bytes(b"fake archive")
    with pytest.raises(DataError, match="archive"):
        raw_archive_checks(LEDGER, bad)


def test_missing_object_proof_prevents_real_score(tmp_path):
    (tmp_path / "proof.json").write_text(json.dumps({"version": "fake"}))
    with pytest.raises(DataError, match="Unreviewed object"):
        evaluate(LEDGER, tmp_path / "proof.json", tmp_path)


def test_exact_overlap_perfect_metre_depth():
    measured = np.ones((6, 8), np.float32)
    predicted = measured.copy()
    mask = np.ones((6, 8), bool)
    result, residual = numeric_score(predicted, measured, mask)
    assert result["coverage"] == 1.0
    assert result["rmse_m"] == 0.0 and len(residual) == 48


def test_same_mask_one_wrong_surface_gets_penalized():
    measured = np.ones((6, 8), np.float32)
    mask = np.ones((6, 8), bool)
    bad = np.full_like(measured, 1.1)
    score, _ = numeric_score(bad, measured, mask)
    assert score["rmse_m"] == pytest.approx(.1, abs=1e-6)


def test_missing_geometry_is_not_mistaken_for_accuracy():
    measured = np.ones((6, 8), np.float32)
    predicted = np.zeros_like(measured)
    predicted[:, :4] = 1.0
    mask = np.ones((6, 8), bool)
    score, _ = numeric_score(predicted, measured, mask)
    assert score["coverage"] == 0.5
    assert score["missing_pixels"] == 24
    assert score["rmse_m"] == 0.0  # conditional RMSE, not evidence of full shape


def test_totally_uncovered_object_never_yields_fake_rmse_zero():
    measured = np.ones((6, 8), np.float32)
    score, errors = numeric_score(np.zeros_like(measured), measured, np.ones_like(measured, bool))
    assert score["rmse_m"] is None
    assert score["coverage"] == 0 and errors.size == 0


@pytest.mark.parametrize("bad", [np.diag([-1, 1, 1]), np.diag([1, 2, 1]), np.zeros((3, 3))])
def test_invalid_object_axes_cannot_fake_dimension_estimate(bad):
    with pytest.raises(DataError, match="right-handed"):
        dimensions(np.array([[0.,0.,0.],[1.,1.,1.]]), bad)


def test_known_extent_m_correct_and_not_rescaled():
    vertices = np.array([[0., 0., 0.], [1., 2., 3.]])
    assert np.allclose(dimensions(vertices, np.eye(3)), [1., 2., 3.])


def test_nonfinite_sensor_depth_rejected():
    measured = np.ones((6, 8), np.float32)
    measured[0, 0] = np.nan
    with pytest.raises(DataError, match="Invalid aligned"):
        numeric_score(np.ones_like(measured), measured, np.ones_like(measured, bool))
