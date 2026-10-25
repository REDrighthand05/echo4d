"""Frozen B-depth holdout evaluator: pixel-weighted scores, no silent failed views."""
from __future__ import annotations

import numpy as np
import pytest

from echo4d.model import DataError
from tum_sensor_mesh_holdout import aggregate_scores


def _frame(name: str, n: int, status="measured"):
    return {"name":name,"status":status,"overlap_pixels":n}


def test_pixel_weighted_rmse_and_coverage():
    a=np.zeros(9,dtype=float)
    b=np.ones(1,dtype=float)*.2
    result=aggregate_scores([a,b], 12, [_frame("one",9),_frame("two",1)])
    assert result["overlap_pixels"]==10
    assert result["observed_depth_coverage"]==pytest.approx(10/12)
    assert result["mae_m"]==pytest.approx(.02)
    assert result["rmse_m"]==pytest.approx(np.sqrt(.04/10))
    assert result["p95_abs_m"]>0
    assert result["gate"]["passed"]
    assert result["threshold_accuracy_on_overlap"]["within_3cm"]==.9


def test_fail_gate_if_coverage_low():
    result=aggregate_scores([np.zeros(2)], 100, [_frame("one",2)])
    assert not result["gate"]["passed"]
    assert result["observed_depth_coverage"]==.02


def test_fail_gate_if_rmse_too_high():
    result=aggregate_scores([np.ones(90)*.4], 100, [_frame("one",90)])
    assert not result["gate"]["passed"]
    assert result["rmse_m"]==pytest.approx(.4)


def test_missing_or_zero_overlap_must_be_visible():
    with pytest.raises(DataError,match="no overlapping"):
        aggregate_scores([np.zeros(2)], 3,
                         [_frame("one",2),_frame("two",0,"no_overlap")])


def test_reject_accounting_that_drops_bad_pixels():
    with pytest.raises(DataError,match="inconsistency"):
        aggregate_scores([np.zeros(3)], 5, [_frame("one",2)])


def test_invalid_holdout_sized_to_zero_rejected():
    with pytest.raises(DataError,match="No withheld"):
        aggregate_scores([], 0, [])


def _frozen_fixture():
    """Actual #15 48/12 NAME split, with mocked metric metadata only."""
    from tum_sensor_metric_mesh import split_names
    source = [{"name": f"frame_{i:05d}.png", "t_tum_rgb_s": float(i)} for i in range(90)]
    train, withheld = split_names(source)
    split = {"source_kind": "measured_registered_z_depth",
             "depth_numeric_units_per_meter": 5000,
             "train": [{"name": n, "depth": f"depth/{i:05d}.png"}
                       for i, n in enumerate(train)],
             "holdout": withheld}
    metric = {"version": "echo-sfm-metric-scale-1", "units": "meters",
              "metric_scale_estimated_from_measured_depth": True,
              "scale_is_from_external_gt_poses": False,
              "scale_m_per_sfm_unit": 0.4,
              "anchor_diagnostics": [{"name": f"frame_{i:05d}.png"}
                                     for i in (15, 45, 75)],
              "frames": [{"name": item["name"]} for item in source]}
    return source, split, metric


def test_frozen_exact_split_and_sensor_scale_accepted():
    from tum_sensor_mesh_holdout import validate_frozen_holdout_contract
    source, split, metric = _frozen_fixture()
    assert validate_frozen_holdout_contract(split, metric, source) == split["holdout"]


@pytest.mark.parametrize("mutation", [
    "wrong_withheld_name", "training_row_swapped", "training_duplicate_depth",
    "measurement_as_prediction", "depth_units_mismatch",
    "gt_driven_scale", "no_measured_scale", "invalid_scale",
    "anchor_in_holdout", "duplicate_source", "missing_metric_pose",
])
def test_frozen_contract_rejects_unsafe_sources_and_leakage(mutation):
    from tum_sensor_mesh_holdout import validate_frozen_holdout_contract
    source, split, metric = _frozen_fixture()
    if mutation == "wrong_withheld_name":
        split["holdout"][0] = "frame_00061.png"
    elif mutation == "training_row_swapped":
        split["train"][0], split["train"][1] = split["train"][1], split["train"][0]
    elif mutation == "training_duplicate_depth":
        split["train"][1]["depth"] = split["train"][0]["depth"]
    elif mutation == "measurement_as_prediction":
        split["source_kind"] = "model_predicted_z"
    elif mutation == "depth_units_mismatch":
        split["depth_numeric_units_per_meter"] = 1
    elif mutation == "gt_driven_scale":
        metric["scale_is_from_external_gt_poses"] = True
    elif mutation == "no_measured_scale":
        metric["metric_scale_estimated_from_measured_depth"] = False
    elif mutation == "invalid_scale":
        metric["scale_m_per_sfm_unit"] = float("nan")
    elif mutation == "anchor_in_holdout":
        metric["anchor_diagnostics"][0]["name"] = split["holdout"][0]
    elif mutation == "duplicate_source":
        source[-1]["name"] = source[-2]["name"]
    elif mutation == "missing_metric_pose":
        metric["frames"] = metric["frames"][:-80]
    with pytest.raises(DataError):
        validate_frozen_holdout_contract(split, metric, source)
