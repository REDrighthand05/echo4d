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
