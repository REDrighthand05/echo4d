"""Known-answer contracts; synthetic numbers are not real-scene evidence."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
from echo4d.model import DataError
from desk_v2_residual_attribution import bucketize, discontinuity_proxy

def test_all_measured_pixels_conserved_and_no_error_dropped():
    measured=np.array([[.5,.5,1.5,2.5],[3.5,.5,0,.5]],np.float32)
    predicted=np.array([[.5,.6,1.8,0],[3.1,.7,0,.3]],np.float32)
    result=bucketize(predicted,measured)
    assert result["valid_pixels"]==7
    assert result["overlap_pixels"]==6
    for entries in result["partitions"].values():
        assert sum(r["valid_pixels"] for r in entries)==7
        assert sum(r["overlap_pixels"] for r in entries)==6
        assert sum(r["missing_pixels"] for r in entries)==1
        assert sum(r["squared_error_sum_m2"] for r in entries)==pytest.approx(
            result["squared_error_sum_m2"])
    assert sum(r["overlap_pixels"] for r in result["abs_error_histogram"])==6

def test_proxy_only_uses_valid_four_neighbors():
    z=np.array([[1,1,0],[1,1.25,1.25],[0,1.25,1.25]],np.float32)
    mask=discontinuity_proxy(z,(z>=.15)&(z<=4))
    assert mask[1,1] and mask[0,1] and mask[1,0]
    assert not mask[0,0] and not mask[0,2]

def test_no_overlap_does_not_turn_missing_into_pass():
    data=bucketize(np.zeros((2,2),np.float32),np.ones((2,2),np.float32))
    assert data["valid_pixels"]==4
    assert data["overlap_pixels"]==0
    assert sum(v["missing_pixels"] for v in data["partitions"]["image_location"])==4

def test_invalid_pixels_fail_closed():
    z=np.ones((2,2),np.float32)
    with pytest.raises(DataError): bucketize(np.array([[np.nan,0],[0,0]]),z)
    with pytest.raises(DataError): bucketize(z,-z)
