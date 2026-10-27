"""Known-answer partition/conservation tests; no real residual inspection."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from desk_residual_diagnostic import edge_partition, merge_summaries, partitions


def protocol():
    return json.loads((Path(__file__).resolve().parents[2]/
                      'docs/experiments/desk-v2-residual-protocol.json').read_text())


def test_partitions_conserve_pixels_missing_and_squared_error():
    measured=np.array([[0.,.15,.49],[.5,1.,2.],[3.,4.,4.1]],dtype=np.float32)
    predicted=measured.copy();predicted[1,1]+=.2;predicted[2,1]=0.
    result=partitions(measured,predicted,protocol())
    assert result['total']['valid_pixels']==7
    assert result['total']['overlap_pixels']==6
    assert result['total']['missing_surface_pixels']==1
    assert result['total']['squared_error_m2']==pytest.approx(.04,abs=1e-7)
    for dimension in ('depth','region','edge_proxy'):
        assert sum(b['valid_pixels'] for b in result[dimension])==7
        assert sum(b['missing_surface_pixels'] for b in result[dimension])==1
        assert sum(b['squared_error_m2'] for b in result[dimension])==pytest.approx(.04,abs=1e-7)
    assert sum(b['overlap_pixels'] for b in result['signed_residual'])==6
    assert result['depth'][-1]['valid_pixels']==2  # 3.0 and final closed 4.0


def test_edge_proxy_has_no_wrap_and_valid_jump_precedes_hole():
    flat=np.ones((3,3),dtype=np.float32)
    assert not edge_partition(flat,.15,4.,.1).any()  # image edges are not holes
    flat[1,1]=0.
    labels=edge_partition(flat,.15,4.,.1)
    assert labels[0,1]==1 and labels[0,0]==0
    flat[0,0]=2.
    labels=edge_partition(flat,.15,4.,.1)
    assert labels[0,1]==2  # both adjacent hole and valid discontinuity


def test_aggregation_is_pixel_weighted_not_mean_of_rmse():
    p=protocol()
    a=partitions(np.ones((2,2)),np.ones((2,2))+.1,p)['total']
    b=partitions(np.ones((4,4)),np.ones((4,4))+.3,p)['total']
    merged=merge_summaries([a,b])
    assert merged['rmse_m']==pytest.approx(np.sqrt((4*.01+16*.09)/20))
    assert merged['mae_m']==pytest.approx(.26)


def test_empty_bins_have_no_fake_zero_rmse():
    result=partitions(np.ones((3,3)),np.ones((3,3)),protocol())
    empty=result['depth'][0]
    assert empty['valid_pixels']==0 and empty['rmse_m'] is None
    assert result['total']['rmse_m']==0.
