"""Pure guard tests; real 640x480 geometry runs in GitHub Actions."""
import numpy as np
import pytest
from stanford_gap_guided_depth import candidate_indices, select_views, depth_gap_cells


def test_candidates_exclude_frozen_and_nearby_frames():
    assert candidate_indices(30,[0,10,20,29],stride=2)==[4,6,14,16,24,26]


def test_greedy_unique_missing_surface_cells_not_raw_frame_count():
    frames={2:{1,2,3,4}, 5:{1,2,3,4,5}, 8:{6,7,8}, 11:{8,9}}
    selected,gains=select_views(frames,3)
    assert selected==[5,8]
    assert gains==[5,3]


def test_reject_invalid_budget_and_calibration_before_open3d():
    with pytest.raises(ValueError,match="budget"):
        select_views({},241)
    with pytest.raises(ValueError,match="stride"):
        candidate_indices(30,[0,29],0)
    with pytest.raises(ValueError,match="640x480"):
        depth_gap_cells(None,None,None,np.eye(4),(160,120,100,100,80,60))
