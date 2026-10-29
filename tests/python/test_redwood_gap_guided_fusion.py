import numpy as np
import pytest
from redwood_gap_guided_fusion import surface_cells,choose_gap_views


def test_measured_cells_unique_and_signed():
    x=np.array([[0,0,1],[.001,0,1],[.12,0,1],[-.12,0,1]],float)
    assert len(surface_cells(x))==3
    assert len(surface_cells(np.array([[-.08,0,0],[.08,0,0],[0,-.08,0],[0,0,-.08]])))==4


@pytest.mark.parametrize("x",[[[np.nan,0,1]],[[0,1]]])
def test_invalid_sensor_coordinates_rejected(x):
    with pytest.raises(ValueError):
        surface_cells(np.asarray(x))


def test_select_distinct_gap_geometry_not_repeated_frame():
    source={10:{1,2,3},20:{1,2,3},30:{4,5},40:set()}
    assert choose_gap_views(source,3)==([10,30],[3,2])
    assert source[20]=={1,2,3}


def test_empty_gap_has_no_invented_selection():
    assert choose_gap_views({2:set(),4:set()},2)==([],[])
