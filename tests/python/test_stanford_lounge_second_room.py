import pytest
from stanford_lounge_second_room import integrate
from open3d_official_room_baseline import spaced_real_frames


def test_second_room_full_trajectory_sampling():
    ids=spaced_real_frames(3000,360)
    assert ids[0]==0 and ids[-1]==2999 and len(set(ids))==360


def test_fusion_rejects_invalid_voxel_before_reading_data():
    with pytest.raises(ValueError,match="voxel"):
        integrate(None,None,None,None,[],.2)
