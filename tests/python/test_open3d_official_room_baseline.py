from dataclasses import dataclass
import numpy as np
import pytest
from open3d_official_room_baseline import spaced_real_frames,read_poses


def test_real_frame_selection_covers_start_middle_end():
    out=spaced_real_frames(21931,360)
    assert out[0]==0 and out[-1]==21930
    assert len(out)==len(set(out))==360
    assert out==sorted(out)


@pytest.mark.parametrize("total,count",[(5,6),(2,2),(601,601),(10,2)])
def test_invalid_frame_budget(total,count):
    with pytest.raises(ValueError):
        spaced_real_frames(total,count)


def test_camera_calibration_and_pose_validation():
    @dataclass
    class K:
        width:int=640
        height:int=480
        intrinsic_matrix:object=None
    x=np.eye(4)
    x[0,3]=1.2
    class P:
        intrinsic=K(intrinsic_matrix=np.array([[525,0,319.5],[0,525,239.5],[0,0,1.]]))
        extrinsic=x
    class T:
        parameters=[P()]
    k, poses=read_poses(T(),1)
    assert k==(640,480,525.,525.,319.5,239.5)
    assert poses[0][0,3]==1.2
    with pytest.raises(ValueError,match="count"):
        read_poses(T(),2)
    bad=x.copy()
    bad[0,0]=2
    P.extrinsic=bad
    with pytest.raises(ValueError,match="extrinsic"):
        read_poses(T(),1)
