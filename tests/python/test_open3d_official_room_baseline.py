"""Real Redwood frame-to-trajectory matching regression (no invented pose)."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pytest
from open3d_official_room_baseline import spaced_real_frames,read_poses,trajectory_frame_ids


def test_real_frame_selection_covers_start_middle_end():
    out=spaced_real_frames(21930,360)
    assert out[0]==0 and out[-1]==21929
    assert len(out)==len(set(out))==360
    assert out==sorted(out)


@pytest.mark.parametrize("total,count",[(5,6),(2,2),(601,601),(10,2)])
def test_invalid_frame_budget(total,count):
    with pytest.raises(ValueError):
        spaced_real_frames(total,count)


@dataclass
class Intr:
    width:int=640
    height:int=480
    intrinsic_matrix:object=None


def test_redwood_bedroom_image_1_maps_to_logged_pose_1_not_0(tmp_path):
    """Official C++ dataset indexes JPGs from 1, not 0."""
    x=np.eye(4)
    class Param:
        def __init__(self,index):
            self.extrinsic=x.copy()
            self.extrinsic[0,3]=float(index)
            self.intrinsic=Intr(intrinsic_matrix=np.array(
                [[525,0,319.5],[0,525,239.5],[0,0,1.]]))
    class T:
        parameters=[Param(0),Param(1),Param(2)]
    log=tmp_path/"bedroom.log"
    log.write_text("".join(f"{i} {i} {i+1}\n"+("1 0 0 0\n"*4) for i in range(3)))
    assert trajectory_frame_ids(log,3)==[0,1,2]
    k, poses=read_poses(T(),["image/000001.jpg","image/000002.jpg"],log)
    assert k==(640,480,525.,525.,319.5,239.5)
    assert [p[0,3] for p in poses]==[1.,2.]
    with pytest.raises(ValueError,match="missing"):
        read_poses(T(),["image/000003.jpg"],log)


def test_redwood_sample_zero_based_mapping_and_invalid_pose(tmp_path):
    x=np.eye(4)
    class P:
        intrinsic=Intr(intrinsic_matrix=np.array(
            [[525,0,319.5],[0,525,239.5],[0,0,1.]]))
        extrinsic=x
    class T:
        parameters=[P()]
    log=tmp_path/"sample.log"
    log.write_text("0 0 1\n"+("1 0 0 0\n"*4))
    assert read_poses(T(),["color/00000.jpg"],log)[1][0][0,3]==0
    bad=x.copy()
    bad[0,0]=2
    P.extrinsic=bad
    with pytest.raises(ValueError,match="extrinsic"):
        read_poses(T(),["color/00000.jpg"],log)
