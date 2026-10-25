"""No posthoc GT scale fitting: numeric SE(3) world gauge and frozen A/B gates."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from echo4d.model import DataError, rigid_matrix
from tum_fair_ab import (align_gt_to_metric_estimated_world, score_gate,
                         GT_ANCHOR_NAME)


def _pose(R, t):
    p=np.eye(4)
    p[:3,:3]=R
    p[:3,3]=t
    return p.flatten(order="F").tolist()


def _fixture():
    names=[f"frame_{i:05d}.png" for i in range(60)]
    train=[n for n in names if n not in ["frame_00004.png","frame_00005.png","frame_00009.png",
      "frame_00010.png","frame_00014.png","frame_00019.png","frame_00024.png",
      "frame_00025.png","frame_00029.png","frame_00034.png","frame_00035.png",
      "frame_00055.png"]]
    hold=[n for n in names if n not in train]
    assert len(train)==48 and len(hold)==12
    Rw=Rotation.from_euler("xyz", [20,-10,15], degrees=True).as_matrix()
    shift=np.array([2.,3.,-4.])
    gt={}
    b=[]
    for i,n in enumerate(names):
        local_t=np.array([i*.02, i*.01, .2])
        gt[n]=_pose(np.eye(3),local_t)
        b.append({"name":n,"camera_to_metric_sfm_world":_pose(Rw, Rw@local_t+shift)})
    return gt,b,train,hold,Rw,shift


def test_one_train_anchor_rigid_world_alignment_preserves_metric_scale():
    gt,b,tr,ho,Rw,shift=_fixture()
    a,Q=align_gt_to_metric_estimated_world(gt,b,tr,ho)
    assert len(a)==60
    q=rigid_matrix(Q)
    assert np.linalg.det(q[:3,:3])==pytest.approx(1,abs=1e-10)
    assert q[:3,:3]==pytest.approx(Rw,abs=1e-10)
    assert q[:3,3]==pytest.approx(shift,abs=1e-10)
    assert rigid_matrix(a[GT_ANCHOR_NAME])==pytest.approx(
        rigid_matrix(next(x for x in b if x["name"]==GT_ANCHOR_NAME)["camera_to_metric_sfm_world"]))
    delta=np.linalg.norm(rigid_matrix(a[ho[0]])[:3,3]-rigid_matrix(a[ho[1]])[:3,3])
    original=np.linalg.norm(rigid_matrix(gt[ho[0]])[:3,3]-rigid_matrix(gt[ho[1]])[:3,3])
    assert delta==pytest.approx(original,abs=1e-10)


def test_missing_gt_or_wrong_anchor_fails_closed():
    gt,b,tr,ho,_,_=_fixture()
    del gt[ho[0]]
    with pytest.raises(DataError,match="lacks GT"):
        align_gt_to_metric_estimated_world(gt,b,tr,ho)
    gt,b,tr,ho,_,_=_fixture()
    with pytest.raises(DataError,match="training view"):
        align_gt_to_metric_estimated_world(gt,b,tr,ho,anchor=ho[0])


def test_fair_ab_gate_needs_both_coverage_and_depth():
    assert score_gate({"coverage":.85,"rmse_m":.14})["passes"]
    assert not score_gate({"coverage":.79,"rmse_m":.01})["passes"]
    assert not score_gate({"coverage":.96,"rmse_m":.16})["passes"]
