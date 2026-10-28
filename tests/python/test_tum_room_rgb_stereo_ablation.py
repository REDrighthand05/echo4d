"""Synthetic CPU stereo unit checks; these do not constitute a room Mesh run."""
import numpy as np
import pytest
cv2 = pytest.importorskip('cv2', reason='Optional OpenCV stereo tested by dedicated real-room workflow')
from types import SimpleNamespace

from echo4d.model import DataError
from tum_room_rgb_stereo_ablation import (
    choose_stereo_partner, infer_rectified_stereo_depth)


def test_partner_requires_real_rightward_parallax():
    poses = [np.eye(4) for _ in range(3)]
    poses[1][0,3] = .12
    poses[2][1,3] = .12
    assert choose_stereo_partner(poses,0) == 1
    assert choose_stereo_partner([np.eye(4),np.eye(4)],0) is None


def test_stereo_recovers_metric_Z_from_known_baseline_without_sensor_Z():
    h,w = 120,256
    rng = np.random.default_rng(27)
    left = cv2.GaussianBlur(rng.integers(0,256,(h,w,3),dtype=np.uint8),(3,3),.4)
    right = np.zeros_like(left)
    disparity = 12
    right[:,:-disparity] = left[:,disparity:]
    camera = SimpleNamespace(width=w,height=h,fx=120.,fy=120.,cx=w/2,cy=h/2)
    p0,p1 = np.eye(4),np.eye(4)
    p1[0,3] = .10
    depth, receipt = infer_rectified_stereo_depth(left,right,camera,p0,p1)
    assert depth.shape == (h,w)
    assert receipt['valid_original_pixels'] > 1000
    assert .7 < np.median(depth[depth>0]) < 1.3
    assert receipt['metric_scale_source'].startswith('known TUM')


def test_rejects_unusable_baseline_and_mismatched_images():
    camera = SimpleNamespace(width=256,height=120,fx=120.,fy=120.,cx=128.,cy=60.)
    image = np.zeros((120,256,3),dtype=np.uint8)
    with pytest.raises(DataError):
        infer_rectified_stereo_depth(image,image,camera,np.eye(4),np.eye(4))
    right_pose = np.eye(4)
    right_pose[0,3] = .1
    with pytest.raises(DataError):
        infer_rectified_stereo_depth(image[:100],image,camera,np.eye(4),right_pose)
