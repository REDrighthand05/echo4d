"""Synthetic stereo oracle and counterexamples; not real-scene geometry evidence."""
import numpy as np
import pytest
cv2 = pytest.importorskip('cv2')
from echo4d.model import Camera, DataError
from echo4d.known_pose_stereo import infer_stereo, rectify_geometry


def setup():
    c=Camera(320,240,240.,240.,159.5,119.5)
    a=np.eye(4);b=np.eye(4);b[0,3]=.1
    return c,a,b


def test_plane_known_metric_depth_and_pose():
    camera,a,b=setup()
    rng=np.random.default_rng(31)
    left=rng.integers(0,256,(240,320,3),dtype=np.uint8)
    right=np.zeros_like(left);right[:,:-12]=left[:,12:]
    rgb,depth,new_camera,pose,stats=infer_stereo(left,right,camera,a,b)
    assert stats['measured_depth_used'] is False
    assert stats['valid_depth_pixels']>20000
    assert abs(np.median(depth[depth>0])-2.0)<.07
    assert rgb.shape==left.shape and new_camera.width==320
    assert np.allclose(pose[:3,3],a[:3,3])


def test_textureless_rgb_does_not_invent_dense_depth():
    camera,a,b=setup()
    img=np.full((240,320,3),128,np.uint8)
    _,depth,_,_,stats=infer_stereo(img,img,camera,a,b)
    assert stats['valid_depth_pixels']<100
    assert np.count_nonzero(depth)<100


def test_reverse_baseline_rejected_not_negative_depth():
    camera,a,b=setup()
    with pytest.raises(DataError):
        rectify_geometry(camera,b,a)


def test_wrong_resolution_and_invalid_disparity_rejected():
    camera,a,b=setup()
    rgb=np.zeros((240,320,3),np.uint8)
    with pytest.raises(DataError):
        infer_stereo(rgb[:200],rgb,camera,a,b)
    with pytest.raises(DataError):
        infer_stereo(rgb,rgb,camera,a,b,num_disparities=63)
