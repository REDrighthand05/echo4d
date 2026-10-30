"""Deterministic geometric tests; no pretrained model or dataset download."""
import numpy as np
import cv2
import pytest
from echo4d.model import Camera
from tum_desk_rgb_mvs import plane_sweep_depth


def test_metric_multiview_plane_recovered_without_sensor_depth():
    rng=np.random.default_rng(31)
    h,w=96,128
    fx=140.
    z=1.8
    cam=Camera(w,h,fx,fx,(w-1)/2,(h-1)/2)
    ref=cv2.GaussianBlur(rng.integers(0,256,(h,w,3),dtype=np.uint8),
                        (3,3),.6)
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    neighbours=[]
    poses=[]
    for baseline in (.08,-.12,.16,-.18):
        # A translated camera sees the plane shifted in the opposite direction.
        src=cv2.remap(ref,xx+fx*baseline/z,yy,cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)
        pose=np.eye(4)
        pose[0,3]=baseline
        neighbours.append(src)
        poses.append(pose)
    depth,receipt=plane_sweep_depth(ref,np.eye(4),neighbours,poses,cam,
                                    planes=88)
    valid=depth>0
    assert valid.mean()>.10,receipt
    assert np.median(np.abs(depth[valid]-z))<.18,receipt
    assert receipt["scale_source"]=="TUM metric reference camera translations"


def test_textureless_images_do_not_fabricate_depth():
    cam=Camera(80,64,95.,95.,39.5,31.5)
    rgb=np.full((64,80,3),120,np.uint8)
    poses=[]
    for shift in (.1,-.1):
        p=np.eye(4)
        p[0,3]=shift
        poses.append(p)
    depth,receipt=plane_sweep_depth(rgb,np.eye(4),[rgb,rgb],poses,cam,
                                    planes=20)
    assert not np.any(depth)
    assert receipt["valid_pixels"]==0


def test_reject_single_neighbour_and_mismatched_image():
    cam=Camera(80,64,95.,95.,39.5,31.5)
    rgb=np.zeros((64,80,3),np.uint8)
    with pytest.raises(ValueError,match="At least two"):
        plane_sweep_depth(rgb,np.eye(4),[rgb],[np.eye(4)],cam)
    with pytest.raises(ValueError,match="size differs"):
        plane_sweep_depth(rgb,np.eye(4),[rgb,rgb[:30]],
                          [np.eye(4),np.eye(4)],cam,planes=20)
