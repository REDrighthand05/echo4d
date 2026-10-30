"""Synthetic calibration checks; real scene output requires GitHub Actions."""
import numpy as np

from echo4d.model import Camera, Frame
from tum_desk_native640_tsdf import even_pixel_frame, recorded_poses


def test_even_pixel_geometry_matches_native_camera_rays():
    camera = Camera(640, 480, 517.3, 516.5, 318.6, 255.3)
    yy, xx = np.indices((480, 640))
    depth = (1 + xx / 10000 + yy / 100000).astype(np.float32)
    rgb = np.stack((xx % 256, yy % 256, (xx + yy) % 256), -1).astype(np.uint8)
    frame = Frame(1.0, np.eye(4), depth, rgb)
    half = even_pixel_frame(frame, camera)
    small = camera.scaled(2)
    assert half.depth.shape == (240, 320)
    assert half.color.shape == (240, 320, 3)
    np.testing.assert_array_equal(half.depth, depth[::2, ::2])
    for u, v in ((0, 0), (100, 60), (319, 239)):
        np.testing.assert_allclose([(u-small.cx)/small.fx, (v-small.cy)/small.fy],
                                   [(2*u-camera.cx)/camera.fx, (2*v-camera.cy)/camera.fy])


def test_exclusion_mask_preserved():
    camera = Camera(4, 4, 2, 2, 1, 1)
    excluded = np.zeros((4, 4), bool)
    excluded[2, 2] = True
    frame = Frame(1., np.eye(4), np.ones((4, 4), np.float32),
                  np.zeros((4, 4, 3), np.uint8), excluded)
    small = even_pixel_frame(frame, camera)
    assert small.excluded.shape == (2, 2)
    assert small.excluded[1, 1]
    assert small.excluded.sum() == 1


def test_fixed_poses_use_column_major_convention():
    pose = np.eye(4)
    pose[0, 3] = 1.25
    entry = {"cameraToWorld": pose.reshape(-1, order="F").tolist()}
    poses = recorded_poses([entry] * 54)
    assert len(poses) == 5
    np.testing.assert_array_equal(poses[11], pose)
