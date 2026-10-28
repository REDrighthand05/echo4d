"""CPU synthetic geometry and negative controls; real room runs in Actions."""
import cv2
import numpy as np
import pytest

from echo4d.model import DataError
from tum_room_multiview_zncc import sweep_zncc


def synthetic_plane(z=1.3, brightness=0):
    h, w = 120, 200
    rng = np.random.default_rng(31)
    ref = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
                           (3, 3), .5)
    K = np.array([[130., 0, w/2], [0, 130., h/2], [0, 0, 1.]])
    poses, sources = [], []
    for dx in (.08, -.10, .16):
        pose = np.eye(4)
        pose[0, 3] = dx
        H = K @ np.array([[1., 0, -dx/z], [0, 1., 0], [0, 0, 1.]]) @ np.linalg.inv(K)
        source = cv2.warpPerspective(ref, H, (w, h))
        if brightness:
            source = np.clip(source.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
        sources.append(source)
        poses.append(pose)
    return ref, sources, K, poses


def test_metric_plane_and_two_camera_support():
    ref, sources, K, poses = synthetic_plane()
    depth, receipt = sweep_zncc(ref, sources, K, np.eye(4), poses)
    valid = depth > 0
    assert receipt["source_cameras"] == 3
    assert valid.sum() > 10000
    assert abs(float(np.median(depth[valid])) - 1.3) < .15
    assert float(np.mean(np.abs(depth[valid] - 1.3) < .2)) > .95


def test_brightness_shift_still_recovers_metric_depth():
    ref, sources, K, poses = synthetic_plane(brightness=20)
    depth, _ = sweep_zncc(ref, sources, K, np.eye(4), poses)
    valid = depth > 0
    assert valid.sum() > 6000
    assert float(np.mean(np.abs(depth[valid] - 1.3) < .2)) > .9


def test_flat_rgb_is_not_invented():
    ref, sources, K, poses = synthetic_plane()
    ref[:] = 128
    sources = [ref.copy() for _ in sources]
    depth, receipt = sweep_zncc(ref, sources, K, np.eye(4), poses)
    assert not depth.any()
    assert receipt["valid_pixels"] == 0


def test_duplicate_camera_center_is_not_a_second_witness():
    ref, sources, K, poses = synthetic_plane()
    with pytest.raises(DataError, match="distinct camera positions"):
        sweep_zncc(ref, [sources[0], sources[0]], K,
                   np.eye(4), [poses[0], poses[0].copy()])
    with pytest.raises(DataError, match="distinct camera positions"):
        sweep_zncc(ref, [ref, sources[0]], K,
                   np.eye(4), [np.eye(4), poses[0]])


def test_invalid_depth_grid_and_calibration_rejected():
    ref, sources, K, poses = synthetic_plane()
    with pytest.raises(DataError, match="ordered metric depths"):
        sweep_zncc(ref, sources, K, np.eye(4), poses,
                   depths=np.array([1.0, .5, 1.5]))
    with pytest.raises(DataError, match="calibrated RGB"):
        sweep_zncc(ref, sources, np.zeros((3, 3)), np.eye(4), poses)
