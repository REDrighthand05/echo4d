"""CPU synthetic stereo metric-geometry regressions for the opt-in OpenCV trial."""
import numpy as np
import pytest

pytest.importorskip('cv2', reason='OpenCV installed by dedicated real-room SGBM workflow')
pytest.importorskip('trimesh', reason='trimesh installed by dedicated real-room SGBM workflow')
from tum_room_opencv_sgbm_trial import camera_to_world, candidates, stereo_pair, mesh_from_pair


def test_synthetic_known_metric_plane():
    rng = np.random.default_rng(7)
    left = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-16] = left[:, 16:]
    K = np.array([[160., 0, 159.5], [0, 160., 119.5], [0, 0, 1.]])
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = .2
    pair, reason = stereo_pair(left, right, a, b, K)
    assert reason is None
    depths = pair['xyz'][:, :, 2][pair['valid']]
    assert len(depths) > 5000
    assert abs(np.median(depths) - 2) < .15
    mesh = mesh_from_pair(pair, a, stride=3)
    assert mesh is not None and len(mesh.faces) > 1000


def test_neighbor_selection_excludes_pure_rotation_and_no_baseline():
    poses = [np.eye(4) for _ in range(3)]
    poses[1] = poses[1].copy()
    poses[1][0, 3] = .2
    poses[2] = poses[2].copy()
    poses[2][2, 3] = .2
    assert candidates(poses, 0) == [1]


def test_recording_pose_column_major():
    p = np.eye(4)
    p[0, 3] = .4
    p[2, 3] = 1.2
    assert np.allclose(camera_to_world({'cameraToWorld': p.T.reshape(-1).tolist()}), p)
