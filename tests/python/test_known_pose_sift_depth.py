"""CPU regressions for SIFT known-pose RGB depth correction."""
import numpy as np
import pytest
pytest.importorskip("cv2", reason="OpenCV is installed in the real-room workflow")
from known_pose_sift_depth import correct_frame, correct_sequence


def test_no_anchors_does_not_invent_geometry():
    depth = np.ones((48, 64), np.float32)
    out, report = correct_frame(depth, np.empty((0, 3)))
    assert np.array_equal(out, depth)
    assert not report["applied"]


def test_local_correction_preserves_unknown_depth_and_is_bounded():
    h, w = 120, 160
    depth = np.full((h, w), 2.15, np.float32)
    depth[:10] = 0
    anchors = np.array([(x, y, 2.) for y in range(10, 110, 8)
                        for x in range(10, 150, 8)])
    out, report = correct_frame(depth, anchors)
    assert report["applied"]
    assert np.all(out[:10] == 0)
    assert np.max(np.abs(out-depth)) <= .22001
    assert np.mean(np.abs(out[depth > 0]-2.)) < .15


def test_invalid_depth_fails_explicitly():
    with pytest.raises(ValueError):
        correct_frame(np.array([[-1.]], np.float32), [])


def test_input_count_mismatch():
    with pytest.raises(ValueError):
        correct_sequence([np.ones((8, 8))], [], [], np.eye(3))
