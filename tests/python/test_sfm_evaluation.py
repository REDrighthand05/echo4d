"""Known-answer and adversarial checks; no synthetic result counts as real SfM."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from echo4d.model import DataError
from echo4d.sfm_evaluation import (sim3_align, score_pairs, evaluate_tum,
                                    interpolate_gt, read_tum_groundtruth)


def poses_and_aligned_ground_truth():
    points = np.array([[0., 0., 0.], [1, 0, 0], [0, 2, 0], [0, 0, 3],
                       [2, 1, 1], [-1, 1, 2]])
    global_r = Rotation.from_euler("xyz", [15, -25, 30], degrees=True).as_matrix()
    global_scale = 2.5
    global_t = np.array([4., -3, 2])
    est, gt = [], []
    for i, point in enumerate(points):
        local_r = Rotation.from_euler("xyz", [2*i, i*3, -i], degrees=True).as_matrix()
        a, b = np.eye(4), np.eye(4)
        a[:3, 3] = point
        a[:3, :3] = local_r
        b[:3, 3] = global_scale * global_r @ point + global_t
        b[:3, :3] = global_r @ local_r
        est.append(a)
        gt.append(b)
    return est, gt


def test_sim3_recovers_known_scale_and_rotation():
    est, gt = poses_and_aligned_ground_truth()
    measured = score_pairs(est, gt)
    assert measured["pairs"] == 6
    assert measured["ate_rmse_m_after_sim3"] < 1e-11
    assert measured["rotation_mean_deg_after_sim3"] < 1e-8
    assert measured["alignment"]["scale_for_evaluation_only"] == pytest.approx(2.5)
    assert measured["metric_scale_estimated_by_sfm"] is False


def test_known_position_error_does_not_get_hidden():
    est, gt = poses_and_aligned_ground_truth()
    gt[2][:3, 3] += [0.4, 0, 0]
    measured = score_pairs(est, gt)
    assert measured["ate_rmse_m_after_sim3"] > 0.05


def test_orientation_error_reported_independently():
    est, gt = poses_and_aligned_ground_truth()
    gt[1][:3, :3] = gt[1][:3, :3] @ Rotation.from_euler("z", 30, degrees=True).as_matrix()
    measured = score_pairs(est, gt)
    assert measured["ate_rmse_m_after_sim3"] < 1e-10
    assert measured["rotation_mean_deg_after_sim3"] > 4


@pytest.mark.parametrize("bad", ["collinear", "stationary", "nan"])
def test_degenerate_trajectory_rejected(bad):
    x = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    if bad == "stationary":
        x[:] = 0
    if bad == "nan":
        x[0, 0] = np.nan
    with pytest.raises(DataError):
        sim3_align(x, x)


def test_reflection_is_not_accepted_as_rotation():
    pose, gt = poses_and_aligned_ground_truth()
    pose[0][:3, :3] = np.diag([-1, 1, 1])
    with pytest.raises(DataError, match="right-handed"):
        score_pairs(pose, gt)


def test_gt_interpolation_no_extrapolation_and_correct_intermediate_rotation():
    times = np.array([10., 10.1, 10.2])
    poses = np.stack([np.eye(4) for _ in range(3)])
    poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    poses[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    result = interpolate_gt(10.05, times, poses)
    assert result[:3, 3] == pytest.approx([.5, 0, 0])
    assert Rotation.from_matrix(result[:3, :3]).magnitude() == pytest.approx(np.pi / 4)
    assert interpolate_gt(10.3, times, poses) is None
    assert interpolate_gt(9.9, times, poses) is None


def test_tum_parser_rejects_invalid_quaternion(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("0 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 0\n2 0 0 0 0 0 0 1\n")
    with pytest.raises(DataError, match="quaternion"):
        read_tum_groundtruth(path)


def test_unmapped_registered_image_rejected():
    est, gt = poses_and_aligned_ground_truth()
    entries = [{"name": str(i), "camera_to_world_row_major": p.tolist()}
               for i, p in enumerate(est)]
    with pytest.raises(DataError, match="absent"):
        evaluate_tum({"metric_scale_known": False, "poses": entries},
                     {"samples": [{"name": "other", "decoded_frame_index": 0}]},
                     [{"t": 1.}], np.array([0., 1., 2.]), np.stack(gt[:3]))


def test_unknown_scale_is_required():
    with pytest.raises(DataError, match="unknown-scale"):
        evaluate_tum({"metric_scale_known": True}, {}, [], np.array([]), np.array([]))
