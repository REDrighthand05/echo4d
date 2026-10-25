"""Offline negative contracts for PR #34 issue-A cross scorer (no TUM download)."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cross_score_pr34_a import compare_scores
from echo4d.holdout import rasterize_mesh_depth
from echo4d.model import Camera, DataError, rigid_matrix


def fixture_scores():
    names = [f"frame_{i:05d}.png" for i in range(12)]
    frames34, frames25, frames26 = [], [], []
    for i, name in enumerate(names):
        error = .1 if i % 2 == 0 else .2
        score = {
            "sensor_valid_pixels": 100, "overlap_pixels": 100,
            "observed_depth_coverage": 1., "mae_m": error,
            "rmse_m": error, "p95_abs_m": error,
        }
        f = {"name": name, "depth_filename": f"depth/{i}.png",
             "depth_sha256": f"sha{i}", "score": score}
        frames34.append(f)
        frames25.append(copy.deepcopy(f))
        frames26.append({
            "name": name, "sensor_depth_sha256": f["depth_sha256"],
            "status": "measured", **score
        })
    rmse = math.sqrt((.1**2 + .2**2) / 2)
    agg = {
        "mesh_sha256": "mesh1", "gt_camera_poses_loaded": False,
        "per_frame": frames34, "valid_measured_pixels": 1200,
        "overlap_pixels": 1200, "coverage": 1.,
        "mae_m": .15, "rmse_m": rmse, "p95_abs_m": .2,
    }
    old25 = copy.deepcopy(agg)
    old25["per_frame"] = frames25
    old26 = {
        "per_frame": frames26, "sensor_valid_pixels": 1200,
        "overlap_pixels": 1200, "observed_depth_coverage": 1.,
        "mae_m": .15, "rmse_m": rmse, "p95_abs_m": .2,
    }
    return agg, old25, old26, {"holdout": names}


def test_complete_same_input_scores_agree():
    result = compare_scores(*fixture_scores())
    assert len(result["per_frame"]) == 12
    assert result["max_abs_numeric_delta"] == 0
    assert result["weighted_mae_m"] == pytest.approx(.15)


@pytest.mark.parametrize("mutation", ["missing_frame", "wrong_mesh", "wrong_depth",
                                      "wrong_count", "wrong_coverage", "wrong_p95",
                                      "ground_truth_in_B", "duplicate_name"])
def test_differences_fail_closed(mutation):
    cur, old25, old26, split = fixture_scores()
    if mutation == "missing_frame":
        old26["per_frame"].pop()
    elif mutation == "wrong_mesh":
        old25["mesh_sha256"] = "different"
    elif mutation == "wrong_depth":
        old26["per_frame"][0]["sensor_depth_sha256"] = "other"
    elif mutation == "wrong_count":
        old26["per_frame"][0]["overlap_pixels"] -= 1
    elif mutation == "wrong_coverage":
        old25["per_frame"][0]["score"]["observed_depth_coverage"] = .1
    elif mutation == "wrong_p95":
        old26["per_frame"][1]["p95_abs_m"] += 0.00001
    elif mutation == "ground_truth_in_B":
        cur["gt_camera_poses_loaded"] = True
    elif mutation == "duplicate_name":
        split["holdout"][2] = split["holdout"][1]
    with pytest.raises(DataError):
        compare_scores(cur, old25, old26, split)


def test_camera_to_world_is_not_world_to_camera():
    camera = Camera(5, 5, 1., 1., 2., 2.)
    vertices = np.array([[-2., -2., 2.], [2., -2., 2.], [0., 2., 2.]])
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    forward = np.eye(4)
    front_depth, _ = rasterize_mesh_depth(vertices, faces, camera, forward)
    assert front_depth[2, 2] == pytest.approx(2.)
    moved = np.eye(4)
    moved[2, 3] = 1.
    translated_depth, _ = rasterize_mesh_depth(vertices, faces, camera, moved)
    assert translated_depth[2, 2] == pytest.approx(1.)
    behind = np.eye(4)
    behind[0, 0] = -1.
    behind[2, 2] = -1.
    back_depth, _ = rasterize_mesh_depth(vertices, faces, camera, behind)
    assert np.count_nonzero(back_depth) == 0


def test_rigid_column_major_camera_pose_contract():
    pose = np.eye(4)
    pose[:3, 3] = [1., 2., 3.]
    decoded = rigid_matrix(pose.flatten(order="F").tolist())
    assert np.array_equal(decoded, pose)
    assert decoded[:3, 3].tolist() == [1., 2., 3.]
