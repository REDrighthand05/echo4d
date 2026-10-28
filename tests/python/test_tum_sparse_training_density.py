"""Fast guards for the real-frame density experiment (not a quality PASS)."""
import copy
import pytest

from echo4d.model import DataError
from tum_composable_graph import HOLDOUT
from tum_sparse_training_density import select_extra_training, select_pose_diverse_training


def source():
    def frame(t):
        return {"t": t, "cameraToWorld": [1, 0, 0, 0, 0, 1, 0, 0,
                                          0, 0, 1, 0, 0, 0, 0, 1]}
    def association(i):
        return {"rgb_path": f"rgb/{i}.png", "depth_path": f"depth/{i}.png",
                "rgb_sha256": f"rgb-{i}", "depth_sha256": f"depth-{i}"}
    old = {"version": "echo-dataset-1", "source": "recorded",
           "frames": [frame(i * .4) for i in range(60)]}
    full = {"version": "echo-dataset-1", "source": "recorded",
            "frames": [frame(i * .2) for i in range(120)]}
    a = {"frames": [association(i * 2) for i in range(60)]}
    b = {"frames": [association(i) for i in range(120)]}
    return old, full, a, b


def test_more_real_training_views_without_holdout_or_temporal_neighbors():
    old, full, a, b = source()
    selected, info = select_extra_training(old, full, a, b)
    times = {frame["t"] for frame in selected["frames"]}
    assert info["baseline_train_frames"] == 48
    assert 48 < info["expanded_train_frames"] < 108
    assert all(old["frames"][i]["t"] in times for i in range(60) if i not in HOLDOUT)
    assert all(min(abs(t - old["frames"][i]["t"]) for i in HOLDOUT) > .24 for t in times)
    assert info["nearest_training_to_holdout_seconds"] > .24


def test_different_original_associations_are_rejected():
    old, full, a, b = source()
    b["frames"][10]["depth_sha256"] = "different-depth"
    with pytest.raises(DataError, match="source association mismatch"):
        select_extra_training(old, full, a, b)


def test_pose_or_timestamp_drift_is_rejected():
    old, full, a, b = source()
    full["frames"][20]["t"] += .01
    with pytest.raises(DataError, match="timestamps"):
        select_extra_training(old, full, a, b)
    old, full, a, b = source()
    full["frames"][20]["cameraToWorld"][12] += .05
    with pytest.raises(DataError, match="poses disagree"):
        select_extra_training(old, full, a, b)


def test_excessive_guard_cannot_silently_delete_original_training():
    old, full, a, b = source()
    with pytest.raises(DataError, match="temporal holdout guard"):
        select_extra_training(old, full, a, b, guard_s=.35)


def dense_source():
    """Synthetic stride-3 source, with camera centers moving through real space."""
    def frame(i):
        pose = [1., 0., 0., 0., 0., 1., 0., 0.,
                0., 0., 1., 0., i * .004, (i % 13) * .002, 0., 1.]
        return {"t": i * .1, "cameraToWorld": pose}
    def association(i):
        return {"rgb_path": f"rgb/{i}.png", "depth_path": f"depth/{i}.png",
                "rgb_sha256": f"rgb-{i}", "depth_sha256": f"depth-{i}"}
    dense = {"version": "echo-dataset-1", "source": "recorded",
             "frames": [frame(i) for i in range(240)]}
    old = {"version": "echo-dataset-1", "source": "recorded",
           "frames": [copy.deepcopy(dense["frames"][4 * i]) for i in range(60)]}
    b = {"frames": [association(i) for i in range(240)]}
    a = {"frames": [copy.deepcopy(b["frames"][4 * i]) for i in range(60)]}
    return old, dense, a, b


def test_pose_diverse_matches_budget_preserves_baseline_and_excludes_holdouts():
    old, dense, a, b = dense_source()
    selected, info = select_pose_diverse_training(old, dense, a, b, budget=88)
    assert len(selected["frames"]) == 88
    assert info["added_frames"] == 40
    assert info["baseline_frames_preserved"] == 48
    times = {f["t"] for f in selected["frames"]}
    assert all(old["frames"][i]["t"] in times for i in range(60) if i not in HOLDOUT)
    assert all(min(abs(t - old["frames"][i]["t"]) for i in HOLDOUT) > .24 for t in times)
    assert len(info["selected_source_indices"]) == 88
    assert info["selected_source_indices"] == sorted(info["selected_source_indices"])
    _, second = select_pose_diverse_training(old, dense, a, b, budget=88)
    assert second["selected_source_indices"] == info["selected_source_indices"]


def test_pose_diverse_rejects_changed_pose_or_source_association():
    old, dense, a, b = dense_source()
    dense["frames"][8]["cameraToWorld"][12] += .1
    with pytest.raises(DataError, match="poses differ"):
        select_pose_diverse_training(old, dense, a, b, budget=88)
    old, dense, a, b = dense_source()
    b["frames"][12]["depth_sha256"] = "modified"
    with pytest.raises(DataError, match="association mismatch"):
        select_pose_diverse_training(old, dense, a, b, budget=88)


def test_pose_diverse_rejects_invalid_budget_and_guard():
    old, dense, a, b = dense_source()
    with pytest.raises(DataError, match="frame budget"):
        select_pose_diverse_training(old, dense, a, b, budget=48)
    with pytest.raises(DataError, match="holdout guard"):
        select_pose_diverse_training(old, dense, a, b, budget=88, guard_s=.35)
