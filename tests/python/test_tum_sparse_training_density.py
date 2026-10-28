"""Fast guards for the real-frame density experiment (not a quality PASS)."""
import copy
import pytest

from echo4d.model import DataError
from tum_composable_graph import HOLDOUT
from tum_sparse_training_density import select_extra_training


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
