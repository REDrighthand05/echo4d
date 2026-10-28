"""Small synthetic selection guards; never mistaken for a real scene run."""
import copy
import numpy as np
import pytest

from echo4d.model import DataError
from tum_room_sparse_quality import (
    BASE_FRAMES, EXPANDED_FRAMES, HOLDOUT, IMPORT_FRAMES, select_training,
)


def synthetic_manifest(n=IMPORT_FRAMES):
    frames = []
    for i in range(n):
        pose = np.eye(4)
        pose[0, 3] = 0.005 * i
        pose[1, 3] = 0.02 * np.sin(i / 18)
        frames.append({"t": 0.1 * i, "cameraToWorld": pose.flatten(order="F").tolist()})
    return {"version": "echo-dataset-1", "source": "recorded", "frames": frames}


def test_full_trajectory_48_subset_96_without_holdout_leakage():
    manifest = synthetic_manifest()
    baseline, expanded, info = select_training(manifest)
    assert len(baseline["frames"]) == BASE_FRAMES
    assert len(expanded["frames"]) == EXPANDED_FRAMES
    assert info["expanded_contains_baseline"]
    assert len(HOLDOUT) == 12
    held_times = {manifest["frames"][i]["t"] for i in HOLDOUT}
    base_times = {f["t"] for f in baseline["frames"]}
    extra_times = {f["t"] for f in expanded["frames"]}
    assert base_times.issubset(extra_times)
    assert all(min(abs(t - h) for h in held_times) > 0.16 for t in extra_times)
    assert info["min_train_holdout_time_gap_s"] > 0.16
    assert info["training_camera_center_extent_m"][0] > 0.5
    _, second, other = select_training(manifest)
    assert [f["t"] for f in second["frames"]] == [f["t"] for f in expanded["frames"]]
    assert info["expanded_indices"] == other["expanded_indices"]


def test_rejects_nonfinite_and_nonmonotonic_timestamps():
    original = synthetic_manifest()
    for value in (float("nan"), -1):
        changed = copy.deepcopy(original)
        changed["frames"][80]["t"] = value
        with pytest.raises(DataError, match="timestamps"):
            select_training(changed)


def test_rejects_missing_real_frames_and_insufficient_budget():
    original = synthetic_manifest()
    changed = copy.deepcopy(original)
    changed["frames"].pop()
    with pytest.raises(DataError, match="180"):
        select_training(changed)
    with pytest.raises(DataError, match="budget"):
        select_training(original, baseline=48, expanded=121)
    with pytest.raises(DataError, match="budget"):
        select_training(original, baseline=48, expanded=96, guard_s=0.4)


def test_shorter_real_room_sequence_keeps_twelve_holdouts():
    data = synthetic_manifest(160)
    baseline, expanded, receipt = select_training(data)
    assert len(baseline["frames"]) == 48
    assert len(expanded["frames"]) == 96
    assert len(receipt["holdout_indices"]) == 12
    assert set(receipt["expanded_indices"]).isdisjoint(receipt["holdout_indices"])
    assert receipt["min_train_holdout_time_gap_s"] > 0.16
