"""Sensor-measured B-path contract; deliberately no reconstructed-geometry claims."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from echo4d.metric_rgbd import convert_metric_sensor
from echo4d.model import DataError, file_sha256, load_recording


def _fixture(root: Path):
    model, rgb, depth = [root / part for part in ("model", "images", "sensor")]
    for p in (model, rgb, depth):
        p.mkdir(parents=True)
    (model / "cameras.txt").write_text("1 PINHOLE 4 4 3 3 1.5 1.5\n")
    (model / "images.txt").write_text("# example; hash-bound SfM model\n")
    sfm_frames, metric_frames = [], []
    scale = .1
    for i in range(4):
        name = f"frame_{i:05d}.png"
        Image.new("RGB", (4, 4), color=(10+i, 40, 90)).save(rgb / name)
        np.save(depth / f"d{i}.npy", np.full((4, 4), 2+i*.1, dtype=np.float32))
        pose = np.eye(4)
        pose[0, 3] = i
        sfm_frames.append({"name": name, "t_s": i, "camera_id": 1,
                           "camera_to_sfm_world": pose.flatten(order="F").tolist()})
        pose[0, 3] *= scale
        metric_frames.append({"name": name, "t_s": i, "camera_id": 1,
                              "camera_to_metric_sfm_world": pose.flatten(order="F").tolist()})
    original = {
        "version": "echo-sfm-trajectory-1", "units": "unknown",
        "metric_scale_estimated": False,
        "model_images_txt_sha256": file_sha256(model / "images.txt"),
        "model_cameras_txt_sha256": file_sha256(model / "cameras.txt"),
        "frames": sfm_frames
    }
    sfm_path = root / "sfm.json"
    sfm_path.write_text(json.dumps(original))
    metric = {
        "version": "echo-sfm-metric-scale-1", "units": "meters",
        "metric_scale_estimated_from_measured_depth": True,
        "scale_is_from_external_gt_poses": False,
        "source_trajectory_sha256": file_sha256(sfm_path),
        "scale_m_per_sfm_unit": scale,
        "anchor_diagnostics": [{"name":f"frame_{i:05d}.png"} for i in range(3)],
        "frames": metric_frames,
    }
    metric_path = root / "metric.json"
    metric_path.write_text(json.dumps(metric))
    manifest = {
        "source_kind": "measured_registered_z_depth",
        "source_description": "independent calibrated physical sensor",
        "depth_numeric_units_per_meter": 1,
        "train": [
            {"name":f"frame_{i:05d}.png", "depth":f"d{i}.npy",
             "timestamp_delta_seconds":0.002} for i in range(3)
        ],
        "holdout": ["frame_00003.png"],
    }
    split_path = root / "split.json"
    split_path.write_text(json.dumps(manifest))
    return [metric_path, sfm_path, model, rgb, depth, split_path, root / "out"]


def _alter(path, callback):
    x = json.loads(path.read_text())
    callback(x)
    path.write_text(json.dumps(x))


def test_real_sensor_depth_emitted_and_holdout_never_loaded(tmp_path):
    arguments = _fixture(tmp_path)
    # The withheld RGB/depth may be absent without stopping conversion: no leakage.
    (tmp_path / "sensor" / "d3.npy").unlink()
    result = convert_metric_sensor(*arguments, downsample=2)
    assert result["training_frames"] == 3
    assert result["heldout_frames"] == 1
    recording = load_recording(tmp_path / "out" / "recording.json")
    assert recording.camera.width == 2 and recording.camera.height == 2
    frames = list(recording.iter_frames())
    assert len(frames) == 3
    assert np.allclose(frames[1][1].camera_to_world[:3, 3], [0.1, 0, 0])
    assert np.allclose(frames[0][1].depth, 2.0)
    audit = json.loads((tmp_path / "out" / "sensor-provenance.json").read_text())
    assert audit["holdout_names_not_loaded"] == ["frame_00003.png"]
    assert len(audit["train"]) == 3
    assert all("sensor_depth_sha256" in v for v in audit["train"])


@pytest.mark.parametrize("mutation", [
    "wrong_metric_scale", "metric_from_gt", "tamper_camera_model",
    "train_holdout_overlap", "holdout_anchor", "bad_offset",
    "predicted_depth_label", "wrong_depth_shape", "wrong_pose",
    "bad_sfm_link", "same_depth_twice",
])
def test_reject_provenance_leakage_and_invalid_geometry(tmp_path, mutation):
    args = _fixture(tmp_path)
    metric, sfm, model, rgb, depth, split, output = args
    if mutation == "wrong_metric_scale":
        _alter(metric, lambda x: x.update(scale_m_per_sfm_unit=.2))
    elif mutation == "metric_from_gt":
        _alter(metric, lambda x: x.update(scale_is_from_external_gt_poses=True))
    elif mutation == "tamper_camera_model":
        (model / "cameras.txt").write_text("1 PINHOLE 4 4 4 4 1.5 1.5\n")
    elif mutation == "train_holdout_overlap":
        _alter(split, lambda x: x["holdout"].append("frame_00001.png"))
    elif mutation == "holdout_anchor":
        _alter(split, lambda x: x.update(holdout=["frame_00000.png"]))
    elif mutation == "bad_offset":
        _alter(split, lambda x: x["train"][0].update(timestamp_delta_seconds=.04))
    elif mutation == "predicted_depth_label":
        _alter(split, lambda x: x.update(source_kind="predicted_depth"))
    elif mutation == "wrong_depth_shape":
        np.save(depth / "d0.npy", np.ones((3, 4), np.float32))
    elif mutation == "wrong_pose":
        def alter(x):
            x["frames"][1]["camera_to_metric_sfm_world"][12] = .9
        _alter(metric, alter)
    elif mutation == "bad_sfm_link":
        _alter(metric, lambda x: x.update(source_trajectory_sha256="a"*64))
    elif mutation == "same_depth_twice":
        _alter(split, lambda x: x["train"][1].update(depth="d0.npy"))
    with pytest.raises((DataError, TypeError)):
        convert_metric_sensor(*args, downsample=2)
    assert not output.exists(), "Invalid input must not publish a partial dataset"


def test_refuse_existing_target(tmp_path):
    args = _fixture(tmp_path)
    args[-1].mkdir()
    with pytest.raises(DataError, match="already exists"):
        convert_metric_sensor(*args)
