"""B-path Mesh validation: holdouts remain separate from training and pose GT."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest

from echo4d.model import Camera, DataError
from echo4d.metric_holdout import validate_holdout_split, evaluate_sensor_metric_holdouts


def _fixture():
    names = [f"frame_{i:05d}.png" for i in range(90)]
    pose = np.eye(4).flatten(order="F").tolist()
    source = [{"name": n, "t_tum_rgb_s": float(i)} for i, n in enumerate(names)]
    metric = {
        "version": "echo-sfm-metric-scale-1",
        "units": "meters",
        "scale_is_from_external_gt_poses": False,
        "scale_m_per_sfm_unit": .1,
        "metric_scale_estimated_from_measured_depth": True,
        "anchor_diagnostics": [{"name": names[i]} for i in (15, 45, 75)],
        "frames": [{"name": n, "camera_id": 1,
                    "camera_to_metric_sfm_world": pose} for n in names],
    }
    split = {
        "source_kind": "measured_registered_z_depth",
        "depth_numeric_units_per_meter": 5000,
        "train": [{"name": n, "depth": f"depth/{i:05d}.png"} for i, n in enumerate(names[:48])],
        "holdout": names[48:60],
    }
    return source, metric, split


def test_fixed_48_12_holdout_contract():
    source, metric, split = _fixture()
    result = validate_holdout_split(split, source, metric)
    assert list(result) == split["holdout"]
    assert len(result) == 12


@pytest.mark.parametrize("mutation", [
    "overlap", "duplicate", "missing_pose", "anchor_in_holdout",
    "fake_scale", "no_external_sensor_scale", "wrong_source_kind",
    "incomplete_holdout", "gt_scale", "wrong_units", "wrong_depth_units",
    "boolean_scale", "duplicate_anchor", "missing_anchor", "duplicate_train_depth"
])
def test_untrustworthy_split_fails_closed(mutation):
    source, metric, split = _fixture()
    if mutation == "overlap":
        split["holdout"][0] = split["train"][0]["name"]
    elif mutation == "duplicate":
        split["holdout"][0] = split["holdout"][1]
    elif mutation == "missing_pose":
        metric["frames"] = metric["frames"][:-31]
    elif mutation == "anchor_in_holdout":
        metric["anchor_diagnostics"][0]["name"] = split["holdout"][0]
    elif mutation == "fake_scale":
        metric["scale_m_per_sfm_unit"] = -1
    elif mutation == "no_external_sensor_scale":
        metric["metric_scale_estimated_from_measured_depth"] = False
    elif mutation == "wrong_source_kind":
        split["source_kind"] = "predicted_depth"
    elif mutation == "incomplete_holdout":
        split["holdout"].pop()
    elif mutation == "gt_scale":
        metric["scale_is_from_external_gt_poses"] = True
    elif mutation == "wrong_units":
        metric["units"] = "unknown"
    elif mutation == "wrong_depth_units":
        split["depth_numeric_units_per_meter"] = 1000
    elif mutation == "boolean_scale":
        metric["scale_m_per_sfm_unit"] = True
    elif mutation == "duplicate_anchor":
        metric["anchor_diagnostics"][0] = metric["anchor_diagnostics"][1]
    elif mutation == "missing_anchor":
        metric["anchor_diagnostics"].pop()
    elif mutation == "duplicate_train_depth":
        split["train"][0]["depth"] = split["train"][1]["depth"]
    with pytest.raises(DataError):
        validate_holdout_split(split, source, metric)


def test_known_10cm_depth_error_after_mesh_without_gt(tmp_path, monkeypatch):
    import echo4d.metric_holdout as mod
    source, metric, split = _fixture()
    root = tmp_path / "tum"
    depth = root / "depth"
    depth.mkdir(parents=True)
    for i in range(90):
        (depth / f"{i:05d}.png").write_bytes(b"pinned synthetic bytes")

    cam = Camera(4, 4, 3., 3., 1.5, 1.5)
    monkeypatch.setattr(mod, "load_recording", lambda _: SimpleNamespace(camera=cam))
    monkeypatch.setattr(mod, "_cameras", lambda _: {1:cam})
    monkeypatch.setattr(mod, "read_echo_ply", lambda _: (np.zeros((3,3)), np.zeros((1,3), dtype=int)))
    monkeypatch.setattr(mod, "read_rows", lambda *_: [[str(i), f"depth/{i:05d}.png"] for i in range(90)])
    monkeypatch.setattr(mod, "read_measured_depth", lambda *_: np.full((4,4), 2., np.float64))
    monkeypatch.setattr(mod, "rasterize_mesh_depth", lambda *args, **kwargs: (
        np.full((4,4), 2.1, np.float32), {"triangles_drawn":1}))
    mesh = tmp_path / "mesh.ply"
    mesh.write_bytes(b"mock test mesh")
    result = evaluate_sensor_metric_holdouts(
        mesh, tmp_path / "train.json", metric, split, source,
        root, tmp_path / "model", downsample=1)
    assert result["holdout_frames"] == 12
    assert result["coverage"] == pytest.approx(1.0)
    assert result["rmse_m"] == pytest.approx(0.1, abs=1e-6)
    assert result["mae_m"] == pytest.approx(0.1, abs=1e-6)
    assert result["gt_camera_poses_loaded"] is False
    assert result["screening_gate_coverage_gte_80pct"] is True
    assert result["screening_gate_overlap_rmse_lte_15cm"] is True


def test_training_depth_reuse_is_rejected(tmp_path, monkeypatch):
    import echo4d.metric_holdout as mod
    source, metric, split = _fixture()
    root = tmp_path / "tum"
    root.mkdir()
    (root / "depth").mkdir()
    for row in split['train']:
        (root / row['depth']).write_bytes(b'training measurement')
    cam = Camera(4, 4, 3., 3., 1.5, 1.5)
    monkeypatch.setattr(mod, "load_recording", lambda _: SimpleNamespace(camera=cam))
    monkeypatch.setattr(mod, "_cameras", lambda _: {1:cam})
    monkeypatch.setattr(mod, "read_echo_ply", lambda _: (np.zeros((3,3)), np.zeros((1,3), dtype=int)))
    # First heldout selection illegally points to the first training depth.
    monkeypatch.setattr(mod, "read_rows", lambda *_: [
        ["48", split["train"][0]["depth"]],
        *[[str(i), f"depth/{i:05d}.png"] for i in range(49, 60)]
    ])
    # Duplicate/overlap must abort before loading heldout measurements.
    monkeypatch.setattr(mod, "read_measured_depth", lambda *_: pytest.fail("must not load heldout Z"))
    mesh = tmp_path / "mesh.ply"
    mesh.write_bytes(b"mock")
    with pytest.raises(DataError, match="used in training"):
        evaluate_sensor_metric_holdouts(mesh, tmp_path / "train.json", metric,
                                        split, source, root, tmp_path / "model", downsample=1)
