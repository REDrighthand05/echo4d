"""Fast tests for same-cohort 96-view depth substitution (no GPU/network)."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", reason="OpenCV is installed in the dedicated real-room workflow")
pytest.importorskip("trimesh", reason="trimesh is installed in the dedicated real-room workflow")

from echo4d.model import DataError
from tum_room_sensor_anchor_ladder import anchor_indices
from tum_room_96view_da2_ablation import replace_with_predicted, validate_cohort


def test_nested_predicted_replacements():
    a = set(anchor_indices(30, 8))
    b = set(anchor_indices(30, 16))
    c = set(anchor_indices(30, 30))
    assert len(a) == 8 and len(b) == 16 and len(c) == 30
    assert a < b < c


def test_only_declared_depths_change(tmp_path: Path):
    control = tmp_path / "control"
    inferred = tmp_path / "inferred"
    for folder in (control, inferred):
        (folder / "depth").mkdir(parents=True)
    frames = []
    predicted = {}
    for j in range(3):
        name = f"depth/{j:03d}.npy"
        np.save(control / name, np.full((240, 320), j + 1, np.float32))
        np.save(inferred / name, np.full((240, 320), j + 2, np.float32))
        frames.append({"source_anchor_index": 10 + j, "depth": name, "color": f"color/{j}.png"})
        predicted[10 + j] = {"depth": name}
    (control / "recording.json").write_text(json.dumps({"frames": frames}))
    target = tmp_path / "mixed"
    replace_with_predicted(control, inferred, predicted, [11], target)
    result = json.loads((target / "recording.json").read_text())
    assert [f["source_anchor_index"] for f in result["frames"]] == [10, 11, 12]
    assert np.load(target / "depth/000.npy")[0, 0] == 1
    assert np.load(target / "depth/001.npy")[0, 0] == 3
    assert np.load(target / "depth/002.npy")[0, 0] == 3
    assert np.load(control / "depth/001.npy")[0, 0] == 2
    with pytest.raises(DataError):
        replace_with_predicted(control, inferred, predicted, [11, 11], tmp_path / "bad")
    with pytest.raises(DataError):
        replace_with_predicted(control, inferred, predicted, [13], tmp_path / "bad2")


def test_refuse_pose_drift_and_wrong_96_cohort(monkeypatch, tmp_path):
    from tum_room_96view_da2_ablation import Camera
    import tum_room_96view_da2_ablation as runner
    original = {"intrinsics": {"width": 640, "height": 480, "fx": 525., "fy": 525.,
                                "cx": 319.5, "cy": 239.5},
                "frames": [{"cameraToWorld": np.eye(4).flatten(order="F").tolist(),
                            "t": float(i)} for i in range(100)]}
    (tmp_path / "source").mkdir()
    (tmp_path / "source/recording.json").write_text(json.dumps(original))
    ids = list(range(96))
    monkeypatch.setattr(runner, "select_training", lambda rec: (
        None, {"intrinsics": rec["intrinsics"]},
        {"expanded_indices": ids, "holdout_indices": [98, 99]}))
    cam = Camera.from_dict(original["intrinsics"]).scaled(2).to_dict()
    f = {"source_anchor_index": 4, "cameraToWorld": np.eye(4).flatten(order="F").tolist()}
    inferred = {"intrinsics": cam, "frames": [f] * 30}
    with pytest.raises(DataError):
        validate_cohort({"selection": {"expanded_indices": ids}}, tmp_path, inferred)
    inferred["frames"] = [dict(source_anchor_index=i, cameraToWorld=np.eye(4).flatten(order="F").tolist()) for i in range(30)]
    inferred["frames"][4]["cameraToWorld"][12] = 1.0
    with pytest.raises(DataError):
        validate_cohort({"selection": {"expanded_indices": ids}}, tmp_path, inferred)
    with pytest.raises(DataError):
        validate_cohort({"selection": {"expanded_indices": ids[:-1]}}, tmp_path,
                        {"intrinsics": cam, "frames": [dict(source_anchor_index=i,
                        cameraToWorld=np.eye(4).flatten(order="F").tolist()) for i in range(30)]})
