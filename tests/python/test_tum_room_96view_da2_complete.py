"""No-sensor source isolation and nested 96-view ablation tests (CPU, no weights)."""
import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")
pytest.importorskip("trimesh")

from echo4d.model import DataError, file_sha256
import tum_room_96view_da2_complete as complete


def test_fixed_nested_sensor_schedule():
    zero = set(complete.sensor_positions(96, 0))
    a = set(complete.sensor_positions(96, 32))
    b = set(complete.sensor_positions(96, 64))
    c = set(complete.sensor_positions(96, 96))
    assert len(zero) == 0
    assert len(a) == 32 and len(b) == 64 and len(c) == 96
    assert zero < a < b < c
    with pytest.raises(DataError):
        complete.sensor_positions(30, 0)
    with pytest.raises(DataError):
        complete.sensor_positions(96, 95)


def test_96_predictions_use_rgb_and_frozen_depth_without_sensor(monkeypatch, tmp_path):
    """Synthetic 96-view replay has NO measured sensor files to read."""
    replay = tmp_path / "replay"
    (replay / "source" / "rgb").mkdir(parents=True)
    (replay / "inferred" / "depth").mkdir(parents=True)
    image = replay / "source" / "rgb" / "one.png"
    Image.fromarray(np.full((480, 640, 3), 110, dtype=np.uint8)).save(image)
    sha = file_sha256(image)
    (replay / "source" / "association.json").write_text(json.dumps({
        "frames": [{"rgb_path": "rgb/one.png", "rgb_sha256": sha}
                   for _ in range(96)]}))
    frozen = {}
    for i in range(30):
        name = f"depth/{i:03d}.npy"
        np.save(replay / "inferred" / name,
                np.full((240, 320), 1., dtype=np.float32))
        frozen[i] = {"depth": name}
    selected = {
        "intrinsics": {"width": 640, "height": 480,
                       "fx": 525., "fy": 525., "cx": 319.5, "cy": 239.5},
        "frames": [{"t": float(i), "cameraToWorld":
                    np.eye(4).flatten(order="F").tolist()} for i in range(96)]}
    calls = {"infer": 0, "sweep": 0}
    monkeypatch.setattr(complete, "choose_neighbors",
                        lambda poses, i: [1, 2] if i == 0 else [0, 1])
    def sweep(*args):
        calls["sweep"] += 1
        return np.ones((240, 320), dtype=np.float32), {"valid_pixels": 76800}
    def infer(*args):
        calls["infer"] += 1
        return np.ones((240, 320), dtype=np.float32)
    monkeypatch.setattr(complete, "sweep_zncc", sweep)
    monkeypatch.setattr(complete, "infer_relative", infer)
    monkeypatch.setattr(complete, "calibrate_relative_depth",
                        lambda rel, seed: (
                            np.full((240, 320), 2., dtype=np.float32),
                            {"status": "calibrated_from_known_pose_rgb_only"}))
    manifest, receipt = complete.build_complete_rgb_prediction(
        replay, selected, list(range(96)), frozen, tmp_path / "out")
    record = json.loads(manifest.read_text())
    assert len(record["frames"]) == 96
    assert calls == {"infer": 66, "sweep": 66}
    assert receipt["predicted_nonempty_frames"] == 96
    assert np.load(manifest.parent / record["frames"][0]["depth"])[0, 0] == 1.
    assert np.load(manifest.parent / record["frames"][30]["depth"])[0, 0] == 2.
    assert record["frames"][95]["source_anchor_index"] == 95
    assert "sensor" not in record["notes"].lower().replace("0 measured sensor", "")
