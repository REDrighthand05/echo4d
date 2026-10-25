"""Frozen A/B protocol guards, with small synthetic data only (not E3)."""
import json
import numpy as np
import pytest

from echo4d.model import DataError
from tum_fair_ab_mesh import make_A_recording, assert_same_comparison_inputs


def _metrics():
    per_frame = [{"name": f"frame_{i:05d}.png",
                  "sensor_depth_sha256": str(i) * 64,
                  "sensor_valid_pixels": 10} for i in range(12)]
    return {"heldout_frames": 12, "sensor_valid_pixels": 120, "per_frame": per_frame}


def test_identical_withheld_file_hashes_and_pixel_count_pass():
    assert_same_comparison_inputs(_metrics(), _metrics())


@pytest.mark.parametrize("mutation", ["frame", "hash", "pixels", "count"])
def test_fair_input_guard_detects_unfair_comparison(mutation):
    a, b = _metrics(), _metrics()
    if mutation == "frame":
        b["per_frame"][7]["name"] = "different.png"
    elif mutation == "hash":
        b["per_frame"][7]["sensor_depth_sha256"] = "f" * 64
    elif mutation == "pixels":
        b["per_frame"][7]["sensor_valid_pixels"] += 1
    else:
        b["sensor_valid_pixels"] = 121
    with pytest.raises(DataError, match="A/B"):
        assert_same_comparison_inputs(a, b)


class _FakeRecording:
    def __len__(self):
        return 48

    def iter_frames(self):
        return iter(range(48))


def test_A_changes_only_pose_and_keeps_depth_and_rgb_inputs(tmp_path, monkeypatch):
    import tum_fair_ab_mesh as module
    monkeypatch.setattr(module, "load_recording", lambda _: _FakeRecording())
    names = [f"frame_{i:05d}.png" for i in range(48)]
    record = {"frames": [{"cameraToWorld": np.eye(4).flatten(order="F").tolist(),
                          "color": f"color/{i:05d}.png",
                          "depth": f"depth/{i:05d}.npy", "t": i*.2}
                         for i in range(48)], "name": "B"}
    original = json.loads(json.dumps(record))
    rec = tmp_path / "recording.json"
    rec.write_text(json.dumps(record))
    prov = tmp_path / "sensor-provenance.json"
    prov.write_text(json.dumps({"train": [{"name": n} for n in names]}))
    gt = {}
    for i, name in enumerate(names):
        pose = np.eye(4)
        pose[:3, 3] = [i*.01, 0, 0]
        gt[name] = pose.flatten(order="F").tolist()
    out = make_A_recording(rec, prov,
                           {"train": [{"name": n} for n in names]}, gt)
    a = json.loads(out.read_text())
    assert len(a["frames"]) == 48
    for i in range(48):
        assert a["frames"][i]["color"] == original["frames"][i]["color"]
        assert a["frames"][i]["depth"] == original["frames"][i]["depth"]
        assert a["frames"][i]["t"] == original["frames"][i]["t"]
        assert a["frames"][i]["cameraToWorld"][12] == pytest.approx(i*.01)
    assert json.loads(rec.read_text()) == original


def test_missing_gt_pose_never_uses_nearest_other_frame(tmp_path, monkeypatch):
    import tum_fair_ab_mesh as module
    monkeypatch.setattr(module, "load_recording", lambda _: _FakeRecording())
    names = [f"frame_{i:05d}.png" for i in range(48)]
    rec = tmp_path / "recording.json"
    rec.write_text(json.dumps({"frames": [{"cameraToWorld": list(range(16))}
                                          for _ in names]}))
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"train": [{"name": n} for n in names]}))
    with pytest.raises(DataError, match="GT unavailable"):
        make_A_recording(rec, prov, {"train": [{"name": n} for n in names]},
                         {n: [1.0]*16 for n in names[:-1]})
    assert not (tmp_path / "recording.A-groundtruth-poses.json").exists()
