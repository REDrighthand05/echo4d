"""Synthetic counterexamples only; real TUM scoring is a separate workflow."""
import json
import numpy as np
import pytest

from echo4d.model import DataError, file_sha256
import tum_graph_holdout as gate
from tum_composable_graph import HOLDOUT, TOTAL
from tum_public_roundtrip import PINNED_ARCHIVE_SHA256


def fixture(tmp_path):
    root = tmp_path / "imported"
    root.mkdir()
    identity = [1., 0., 0., 0., 0., 1., 0., 0.,
                0., 0., 1., 0., 0., 0., 0., 1.]
    frames = []
    for i in range(TOTAL):
        filename = f"depth-{i:02d}.npy"
        np.save(root / filename, np.ones((2, 2), dtype=np.float32))
        frames.append({"t": float(i + 1), "cameraToWorld": identity, "depth": filename})
    data = {
        "version": "echo-dataset-1", "source": "recorded", "units": "meters",
        "duration": 61., "intrinsics": {
            "width": 2, "height": 2, "fx": 1., "fy": 1., "cx": 0., "cy": 0.
        }, "frames": frames,
    }
    full, train = root / "recording.json", root / "recording-train.json"
    full.write_text(json.dumps(data), encoding="utf-8")
    data["frames"] = [f for i, f in enumerate(frames) if i not in HOLDOUT]
    train.write_text(json.dumps(data), encoding="utf-8")
    mesh = tmp_path / "mesh.ply"
    mesh.write_bytes(b"test-mesh")
    target = "mesh/world/metric/none/estimated/train/fixture"
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps({
        "status": "executed_unscored", "quality": "not_evaluated",
        "input_sha256": {"echo_recording/fixture": file_sha256(train)},
        "target": target, "outputs": {target: {"sha256": file_sha256(mesh)}},
    }), encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "status": "executed_unscored", "mesh_sha256": file_sha256(mesh),
        "source": {"archive_sha256": PINNED_ARCHIVE_SHA256},
        "input": {
            "original_manifest_sha256": file_sha256(full),
            "train_manifest_sha256": file_sha256(train),
            "holdout_indices": list(HOLDOUT), "train_frames": 48, "holdout_frames": 12,
        },
    }), encoding="utf-8")
    return full, train, mesh, execution, report


def fake_renderer(monkeypatch, mutate=None, omit=False):
    monkeypatch.setattr(gate, "read_echo_ply",
                        lambda _: (np.zeros((3, 3)), np.zeros((1, 3), dtype=int)))
    def render(vertices, faces, recording, indices, max_depth):
        if mutate:
            mutate(recording)
        return {
            "heldout_frames": 12, "heldout_indices": list(indices),
            "per_frame": [{"index": i} for i in (indices[:-1] if omit else indices)],
            "rmse_m": .25, "observed_depth_coverage": .6,
        }
    monkeypatch.setattr(gate, "evaluate_frames", render)


def test_diagnostic_never_claims_independent_acceptance(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    fake_renderer(monkeypatch)
    result = gate.score_frozen_holdout(*paths)
    assert result["status"] == "diagnostic_only"
    assert result["independent_surface_truth"] is False
    assert result["quality_acceptance"] == "not_evaluated"
    assert result["diagnostic"]["heldout_indices"] == list(HOLDOUT)
    assert len(result["holdout_source_sha256"]) == 12
    assert result["mesh_sha256"] == file_sha256(paths[2])


def test_train_split_mutation_fails_before_render(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    fake_renderer(monkeypatch)
    train = json.loads(paths[1].read_text())
    train["frames"][0]["depth"] = json.loads(paths[0].read_text())["frames"][HOLDOUT[0]]["depth"]
    paths[1].write_text(json.dumps(train))
    with pytest.raises(DataError, match="complement"):
        gate.score_frozen_holdout(*paths)


def test_reused_train_depth_in_holdout_is_rejected(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    fake_renderer(monkeypatch)
    full = json.loads(paths[0].read_text())
    full["frames"][HOLDOUT[0]]["depth"] = full["frames"][0]["depth"]
    paths[0].write_text(json.dumps(full))
    report = json.loads(paths[4].read_text())
    report["input"]["original_manifest_sha256"] = file_sha256(paths[0])
    paths[4].write_text(json.dumps(report))
    with pytest.raises(DataError, match="overlap or repeat"):
        gate.score_frozen_holdout(*paths)


def test_mesh_tamper_is_blocked(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    fake_renderer(monkeypatch)
    paths[2].write_bytes(b"tampered")
    with pytest.raises(DataError, match="Mesh does not match"):
        gate.score_frozen_holdout(*paths)


def test_source_mutation_during_evaluation_fails(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    def change(recording):
        (recording.root / recording.entries[HOLDOUT[0]]["depth"]).write_bytes(b"tampered")
    fake_renderer(monkeypatch, mutate=change)
    with pytest.raises(DataError, match="changed during evaluation"):
        gate.score_frozen_holdout(*paths)


def test_missing_holdout_frame_fails(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    fake_renderer(monkeypatch, omit=True)
    with pytest.raises(DataError, match="omitted or reordered"):
        gate.score_frozen_holdout(*paths)


def test_failed_graph_cannot_be_scored(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    fake_renderer(monkeypatch)
    receipt = json.loads(paths[3].read_text())
    receipt["status"] = "failed"
    paths[3].write_text(json.dumps(receipt))
    with pytest.raises(DataError, match="unverified or incomplete"):
        gate.score_frozen_holdout(*paths)
