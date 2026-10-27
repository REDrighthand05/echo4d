"""Frozen E3 B-path selection: one failed match must not quietly alter split."""
from __future__ import annotations

import hashlib
import pytest
import tum_sensor_metric_mesh

from echo4d.model import DataError
from tum_sensor_metric_mesh import split_names, HOLDOUT_INDICES, ANCHOR_INDICES, archive_training_input_evidence


def _frames():
    return [{"name":f"frame_{i:05d}.png","t_tum_rgb_s":float(i)/30} for i in range(90)]


def test_pre_registered_48_12_split():
    train, holdout = split_names(_frames())
    assert len(train) == 48 and len(holdout) == 12
    assert set(train).isdisjoint(holdout)
    assert holdout == [f"frame_{i:05d}.png" for i in HOLDOUT_INDICES]
    assert all(f"frame_{i:05d}.png" not in holdout for i in ANCHOR_INDICES)
    assert all(name.startswith("frame_000") for name in train+holdout)


def test_wrong_source_count_fails_closed():
    with pytest.raises(DataError, match="90"):
        split_names(_frames()[:89])


def test_duplicate_registered_name_fails():
    src = _frames()
    src[1]["name"] = src[0]["name"]
    with pytest.raises(DataError, match="Duplicate"):
        split_names(src)


def test_archive_training_input_evidence_keeps_verifiable_bytes(tmp_path):
    source, target = tmp_path / "source", tmp_path / "artifact"
    source.mkdir()
    target.mkdir()
    samples = {
        "recording.json": b'{"frames":[{"depth":[1,2,3]}]}\\n',
        "sensor-provenance.json": b'{"source_kind":"measured_registered_z_depth"}\\n',
    }
    for name, content in samples.items():
        (source / name).write_bytes(content)
    receipts = archive_training_input_evidence(source, target)
    for original, archived, key in (
        ("recording.json", "b-training-recording.json", "input_manifest_sha256"),
        ("sensor-provenance.json", "b-sensor-provenance.json", "sensor_provenance_sha256"),
    ):
        assert (target / archived).read_bytes() == samples[original]
        assert receipts[key] == hashlib.sha256(samples[original]).hexdigest()
    assert sorted(p.name for p in target.iterdir()) == [
        "b-sensor-provenance.json", "b-training-recording.json"
    ]


def test_archive_training_input_evidence_rejects_corrupted_copy(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "artifact"
    source.mkdir()
    target.mkdir()
    (source / "recording.json").write_bytes(b"real training input")
    (source / "sensor-provenance.json").write_bytes(b"real provenance")

    def corrupted_copy(_source, destination):
        destination.write_bytes(b"corrupted")

    monkeypatch.setattr(tum_sensor_metric_mesh.shutil, "copyfile", corrupted_copy)
    with pytest.raises(DataError, match="Archived training input hash mismatch"):
        archive_training_input_evidence(source, target)
