"""Byte-preserving metadata retention; no real-scene or image claim."""
import json
import pytest
from echo4d.graph_runner import GraphBlocked, InputFile, archive_pinned_files
from echo4d.model import DataError, file_sha256


def test_archive_preserves_actual_bytes_and_pins(tmp_path):
    original = tmp_path / "original.json"
    raw = b'{  "label": "\\u4fdd\\u7559", "value": 1 }\r\n'
    original.write_bytes(raw)
    sha = file_sha256(original)
    result = archive_pinned_files({"train-recording.json": InputFile(original, sha)}, tmp_path / "metadata")
    assert (tmp_path / "metadata/train-recording.json").read_bytes() == raw
    assert result["train-recording.json"] == {"sha256": sha, "bytes": len(raw)}
    receipt = json.loads((tmp_path / "metadata/archive.json").read_text())
    assert receipt["files"] == result
    assert original.read_bytes() == raw


def test_changed_metadata_cannot_publish_partial_archive(tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.write_bytes(b"valid")
    bad.write_bytes(b"before")
    expected = file_sha256(bad)
    bad.write_bytes(b"after")
    with pytest.raises(GraphBlocked, match="execution pin"):
        archive_pinned_files({"full-recording.json": InputFile(good, file_sha256(good)),
                              "train-recording.json": InputFile(bad, expected)}, tmp_path / "metadata")
    assert not (tmp_path / "metadata").exists()


def test_archive_never_overwrites_previous_metadata(tmp_path):
    original = tmp_path / "source"
    original.write_bytes(b"new")
    dest = tmp_path / "metadata"
    dest.mkdir()
    (dest / "keep").write_bytes(b"old")
    with pytest.raises(DataError, match="already exists"):
        archive_pinned_files({"association.json": InputFile(original, file_sha256(original))}, dest)
    assert (dest / "keep").read_bytes() == b"old"


@pytest.mark.parametrize("name", ["../escape.json", "nested/file.json"])
def test_archive_rejects_path_escape(tmp_path, name):
    original = tmp_path / "source"
    original.write_bytes(b"metadata")
    with pytest.raises(GraphBlocked, match="archive entry"):
        archive_pinned_files({name: InputFile(original, file_sha256(original))}, tmp_path / "metadata")
    assert not (tmp_path / "escape.json").exists()
    assert not (tmp_path / "metadata").exists()
