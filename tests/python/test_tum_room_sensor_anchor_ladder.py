"""CPU regression tests for real TUM sensor-frame ablation boundaries."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("cv2")
pytest.importorskip("trimesh")

from echo4d.model import DataError
from tum_room_sensor_anchor_ladder import (
    ROOT, anchor_indices, make_candidate, verified_sensor_depths)


def test_nested_sensor_levels_and_reject_invalid():
    zero = anchor_indices(30, 0)
    four = anchor_indices(30, 4)
    eight = anchor_indices(30, 8)
    full = anchor_indices(30, 30)
    assert zero == []
    assert len(four) == 4 and len(eight) == 8 and len(full) == 30
    assert set(four) < set(eight) < set(full)
    assert four[0] == 0 and four[-1] == 29
    with pytest.raises(DataError):
        anchor_indices(30, 31)


def test_only_requested_sensor_png_is_read_and_verified(tmp_path):
    arr = np.zeros((480, 640), np.uint16)
    arr[100:400, 100:500] = 7500
    data = BytesIO()
    Image.fromarray(arr).save(data, format="PNG")
    raw = data.getvalue()
    path = ROOT + "/depth/test.png"
    archive = tmp_path / "room.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(path)
        member.size = len(raw)
        tar.addfile(member, BytesIO(raw))
    assoc = {"frames": [
        {"depth_path": "depth/test.png", "depth_sha256": hashlib.sha256(raw).hexdigest()},
        {"depth_path": "depth/absent.png", "depth_sha256": "0" * 64}]}
    depths, hashes = verified_sensor_depths(archive, assoc, [0])
    assert set(depths) == {0}
    assert depths[0].shape == (240, 320)
    assert float(depths[0].max()) == 1.5
    assert hashes[0] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(DataError):
        verified_sensor_depths(archive, assoc, [1])
    assoc["frames"][0]["depth_sha256"] = "0" * 64
    with pytest.raises(DataError):
        verified_sensor_depths(archive, assoc, [0])


def test_replaces_only_declared_anchor_and_keeps_other_predicted_z(tmp_path):
    frozen = tmp_path / "frozen"
    (frozen / "depth").mkdir(parents=True)
    (frozen / "color").mkdir()
    frames = []
    for i in range(3):
        np.save(frozen / "depth" / f"{i:03d}.npy",
                np.full((240, 320), float(i + 1), np.float32))
        Image.fromarray(np.zeros((240, 320, 3), np.uint8)).save(
            frozen / "color" / f"{i:03d}.png")
        frames.append({"source_anchor_index": i + 10,
                       "depth": f"depth/{i:03d}.npy",
                       "color": f"color/{i:03d}.png"})
    manifest = {"frames": frames}
    (frozen / "recording.json").write_text(json.dumps(manifest))
    output = tmp_path / "candidate"
    replacement = np.full((240, 320), 2.5, np.float32)
    _, receipt = make_candidate(frozen, manifest, output, [1], {11: replacement}, 1)
    assert receipt["anchors"][0]["source_index"] == 11
    assert np.load(output / "depth/001.npy")[0, 0] == 2.5
    assert np.load(output / "depth/000.npy")[0, 0] == 1.0
    assert np.load(output / "depth/002.npy")[0, 0] == 3.0
    with pytest.raises(DataError):
        make_candidate(frozen, manifest, output, [1], {11: replacement}, 1)
