"""Synthetic contract tests for real-Mesh 3+2 review preparation (not real evidence)."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from echo4d.model import DataError, file_sha256
from tum_graph_visual import VIEWS, depth_image, look_at, verify_original_rgb


def test_frozen_views_are_three_distinct_holdouts():
    from tum_composable_graph import HOLDOUT
    assert len(VIEWS) == len(set(VIEWS)) == 3
    assert set(VIEWS).issubset(HOLDOUT)


@pytest.mark.parametrize("eye", [[1., 0., 1.], [0., 0., 2.], [-2., 3., 0.5]])
def test_free_camera_is_rigid_and_points_at_target(eye):
    eye = np.asarray(eye)
    pose = look_at(eye, np.zeros(3))
    assert np.allclose(pose[:3, 3], eye)
    assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.)
    assert np.dot(pose[:3, 2], -eye) > 0


def test_degenerate_camera_and_invalid_depth_fail():
    with pytest.raises(DataError, match="coincides"):
        look_at(np.zeros(3), np.zeros(3))
    with pytest.raises(DataError, match="finite"):
        look_at([float("nan"), 0, 1], [0, 0, 0])
    with pytest.raises(DataError, match="Invalid rendered"):
        depth_image(np.array([[-1.]], dtype=float))


def test_depth_image_has_fixed_metric_scale_and_unknown_black():
    pixels = np.asarray(depth_image(np.array([[0., .15, 20., 21.]], dtype=float)))
    assert pixels.tolist() == [[0, 255, 1, 0]]


def rgb_fixture(tmp_path):
    source = tmp_path / "source"
    imported = tmp_path / "imported"
    source.mkdir()
    imported.mkdir()
    (imported / "color").mkdir()
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(640, dtype=np.uint16)[None, :] % 256
    pixels[:, :, 1] = np.arange(480, dtype=np.uint16)[:, None] % 256
    Image.fromarray(pixels).save(source / "original.png")
    Image.fromarray(pixels[::4, ::4]).save(imported / "color" / "00004.png")
    association = {"frames": [{} for _ in range(5)]}
    association["frames"][4] = {
        "index": 4, "rgb_path": "original.png",
        "rgb_sha256": file_sha256(source / "original.png")}
    recording = SimpleNamespace(root=imported, entries=[{} for _ in range(5)])
    recording.entries[4] = {"color": "color/00004.png"}
    return source, imported, association, recording


def test_original_rgb_verified_but_not_redistributed(tmp_path):
    source, imported, association, recording = rgb_fixture(tmp_path)
    result = verify_original_rgb(source, association, recording, 4)
    assert result["source_resolution"] == [640, 480]
    assert result["imported_resolution"] == [160, 120]
    assert result["source_redistributed"] is False
    assert result["source_sha256"] == file_sha256(source / "original.png")


def test_rgb_association_hash_mutation_rejected(tmp_path):
    source, imported, association, recording = rgb_fixture(tmp_path)
    association["frames"][4]["rgb_sha256"] = "0" * 64
    with pytest.raises(DataError, match="SHA"):
        verify_original_rgb(source, association, recording, 4)


def test_downsampled_rgb_pixel_mutation_rejected(tmp_path):
    source, imported, association, recording = rgb_fixture(tmp_path)
    path = imported / "color" / "00004.png"
    pixels = np.asarray(Image.open(path).convert("RGB")).copy()
    pixels[0, 0, 0] ^= 255
    Image.fromarray(pixels).save(path)
    with pytest.raises(DataError, match="pixels differ"):
        verify_original_rgb(source, association, recording, 4)


def test_association_index_swap_rejected(tmp_path):
    source, imported, association, recording = rgb_fixture(tmp_path)
    association["frames"][4]["index"] = 3
    with pytest.raises(DataError, match="index"):
        verify_original_rgb(source, association, recording, 4)
