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


def test_vertex_rgb_interpolation_and_unknown_pixels():
    from echo4d.model import Camera
    from tum_graph_visual import rasterize_vertex_rgb
    points = np.array([[0., 0., 2.], [8., 0., 2.], [0., 8., 2.]])
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    image, count = rasterize_vertex_rgb(points, faces, colors, Camera(5, 5, 1., 1., 0., 0.), np.eye(4))
    assert image[1, 1].tolist() == [128, 64, 64]
    assert image[4, 4].tolist() == [0, 0, 0]
    assert 0 < count < 25


def test_vertex_rgb_zbuffer_and_perspective_correction():
    from echo4d.model import Camera
    from tum_graph_visual import rasterize_vertex_rgb
    points = np.array([[0., 0., 2.], [8., 0., 4.], [0., 8., 2.]])
    # x=8,z=4 projects to x=2, unlike x=4; set x=16 to preserve uv=4.
    points[1, 0] = 16.
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    camera = Camera(5, 5, 1., 1., 0., 0.)
    image, _ = rasterize_vertex_rgb(points, np.array([[0, 1, 2]], dtype=np.int32),
                                     colors, camera, np.eye(4))
    assert image[1, 1].tolist() == [146, 36, 73]
    near_points = np.array([[0., 0., 1.], [4., 0., 1.], [0., 4., 1.]])
    all_points = np.vstack((points, near_points))
    all_colors = np.vstack((colors, np.full((3, 3), [0, 255, 0], dtype=np.uint8)))
    for faces in (np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
                  np.array([[3, 4, 5], [0, 1, 2]], dtype=np.int32)):
        image, _ = rasterize_vertex_rgb(all_points, faces, all_colors, camera, np.eye(4))
        assert image[1, 1].tolist() == [0, 255, 0]


def test_vertex_rgb_invalid_source_and_pose_fail_closed():
    from echo4d.model import Camera
    from tum_graph_visual import rasterize_vertex_rgb
    points = np.array([[0., 0., 2.], [8., 0., 2.], [0., 8., 2.]])
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.zeros((3, 3), dtype=np.uint8)
    camera = Camera(5, 5, 1., 1., 0., 0.)
    with pytest.raises(DataError, match="RGB render inputs"):
        rasterize_vertex_rgb(points, faces, colors.astype(float), camera, np.eye(4))
    with pytest.raises(DataError, match="rigid"):
        rasterize_vertex_rgb(points, faces, colors, camera, np.diag([-1., 1., 1., 1.]))
    points[0, 2] = .01
    image, count = rasterize_vertex_rgb(points, faces, colors, camera, np.eye(4))
    assert count == 0 and not image.any()
