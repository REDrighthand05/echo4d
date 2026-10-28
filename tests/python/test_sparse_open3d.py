"""Small synthetic geometry guards, never evidence of real-scene improvement."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from echo4d.holdout import read_echo_ply
from echo4d.model import DataError
from sparse_open3d_fusion import fuse_sparse, validate_settings, reject_far_side_of_depth_edges
from tum_sparse_open3d import comparison_sheet, verify_inputs


def make_recording(root: Path, factor: int = 1, *, color: bool = True,
                   mask_all: bool = False) -> Path:
    root.mkdir()
    width, height = 64 // factor, 48 // factor
    depth = np.ones((height, width), dtype=np.float32)
    np.save(root / "depth.npy", depth)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = 220
    Image.fromarray(rgb).save(root / "color.png")
    if mask_all:
        Image.fromarray(np.full((height, width), 255, dtype=np.uint8)).save(root / "mask.png")
    pose = np.eye(4).flatten(order="F").tolist()
    frames = []
    for i in range(2):
        frame = {"t": i * .5, "cameraToWorld": pose, "depth": "depth.npy"}
        if color:
            frame["color"] = "color.png"
        if mask_all:
            frame["mask"] = "mask.png"
        frames.append(frame)
    (root / "recording.json").write_text(json.dumps({
        "version": "echo-dataset-1", "source": "synthetic", "units": "meters",
        "name": "test plane", "duration": 1., "depth_scale": 1,
        "intrinsics": {"width": width, "height": height, "fx": 60. / factor,
                       "fy": 60. / factor, "cx": 31.5 / factor, "cy": 23.5 / factor},
        "frames": frames}), encoding="utf8")
    return root / "recording.json"


def test_reject_invalid_sparse_settings():
    for settings in ((0.0, .09, 4., 2), (.03, .03, 4., 2),
                     (.03, .09, 11., 2), (.03, .09, 4., 3),
                     (float("nan"), .09, 4., 2)):
        with pytest.raises(DataError):
            validate_settings(*settings)


def test_visual_sheet_preserves_missing_geometry(tmp_path):
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    comparison_sheet(None, blank, blank, tmp_path / "free.png", "Free")
    with Image.open(tmp_path / "free.png") as image:
        assert image.size == (1920, 514)
        assert image.getpixel((1100, 200)) == (0, 0, 0)
    with pytest.raises(DataError):
        comparison_sheet(None, blank[:120], blank, tmp_path / "bad.png", "Bad")


def test_synthetic_open3d_plane_produces_colored_mesh(tmp_path):
    pytest.importorskip("open3d")
    source = make_recording(tmp_path / "source")
    result = fuse_sparse(source, tmp_path / "output", voxel_m=.03,
                         truncation_m=.09, max_depth_m=2., pixel_stride=1)
    vertices, faces = read_echo_ply(tmp_path / "output" / "mesh.ply")
    assert result["frames_integrated"] == 2
    assert len(vertices) > 10 and len(faces) > 10
    assert np.all(np.abs(vertices[:, 2] - 1.) < .10)
    assert result["unknown_surface_fill"] is False


def test_synthetic_exclusion_mask_cannot_create_geometry(tmp_path):
    pytest.importorskip("open3d")
    source = make_recording(tmp_path / "source", mask_all=True)
    with pytest.raises(DataError, match="No valid observed depth"):
        fuse_sparse(source, tmp_path / "output", pixel_stride=1)
    assert not (tmp_path / "output").exists()


def test_missing_registered_rgb_rejected(tmp_path):
    pytest.importorskip("open3d")
    source = make_recording(tmp_path / "source", color=False)
    with pytest.raises(DataError, match="registered training RGB"):
        fuse_sparse(source, tmp_path / "output", pixel_stride=1)


def test_edge_guard_rejects_only_far_side_of_measured_jump():
    depth = np.array([[1., 1., 2., 2.],
                      [1., 1., 2., 2.],
                      [1., 1., 2., 2.]], dtype=np.float32)
    filtered, removed = reject_far_side_of_depth_edges(depth, .08)
    assert removed == 3
    assert np.array_equal(filtered[:, :2], depth[:, :2])
    assert np.all(filtered[:, 2] == 0)
    assert np.array_equal(filtered[:, 3], depth[:, 3])
    assert np.array_equal(depth[:, 2], np.full(3, 2., dtype=np.float32))


def test_edge_guard_keeps_unobserved_boundaries_and_smooth_planes():
    depth = np.array([[0., 1., 1.02, 1.],
                      [0., 1., 1.02, 1.]], dtype=np.float32)
    filtered, removed = reject_far_side_of_depth_edges(depth, .08)
    assert removed == 0
    assert np.array_equal(filtered, depth)
    unchanged, removed = reject_far_side_of_depth_edges(depth, 0.)
    assert removed == 0 and unchanged is depth
    with pytest.raises(DataError, match="2D"):
        reject_far_side_of_depth_edges(depth[0], .08)


def test_edge_guard_rejects_invalid_settings():
    for bad in (-.01, .31, float("nan"), True):
        with pytest.raises(DataError):
            validate_settings(.03, .09, 4., 2, bad)
