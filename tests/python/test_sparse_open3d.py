"""Small synthetic geometry guards, never evidence of real-scene improvement."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from echo4d.holdout import read_echo_ply
from echo4d.model import Camera, DataError
from sparse_open3d_fusion import (fuse_sparse, validate_settings, reject_far_side_of_depth_edges,
                                  reject_cross_frame_foreground, select_cross_frame_neighbors)
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



def test_cross_frame_two_stable_farther_observations_reject_spurious_foreground():
    camera = Camera(12, 12, 10., 10., 5.5, 5.5)
    pose = np.eye(4)
    depth = np.ones((12, 12), dtype=np.float32)
    depth[0, 0] = 0.
    far = np.full((12, 12), 1.4, dtype=np.float32)
    filtered, removed = reject_cross_frame_foreground(
        depth, pose, [(far, pose), (far, pose)], camera, .10)
    assert removed == 100
    assert np.all(filtered[1:11, 1:11] == 0)
    assert depth[5, 5] == 1.0  # no mutation or invented depth
    assert filtered[0, 0] == 0


def test_cross_frame_agreement_and_occlusion_do_not_erase_observed_geometry():
    camera = Camera(12, 12, 10., 10., 5.5, 5.5)
    pose = np.eye(4)
    depth = np.ones((12, 12), dtype=np.float32)
    far = np.full_like(depth, 1.4)
    agree = np.ones_like(depth)
    nearer = np.full_like(depth, .5)
    for witnesses in ([(far, pose), (agree, pose)],
                      [(far, pose), (nearer, pose)],
                      [(far, pose)], [(np.zeros_like(depth), pose), (far, pose)]):
        filtered, removed = reject_cross_frame_foreground(
            depth, pose, witnesses, camera, .10)
        assert removed == 0
        assert np.array_equal(filtered, depth)
    with pytest.raises(DataError, match="calibrated depth"):
        reject_cross_frame_foreground(depth[:10], pose, [], camera, .10)


def test_cross_frame_neighbor_selection_requires_real_baseline_and_overlap():
    poses = []
    for i in range(5):
        pose = np.eye(4)
        pose[0, 3] = .06 * i
        poses.append(pose)
    chosen = select_cross_frame_neighbors(poses)
    assert len(chosen[2]) >= 2
    assert 2 not in chosen[2]
    assert all(abs(i - 2) >= 1 for i in chosen[2])
    no_parallax = select_cross_frame_neighbors([np.eye(4) for _ in range(3)])
    assert no_parallax == [[], [], []]
    with pytest.raises(DataError, match="Cross-frame depth guard"):
        validate_settings(.03, .09, 4., 2, 0., -.1)


def test_two_pixel_far_side_halo_preserves_foreground_and_far_interior():
    depth = np.tile(np.array([2., 2., 2., 1., 1., 1.], dtype=np.float32), (3, 1))
    one, removed_one = reject_far_side_of_depth_edges(depth, .08, halo_px=1)
    two, removed_two = reject_far_side_of_depth_edges(depth, .08, halo_px=2)
    assert removed_one == 3 and removed_two == 6
    assert np.all(one[:, 2] == 0) and np.all(one[:, 1] == 2.)
    assert np.all(two[:, 1:3] == 0) and np.all(two[:, 0] == 2.)
    assert np.array_equal(two[:, 3:], depth[:, 3:])
    assert np.all(depth[:, 1:3] == 2.)
    with pytest.raises(DataError, match="Edge halo"):
        validate_settings(.025, .12, 4., 1, 0., 0., edge_halo_pixels=3)
    with pytest.raises(DataError, match="Edge halo"):
        reject_far_side_of_depth_edges(depth, .08, halo_px=True)
