"""Synthetic checks for the opt-in observed-edge TSDF extractor."""
import numpy as np

from echo4d.fusion import FusionConfig, TSDFVolume
from echo4d.known_pose_edge_extraction import (
    _attached_faces, _observed_edge_vertices, extract_observed_edge_mesh)


def _plane_volume(missing=False):
    cfg = FusionConfig((0, 0, 0, .5, .5, .5), voxel=.05,
                       truncation=.15, max_depth=4)
    vol = TSDFVolume(cfg)
    vol.tsdf[:] = np.broadcast_to(
        (np.arange(10, dtype=np.float32)[:, None, None] - 4.5) / 5,
        vol.shape)
    vol.weight[:] = 1
    vol.first[:] = 0
    vol.last[:] = 1
    vol.color[:] = [80, 120, 160]
    vol.color_weight[:] = 1
    if missing:
        vol.weight[4, 4, 4] = 0
        vol.color_weight[4, 4, 4] = 0
    return vol


def test_complete_volume_keeps_original_triangles():
    vol = _plane_volume()
    old = vol.extract_mesh()
    new, report = extract_observed_edge_mesh(vol)
    assert len(new.faces) == len(old.faces)
    assert report["attached_added_faces"] == 0
    assert report["unknown_endpoint_accepted"] is False
    assert np.all(new.observations >= 1)


def test_partially_observed_cell_can_extend_supported_surface():
    vol = _plane_volume(missing=True)
    original = vol.extract_mesh()
    new, report = extract_observed_edge_mesh(vol)
    assert report["baseline_fully_observed_faces"] == len(original.faces)
    assert report["attached_added_faces"] > 0
    assert len(new.faces) > len(original.faces)
    assert np.all(new.observations > 0)
    assert vol.weight[4, 4, 4] == 0  # no invented integration


def test_missing_crossing_endpoint_is_never_supported():
    sdf = np.broadcast_to(
        (np.arange(5, dtype=np.float32)[:, None, None] - 1.5),
        (5, 5, 5)).copy()
    seen = np.ones(sdf.shape, dtype=bool)
    seen[2, 3, 2] = False
    vertices = np.array([[1.5, 2., 2.], [1.5, 3., 2.]])
    assert _observed_edge_vertices(vertices, seen, sdf).tolist() == [True, False]


def test_new_island_without_old_shared_edge_is_rejected():
    faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6]], dtype=np.int32)
    attached = _attached_faces(faces, np.array([True, False, False]))
    assert attached.tolist() == [True, True, False]
