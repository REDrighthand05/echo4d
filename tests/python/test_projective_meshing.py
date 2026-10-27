"""Synthetic projective-depth provider oracle; no real TUM performance claim."""
from argparse import Namespace
import json

import numpy as np
import pytest

from echo4d.model import DataError, file_sha256, load_recording
from echo4d.projective_meshing import projective_mesh, run_projective


def fixture(tmp_path, *, depth=None, mask=None, frames=1, color=True):
    depth = np.full((4, 4), 2., dtype=np.float64) if depth is None else np.asarray(depth, dtype=np.float64)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 120
    pose = np.eye(4)
    pose[0, 3] = .3
    entries = []
    for i in range(frames):
        entry = {"t": float(i), "cameraToWorld": pose.flatten(order="F").tolist(),
                 "depth": depth.ravel().tolist()}
        if color:
            entry["colors"] = rgb.ravel().tolist()
        if mask is not None:
            entry["excluded"] = np.asarray(mask, dtype=bool).ravel().tolist()
        entries.append(entry)
    path = tmp_path / "recording.json"
    path.write_text(json.dumps({
        "version": "echo-1", "name": "projective-test", "source": "synthetic",
        "units": "meters", "duration": float(frames), "intrinsics":
            {"width": 4, "height": 4, "fx": 100., "fy": 100., "cx": 1.5, "cy": 1.5},
        "frames": entries,
    }))
    return path


def test_plane_has_known_world_geometry_and_faces(tmp_path):
    recording = load_recording(fixture(tmp_path))
    mesh, deps = projective_mesh(recording, pixel_stride=1)
    assert len(mesh.vertices) == 16 and len(mesh.faces) == 18
    assert not deps
    assert np.allclose(mesh.vertices[:, 2], 2.)
    assert np.allclose(mesh.vertices[:, 0].mean(), .3)
    assert np.all(mesh.normals[:, 2] < -.99)
    assert np.all(mesh.colors[:, 0] == 120)
    assert set(mesh.observations.tolist()) == {1}


def test_two_frames_are_not_fused_or_invented(tmp_path):
    recording = load_recording(fixture(tmp_path, frames=2))
    mesh, _ = projective_mesh(recording, pixel_stride=1)
    assert len(mesh.vertices) == 32 and len(mesh.faces) == 36
    assert set(mesh.first_frame.tolist()) == {0, 1}
    assert np.array_equal(mesh.first_frame, mesh.last_frame)


def test_depth_discontinuity_never_bridged(tmp_path):
    depth = np.full((4, 4), 2.)
    depth[:, 2:] = 3.
    recording = load_recording(fixture(tmp_path, depth=depth))
    mesh, _ = projective_mesh(recording, pixel_stride=1, max_depth_jump=.1)
    assert len(mesh.faces) == 12
    assert all(np.ptp(mesh.vertices[face, 2]) < .1 for face in mesh.faces)


def test_missing_and_excluded_depth_not_meshed(tmp_path):
    depth = np.full((4, 4), 2.)
    depth[1, 1] = 0
    mask = np.zeros((4, 4), dtype=bool)
    mask[2, 2] = True
    mesh, _ = projective_mesh(load_recording(fixture(tmp_path, depth=depth, mask=mask)), pixel_stride=1)
    assert 0 < len(mesh.faces) < 18


def test_too_small_sampling_and_no_valid_faces_rejected(tmp_path):
    recording = load_recording(fixture(tmp_path))
    with pytest.raises(DataError, match="Sampling"):
        projective_mesh(recording, pixel_stride=4)
    with pytest.raises(DataError, match="No supported"):
        projective_mesh(recording, pixel_stride=1, min_depth=3, max_depth=4)


def test_resource_budget_fails_before_frame_loading(tmp_path):
    recording = load_recording(fixture(tmp_path, frames=2))
    with pytest.raises(DataError, match="upper bound"):
        projective_mesh(recording, pixel_stride=1, max_vertices=31)
    with pytest.raises(DataError, match="upper bound"):
        projective_mesh(recording, pixel_stride=1, max_faces=35)


@pytest.mark.parametrize("name,value", [
    ("pixel_stride", 0), ("frame_stride", True), ("max_vertices", -1),
    ("max_depth_jump", float("nan")), ("max_edge_length", 0),
    ("max_depth", 101),
])
def test_invalid_configuration_fails_closed(tmp_path, name, value):
    with pytest.raises(DataError):
        projective_mesh(load_recording(fixture(tmp_path)), **{name: value})


def test_output_receipt_and_ply_are_real_and_non_tsdf(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = fixture(source)
    out = tmp_path / "output"
    args = Namespace(recording=path, output=out,
                     expected_manifest_sha256=file_sha256(path),
                     pixel_stride=1, frame_stride=1, max_depth=6.,
                     max_depth_jump=.12, max_edge_length=.3)
    receipt = run_projective(args)
    assert receipt["status"] == "executed_unscored"
    assert receipt["fusion"] is False and receipt["quality"] == "not_evaluated"
    assert receipt["mesh_sha256"] == file_sha256(out / "mesh.ply")
    assert b"comment ECHO projective depth triangles" in (out / "mesh.ply").read_bytes()[:200]
    assert json.loads((out / "report.json").read_text())["mesh_sha256"] == receipt["mesh_sha256"]
    with pytest.raises(DataError, match="already exists"):
        run_projective(args)


def test_manifest_tampering_rejected_without_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = fixture(source)
    args = Namespace(recording=path, output=tmp_path / "output",
                     expected_manifest_sha256="0" * 64,
                     pixel_stride=1, frame_stride=1, max_depth=6.,
                     max_depth_jump=.12, max_edge_length=.3)
    with pytest.raises(DataError, match="frozen"):
        run_projective(args)
    assert not args.output.exists()


@pytest.mark.parametrize("defect", ["missing", "masked", "discontinuity"])
def test_unsampled_tile_defect_cannot_be_bridged(defect, tmp_path):
    depth = np.full((4, 4), 2.)
    mask = np.zeros((4, 4), dtype=bool)
    if defect == "missing":
        depth[1, 1] = 0.
    elif defect == "masked":
        mask[1, 1] = True
    else:
        depth[1, 1] = 3.
    recording = load_recording(fixture(tmp_path, depth=depth, mask=mask))
    mesh, _ = projective_mesh(recording, pixel_stride=2, max_depth_jump=.1)
    # One of two triangles remains; the triangle touching the unsampled
    # defective source tile must not be generated.
    assert len(mesh.faces) == 1


def test_sampling_stride_has_bounded_cpu_work(tmp_path):
    recording = load_recording(fixture(tmp_path))
    with pytest.raises(DataError, match="bounded sampling"):
        projective_mesh(recording, pixel_stride=17)
