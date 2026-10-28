"""Frozen RGB-inferred local-normal gate: actual geometry, not sensor-Z tuning."""
import numpy as np
import pytest

from echo4d.known_pose_edge_orientation import filter_oriented_bridge_faces
from echo4d.model import Camera


def scene(raised_z=1.0):
    camera = Camera(width=64, height=48, fx=40., fy=40., cx=32., cy=24.)
    vertices = np.array([[0., 0., 1.], [.1, 0., 1.],
                         [0., .1, 1.], [.1, .1, raised_z]], dtype=np.float32)
    original = np.array([[0, 1, 2]], dtype=np.int32)
    candidate = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    depths = [np.ones((48, 64), dtype=np.float32) for _ in range(2)]
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = .06
    return vertices, original, candidate, depths, poses, camera


def test_flat_inferred_plane_preserves_supported_extension():
    vertices, original, candidate, depths, poses, camera = scene()
    keep, report = filter_oriented_bridge_faces(
        vertices, original, vertices, candidate, depths, poses, camera)
    assert keep.tolist() == [True, True]
    assert report["retained_extra_faces"] == 1
    assert report["sensor_depth_used"] is False


def test_steep_bridge_rejected_despite_matching_centroid_depth():
    vertices, original, candidate, depths, poses, camera = scene(1.18)
    keep, report = filter_oriented_bridge_faces(
        vertices, original, vertices, candidate, depths, poses, camera)
    assert keep.tolist() == [True, False]
    assert report["previous_extra_faces"] == 1
    assert report["rejected_by_orientation_or_attachment"] == 1


def test_same_camera_center_not_independent():
    vertices, original, candidate, depths, poses, camera = scene()
    poses[1][0, 3] = 0
    keep, _ = filter_oriented_bridge_faces(
        vertices, original, vertices, candidate, depths, poses, camera)
    assert keep.tolist() == [True, False]


def test_invalid_threshold_is_rejected():
    vertices, original, candidate, depths, poses, camera = scene()
    with pytest.raises(ValueError, match="threshold"):
        filter_oriented_bridge_faces(
            vertices, original, vertices, candidate, depths, poses, camera,
            normal_cosine=1.1)



def test_parallel_neighbour_depth_sheet_does_not_support_center_surface():
    from echo4d.known_pose_edge_orientation import _orientation_votes
    vertices, _, candidate, depths, poses, camera = scene()
    # Five-pixel cross: the central inferred Z agrees with the face, but all
    # neighbours belong to a different parallel sheet two metres behind it.
    for depth, pose in zip(depths, poses):
        center = vertices[candidate[1]].mean(axis=0)
        xyz = (center - pose[:3, 3]) @ pose[:3, :3]
        u = int(np.rint(camera.fx * xyz[0] / xyz[2] + camera.cx))
        v = int(np.rint(camera.fy * xyz[1] / xyz[2] + camera.cy))
        depth[:] = 3.
        depth[v, u] = 1.
    independent, _ = _orientation_votes(vertices, candidate[1:], depths, poses, camera)
    assert independent.tolist() == [False]


def test_consistent_sloped_patch_still_supports_local_plane():
    from echo4d.known_pose_edge_orientation import _orientation_votes
    vertices, _, candidate, depths, poses, camera = scene()
    vertices[:, 2] = 1. + .5 * vertices[:, 0]
    _, u = np.indices(depths[0].shape)
    for i, pose in enumerate(poses):
        depths[i] = ((1. + .5 * pose[0, 3]) /
                     (1. - .5 * (u - camera.cx) / camera.fx)).astype(np.float32)
    independent, _ = _orientation_votes(vertices, candidate[1:], depths, poses, camera)
    assert independent.tolist() == [True]


def test_oriented_replay_uses_actual_prior_mesh_identity_and_unavailable_counts(tmp_path, monkeypatch):
    pytest.importorskip('trimesh', reason='Dedicated real MVS workflow installs optional trimesh')
    import json
    from pathlib import Path
    from echo4d.fusion import Mesh
    from echo4d.model import file_sha256
    from echo4d.output import write_ply
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'tools'))
    import tum_mvs_oriented_bridges as tool
    vertices, original, candidate, depths, poses, camera = scene()
    old = tmp_path / 'old'
    (old / 'inputs/depth').mkdir(parents=True)
    frames = []
    for i in range(48):
        name = f'depth/{i:05d}.npy'
        np.save(old / 'inputs' / name, depths[i % 2])
        frames.append({'depth': name, 'cameraToWorld': poses[i % 2].flatten(order='F').tolist()})
    (old / 'inputs/recording.json').write_text(json.dumps({'frames': frames, 'units': 'meters', 'intrinsics': camera.to_dict()}))
    def fake_previous(old, source, prior):
        (prior / 'prior/unfiltered/reconstruction').mkdir(parents=True)
        (prior / 'reconstruction').mkdir()
        (prior / 'prior-comparison').mkdir()
        def mesh(faces):
            return Mesh(vertices, faces, np.tile([0., 0., 1.], (4, 1)).astype(np.float32), np.full((4, 3), 100, np.uint8), np.ones(4, np.uint32), np.zeros(4, np.int32), np.zeros(4, np.int32))
        write_ply(prior / 'prior/unfiltered/control-replayed.ply', mesh(original))
        write_ply(prior / 'prior/unfiltered/reconstruction/mesh.ply', mesh(candidate))
        write_ply(prior / 'reconstruction/mesh.ply', mesh(candidate))
        (prior / 'prior-comparison/report.json').write_text(json.dumps({'checkout_sha': 'STALE', 'mesh_sha256': 'WRONG_PRIOR_MESH', 'views': [{'name': 'frozen', 'pose': np.eye(4).flatten(order='F').tolist(), 'mesh_render_sha256': 'STALE_RENDER'}]}))
    monkeypatch.setattr(tool, 'previous_run', fake_previous)
    monkeypatch.setenv('GITHUB_SHA', 'CURRENT')
    target = tmp_path / 'new'
    tool.run(old, tmp_path, target)
    reference = json.loads((target / 'prior-comparison/report.json').read_text())
    report = json.loads((target / 'report.json').read_text())
    assert reference['checkout_sha'] == report['checkout_sha'] == 'CURRENT'
    assert reference['mesh_sha256'] == file_sha256(target / 'prior-comparison/reconstruction/mesh.ply')
    assert set(reference['views'][0]) == {'name', 'pose'}
    assert report['summary']['min_nearest_voxel_observations'] is None
    assert report['summary']['max_nearest_voxel_observations'] is None
