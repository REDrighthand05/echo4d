"""Geometry tests: new faces need two independent inferred-Z observations."""
import numpy as np
import pytest

from echo4d.model import Camera
from echo4d.known_pose_edge_crossview import filter_crossview_faces


def _case(z=1., offset=.06, observed=True):
    vertices = np.array([[0., 0., 1.], [.10, 0., 1.],
                         [0., .10, 1.], [.10, .10, z]], dtype=np.float32)
    original = np.array([[0, 1, 2]], dtype=np.int32)
    candidate = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    camera = Camera(width=64, height=48, fx=40., fy=40., cx=32., cy=24.)
    depth = np.full((48, 64), 1. if observed else 0., dtype=np.float32)
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = offset
    return vertices, original, candidate, [depth, depth.copy()], poses, camera


def _run(z=1., offset=.06, observed=True):
    v, old, new, depths, poses, k = _case(z, offset, observed)
    return filter_crossview_faces(v, old, v, new, depths, poses, k)


def test_two_distinct_matching_inferred_views_extend_original_surface():
    keep, stats = _run()
    assert keep.tolist() == [True, True]
    assert stats["baseline_faces_preserved"] == 1
    assert stats["extra_faces_retained_after_connectivity"] == 1
    assert not stats["sensor_depth_used"]


def test_same_camera_center_cannot_vote_twice():
    keep, stats = _run(offset=0.)
    assert keep.tolist() == [True, False]
    assert stats["extra_faces_with_two_independent_views"] == 0


def test_unsupported_or_wrong_depth_preserves_original_only():
    keep, stats = _run(observed=False)
    assert keep.tolist() == [True, False]
    assert stats["extra_faces_rejected"] == 1
    keep, _ = _run(z=1.8)
    assert keep.tolist() == [True, False]


def test_missing_baseline_face_is_rejected():
    v, old, new, depths, poses, k = _case()
    with pytest.raises(ValueError, match="lost original"):
        filter_crossview_faces(v, old, v, new[1:], depths, poses, k)


def test_invalid_inferred_depth_is_rejected():
    v, old, new, depths, poses, k = _case()
    depths[0][1, 1] = np.nan
    with pytest.raises(ValueError, match="finite inferred Z"):
        filter_crossview_faces(v, old, v, new, depths, poses, k)



@pytest.mark.parametrize('order', [(0, 1, 2), (1, 0, 2), (2, 0, 1)])
def test_any_distinct_support_pair_counts_regardless_of_first_view(order):
    v, old, new, depths, poses, k = _case(offset=.03)
    third = np.eye(4)
    third[0, 3] = -.03
    poses.append(third)
    depths.append(depths[0].copy())
    keep, stats = filter_crossview_faces(v, old, v, new,
        [depths[i] for i in order], [poses[i] for i in order], k)
    # The center view is <4cm from each side, but the two sides are 6cm apart.
    assert keep.tolist() == [True, True]
    assert stats['extra_faces_with_two_independent_views'] == 1


def test_replay_report_uses_current_identity_and_no_placeholder_observation_counts(tmp_path, monkeypatch):
    pytest.importorskip('trimesh', reason='Dedicated real MVS workflow installs optional trimesh')
    import json
    from pathlib import Path
    from echo4d.fusion import Mesh
    from echo4d.model import file_sha256
    from echo4d.output import write_ply
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'tools'))
    import tum_mvs_edge_crossview as tool
    v, original, candidate, depths, poses, camera = _case()
    old = tmp_path / 'old'
    (old / 'inputs/depth').mkdir(parents=True)
    entries = []
    for i in range(48):
        name = f'depth/{i:05d}.npy'
        np.save(old / 'inputs' / name, depths[i % 2])
        entries.append({'depth': name, 'cameraToWorld': poses[i % 2].flatten(order='F').tolist()})
    (old / 'inputs/recording.json').write_text(json.dumps({'frames': entries, 'units': 'meters', 'intrinsics': camera.to_dict()}))
    (old / 'report.json').write_text(json.dumps({'checkout_sha': 'OLD', 'mesh_sha256': 'OLD_MESH', 'views': [{'name': 'frozen', 'pose': np.eye(4).flatten(order='F').tolist(), 'mesh_render_sha256': 'OLD_RENDER'}]}))
    def fake_replay(old, source_root, base):
        (base / 'reconstruction').mkdir(parents=True)
        def mesh(faces):
            return Mesh(v, faces, np.tile([0., 0., 1.], (4, 1)).astype(np.float32), np.full((4, 3), 100, np.uint8), np.ones(4, np.uint32), np.zeros(4, np.int32), np.zeros(4, np.int32))
        write_ply(base / 'control-replayed.ply', mesh(original))
        write_ply(base / 'reconstruction/mesh.ply', mesh(candidate))
        (base / 'report.json').write_text(json.dumps({'control_replay': {'replayed_sha256': file_sha256(base / 'control-replayed.ply')}}))
    monkeypatch.setattr(tool, 'replay', fake_replay)
    monkeypatch.setenv('GITHUB_SHA', 'CURRENT')
    target = tmp_path / 'new'
    tool.run(old, tmp_path, target)
    report = json.loads((target / 'report.json').read_text())
    reference = json.loads((target / 'unfiltered-comparison.json').read_text())
    assert report['checkout_sha'] == reference['checkout_sha'] == 'CURRENT'
    assert reference['mesh_sha256'] == file_sha256(target / 'unfiltered/reconstruction/mesh.ply')
    assert 'mesh_render_sha256' not in reference['views'][0]
    assert report['new_mesh_summary']['min_nearest_voxel_observations'] is None
    assert report['new_mesh_summary']['max_nearest_voxel_observations'] is None
