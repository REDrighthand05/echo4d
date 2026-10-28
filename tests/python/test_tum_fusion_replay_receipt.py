"""Replay lineage must describe new saved bytes, not copied old run metadata."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from echo4d.model import file_sha256


def test_replay_receipt_tracks_new_inputs_and_checkout(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'tools'))
    import tum_plane_sweep_fusion_replay as replay
    root = tmp_path / 'dataset' / replay.ROOT
    root.mkdir(parents=True)
    source = root / 'rgb.png'
    source.write_bytes(b'fixed source identity')
    for name in ['rgb.txt', 'depth.txt', 'groundtruth.txt']:
        (root / name).write_bytes(b'fixed raw index\r\n')
    frames = [{'index': i, 't_s': float(i), 'pose': np.eye(4), 'rgb': source,
               'rgb_sha256': file_sha256(source)} for i in range(60)]
    old = tmp_path / 'old'
    (old / 'inputs/depth').mkdir(parents=True)
    entries = []
    for i in range(60):
        if i in replay.HOLDOUT:
            continue
        n = len(entries)
        name = f'depth/{n:05d}.npy'
        np.save(old / 'inputs' / name, np.ones((240, 320), np.float32))
        entries.append({'t': float(i), 'depth': name, 'color': f'color/{n:05d}.png',
                        'cameraToWorld': np.eye(4).flatten(order='F').tolist()})
    manifest = {'frames': entries, 'intrinsics': {'width': 320}}
    (old / 'inputs/recording.json').write_text(json.dumps(manifest))
    (old / 'report.json').write_text(json.dumps({'checkout_sha': 'OLD', 'input': {},
        'mesh_sha256': 'OLDMESH', 'inferred_manifest_sha256': 'OLDINPUT',
        'views': ['OLDVIEW'], 'mesh_output': {'old': True}, 'archived_inputs': {}}))
    monkeypatch.setenv('GITHUB_SHA', 'NEW')
    monkeypatch.setattr(replay, 'rgb_pose_frames', lambda root: frames)
    monkeypatch.setattr(replay, 'load_stereo_rgb', lambda *a: np.zeros((240, 320, 3), np.uint8))
    def fuse(depths, *a, **k):
        return [d + np.float32(.01) for d in depths], [
            {'added_pixels': 0, 'adjusted_pixels': d.size} for d in depths]
    monkeypatch.setattr(replay, 'fuse_reprojected_depths', fuse)
    def reconstruct(cmd, **kwargs):
        out = Path(cmd[cmd.index('--output') + 1])
        out.mkdir()
        (out / 'mesh.ply').write_bytes(b'NEW MESH')
        return SimpleNamespace(returncode=0, stdout=json.dumps({'frames': 48})+'\n', stderr='')
    monkeypatch.setattr(replay.subprocess, 'run', reconstruct)
    monkeypatch.setattr(replay, 'render_mesh', lambda mesh, frames, folder, report, camera:
                        report.update(views=['NEWVIEW']))
    target = tmp_path / 'new'
    replay.replay(old, tmp_path / 'dataset', target)
    report = json.loads((target / 'report.json').read_text())
    assert report['checkout_sha'] == 'NEW'
    assert report['inference_source_checkout_sha'] == 'OLD'
    assert report['views'] == ['NEWVIEW']
    assert report['mesh_output'] == {'frames': 48}
    assert report['inferred_manifest_sha256'] == file_sha256(target / 'inputs/recording.json')
    for name, receipt in report['archived_inputs'].items():
        assert receipt['sha256'] == file_sha256(target / 'inputs' / name)
