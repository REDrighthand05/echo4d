"""Issue #47/B semantic merge gate: PR #37 PTS + PR #40 COLMAP models.

All COLMAP activity in these tests is explicitly mocked. Real FFmpeg/COLMAP
checks have separate TUM workflows, and this fixture cannot prove pose quality.
"""
import csv
import json
from argparse import Namespace

import pytest

from echo4d import video_sfm
from echo4d.model import DataError


def _args(root):
    return Namespace(
        video=root / 'fixture.mp4', images=None, timestamps=None,
        output=root / 'result', fps=4.0, max_frames=3,
        timestamp_mode='source-pts', camera_model='PINHOLE',
        camera_params='80,80,32,24')


def _fake_pipeline(tmp_path, monkeypatch, *, good_model=True):
    def sample_pts(video, images, fps, max_frames, receipt, timestamp_mode):
        assert timestamp_mode == 'source-pts'
        assert fps == 4.0 and max_frames == 3
        images.mkdir()
        receipt['source_pts_sampling'] = {
            'mode': 'source-pts',
            'frame_map': [
                {'name': f'frame_{i:05d}.png',
                 'decoded_frame_index': 2*i,
                 'source_pts_seconds': v + 12,
                 'relative_pts_seconds': v}
                for i, v in enumerate([0.0, 0.37, 1.02])]}
        return {f'frame_{i:05d}.png': v
                for i, v in enumerate([0.0, 0.37, 1.02])}

    def fake_colmap(argv, receipt, label, timeout=900):
        receipt.setdefault('commands', []).append({
            'stage': label, 'status': 'passed', 'command': argv})
        output = tmp_path / 'result'
        if label == 'colmap-sparse-mapper':
            for index in (0, 1):
                model = output / 'sparse' / str(index)
                model.mkdir()
                (model / 'images.bin').write_bytes(b'fixture only')
        elif label in ('colmap-text-0', 'colmap-text-1'):
            model = output / ('text-model-0' if label.endswith('0')
                               else 'text-model-1')
            if label.endswith('0') or not good_model:
                (model / 'cameras.txt').write_text(
                    '1 PINHOLE 64 48 -80 80 32 24\n')
            else:
                (model / 'cameras.txt').write_text(
                    '1 PINHOLE 64 48 80 80 32 24\n')
            rows = []
            for i in range(3):
                rows.extend([
                    f'{i+1} 1 0 0 0 {-i} 0 0 1 frame_{i:05d}.png',
                    ''])
            (model / 'images.txt').write_text('\n'.join(rows) + '\n')
            (model / 'points3D.txt').write_text('# empty sparse model fixture\n')
        return ''

    monkeypatch.setattr(video_sfm, 'extract_video', sample_pts)
    monkeypatch.setattr(video_sfm, 'run_step', fake_colmap)


def test_pts_reaches_multi_model_track_without_changing_metric_bridge_contract(
        tmp_path, monkeypatch):
    _fake_pipeline(tmp_path, monkeypatch)
    report = video_sfm.run_pipeline(_args(tmp_path))
    assert report['status'] == 'passed_rgb_only_sfm_not_metric_geometry'
    assert report['selected_model_id'] == '1'
    assert [x['status'] for x in report['model_candidates']] == ['rejected', 'valid']
    assert report['result']['sfm_model_dir'] == 'text-model-1'
    assert 'source_pts_sampling' in report
    assert report['result']['frames_registered'] == 3

    out = tmp_path / 'result'
    trajectory = json.loads((out / 'trajectory-scale-unknown.json').read_text())
    assert trajectory['version'] == 'echo-sfm-trajectory-1'
    assert trajectory['units'] == 'unknown'
    assert trajectory['metric_scale_estimated'] is False
    assert trajectory['depth_estimated'] is False
    assert set(trajectory['frames'][0]) == {
        'name', 't_s', 'camera_id', 'camera_to_sfm_world'}
    assert len(trajectory['frames'][0]['camera_to_sfm_world']) == 16
    assert [x['t_s'] for x in trajectory['frames']] == pytest.approx(
        [0.0, .37, 1.02])
    assert (out / report['result']['sfm_model_dir'] / 'images.txt').is_file()
    assert (out / report['result']['sfm_model_dir'] / 'cameras.txt').is_file()
    assert trajectory['model_images_txt_sha256'] == report['result']['model_images_txt_sha256']
    with (out / 'registered_timestamps.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ['name', 't']
    assert [float(row['t']) for row in rows] == pytest.approx([0, .37, 1.02])
    with (out / 'all_input_timestamps.csv').open(newline='') as f:
        all_rows = list(csv.DictReader(f))
    assert len(all_rows) == 3


def test_no_legal_colmap_model_fails_and_does_not_publish_pose(
        tmp_path, monkeypatch):
    _fake_pipeline(tmp_path, monkeypatch, good_model=False)
    report = video_sfm.run_pipeline(_args(tmp_path))
    assert report['status'] == 'failed'
    assert 'No valid COLMAP model' in report['error']['message']
    assert [x['status'] for x in report['model_candidates']] == [
        'rejected', 'rejected']
    out = tmp_path / 'result'
    assert (out / 'report.json').is_file()
    assert not (out / 'trajectory-scale-unknown.json').exists()
    assert not (out / 'registered_timestamps.csv').exists()


def test_pts_mode_does_not_silently_apply_to_images_csv(tmp_path):
    frames = tmp_path / 'frames'
    frames.mkdir()
    csv_path = tmp_path / 'times.csv'
    csv_path.write_text('name,t\na.png,0\nb.png,1\nc.png,2\n')
    with pytest.raises(SystemExit) as result:
        video_sfm.main([
            '--images', str(frames), '--timestamps', str(csv_path),
            '--output', str(tmp_path / 'wrong'),
            '--timestamp-mode', 'source-pts'])
    assert result.value.code == 2
    assert not (tmp_path / 'wrong').exists()


def test_invalid_mode_cannot_bypass_video_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(video_sfm, 'probe_video',
                        lambda *_: {'width': 64, 'height': 48, 'duration_seconds': 3})
    with pytest.raises(DataError, match='Unknown video timestamp mode'):
        video_sfm.extract_video(tmp_path / 'in.mp4', tmp_path / 'images',
                                4.0, 3, {}, 'silent-fallback')
