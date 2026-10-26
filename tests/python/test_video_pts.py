"""Tests for opt-in source-PTS mode, including real FFmpeg CFR/VFR clips."""
import json
import shutil
import subprocess

import pytest

from echo4d.model import DataError
from echo4d.video_pts import decoded_frame_pts, select_pts_frames, extract_source_pts
from echo4d.video_sfm import run_step


def _raw(values):
    return json.dumps({'frames': [
        {'best_effort_timestamp_time': str(t)} for t in values]})


def test_decoded_pts_uses_actual_irregular_intervals():
    times = decoded_frame_pts(_raw([5, 5.12, 5.48, 5.51, 6.7, 7.01]))
    stride, indices = select_pts_frames(times, 3)
    assert stride == 2 and indices == [0, 2, 4]
    assert [times[i] - times[0] for i in indices] == pytest.approx(
        [0, .48, 1.7])


@pytest.mark.parametrize('values', [
    [0, 0, .2], [0, .2, .1], [-.1, 0, .2],
    [0, float('inf'), 1], [0, float('nan'), 1],
])
def test_decoded_pts_rejects_unsafe_timeline(values):
    with pytest.raises(DataError, match='PTS'):
        decoded_frame_pts(_raw(values))


def test_decoded_pts_rejects_missing_fields_and_oversize():
    with pytest.raises(DataError, match='Missing'):
        decoded_frame_pts('{"frames":[{}, {}, {}]}')
    with pytest.raises(DataError, match='8MB'):
        decoded_frame_pts('x' * 8_000_001)


def test_pts_budget_fails_closed_without_three_frames():
    with pytest.raises(DataError, match='3-100000 decoded frames'):
        select_pts_frames([0, 1], 3)
    with pytest.raises(DataError, match='max_frames'):
        select_pts_frames([0, 1, 2], 2)


def test_ffmpeg_output_mismatch_is_rejected(tmp_path):
    source = tmp_path / 'fake.mp4'
    source.touch()
    calls = []

    def fake_step(argv, receipt, label, timeout):
        calls.append(label)
        if label == 'probe-decoded-frame-pts':
            return _raw([0, .1, .4, 1, 1.2, 2])
        return ''

    with pytest.raises(DataError, match='count/names mismatch'):
        extract_source_pts(source, tmp_path / 'images', 3, {}, fake_step)
    assert calls == ['probe-decoded-frame-pts', 'extract-frames-source-pts']


@pytest.mark.skipif(not (shutil.which('ffmpeg') and shutil.which('ffprobe')),
                    reason='Requires real FFmpeg and FFprobe')
@pytest.mark.parametrize('mode', ['cfr', 'vfr'])
def test_real_video_pts_tracks_selected_decoded_frames(tmp_path, mode):
    # This test probes/extracts actual encoded MP4, not mocked frames.
    from PIL import Image
    source = tmp_path / 'rgb'
    source.mkdir()
    for i in range(8):
        Image.new('RGB', (80, 64), (i * 30, i * 17, i * 11)).save(
            source / f'{i:03d}.png')
    video = tmp_path / (mode + '.mp4')
    common = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y']
    if mode == 'cfr':
        cmd = common + ['-framerate', '4', '-i', str(source / '%03d.png')]
    else:
        filelist = source / 'timestamps.ffconcat'
        durations = [.08, .32, .16, .40, .12, .36, .18]
        lines = ['ffconcat version 1.0']
        for i in range(8):
            lines.append(f"file '{source / f'{i:03d}.png'}'")
            if i < 7:
                lines.append(f'duration {durations[i]}')
        filelist.write_text('\n'.join(lines) + '\n')
        cmd = common + ['-safe', '0', '-f', 'concat', '-i', str(filelist),
                        '-fps_mode', 'vfr']
    subprocess.run(cmd + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                          str(video)], check=True, capture_output=True)
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'frame=best_effort_timestamp_time', '-of',
         'json', str(video)],
        check=True, capture_output=True, text=True)
    all_pts = decoded_frame_pts(probe.stdout)
    assert len(all_pts) == 8
    receipt = {'commands': []}
    mapping = extract_source_pts(video, tmp_path / 'out', 4, receipt, run_step)
    assert list(mapping) == [f'frame_{i:05d}.png' for i in range(4)]
    assert list(mapping.values()) == pytest.approx(
        [all_pts[i] - all_pts[0] for i in [0, 2, 4, 6]])
    assert receipt['source_pts_sampling']['stride'] == 2
    assert all(x['status'] == 'passed' for x in receipt['commands'])
    gaps = [b - a for a, b in zip(mapping.values(),
                                  list(mapping.values())[1:])]
    if mode == 'vfr':
        assert max(gaps) - min(gaps) > .05
    else:
        assert max(gaps) - min(gaps) < 1e-8


def test_video_sfm_pts_mode_keeps_existing_timestamp_interface(tmp_path, monkeypatch):
    from echo4d import video_sfm

    def fake_probe(video, receipt):
        return {'width': 80, 'height': 64, 'duration_seconds': 3}

    def fake_source_pts(video, images, max_frames, receipt, runner):
        assert max_frames == 12
        receipt['source_pts_sampling'] = {'mode': 'source-pts'}
        return {'frame_00000.png': 0, 'frame_00001.png': .37,
                'frame_00002.png': .84}

    monkeypatch.setattr(video_sfm, 'probe_video', fake_probe)
    monkeypatch.setattr(video_sfm, 'extract_source_pts', fake_source_pts)
    receipt = {}
    stamps = video_sfm.extract_video(tmp_path / 'unused.mp4', tmp_path / 'frames',
                                    4., 12, receipt, 'source-pts')
    assert stamps == {'frame_00000.png': 0, 'frame_00001.png': .37,
                      'frame_00002.png': .84}
    assert receipt['source_pts_sampling']['mode'] == 'source-pts'
