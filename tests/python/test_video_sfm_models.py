"""PR #13's useful disconnected-model/failure cases against PR #12's format.

No synthetic fixture in this file is evidence of real video motion accuracy.
"""
from pathlib import Path

import numpy as np
import pytest

from echo4d.model import DataError
from echo4d import video_sfm


def _fixture(tmp_path, monkeypatch, candidates, *, broken=(), camera_ids=None,
             bad_rotation=()):
    sparse, output = tmp_path / 'sparse', tmp_path / 'output'
    sparse.mkdir()
    output.mkdir()
    names = [f'frame_{i:05d}.png' for i in range(6)]
    timestamps = {name: float(i) for i, name in enumerate(names)}
    for model_id in candidates:
        model = sparse / str(model_id)
        model.mkdir()
        (model / 'images.bin').write_bytes(b'fixture')
    camera_ids = camera_ids or {}
    def converter(command, report, label, timeout=90):
        model_id = int(Path(command[command.index('--input_path') + 1]).name)
        if model_id in broken:
            raise DataError('intentional model converter failure')
        dest = Path(command[command.index('--output_path') + 1])
        cid = camera_ids.get(model_id, 1)
        (dest / 'cameras.txt').write_text(
            '1 SIMPLE_RADIAL 640 480 525 319.5 239.5 0.01\n')
        lines = []
        for i, name in enumerate(candidates[model_id]):
            q = '0 0 0 0' if model_id in bad_rotation else '1 0 0 0'
            lines.append(f'{i + 1} {q} {-i} 0 0 {cid} {name}\n\n')
        (dest / 'images.txt').write_text(''.join(lines))
        (dest / 'points3D.txt').write_text('')
        return ''
    monkeypatch.setattr(video_sfm, 'run_step', converter)
    return sparse, output, timestamps


def test_selects_largest_valid_model_and_keeps_pr12_pose_layout(tmp_path, monkeypatch):
    sparse, out, stamps = _fixture(tmp_path, monkeypatch,
        {0: ['frame_00000.png', 'frame_00001.png', 'frame_00002.png'],
         1: ['frame_00000.png', 'frame_00002.png', 'frame_00003.png', 'frame_00004.png']})
    receipt = {}
    model, frames, count = video_sfm.select_sparse_model(sparse, out, stamps, receipt)
    assert (model.name, count) == ('text-model-1', 4)
    assert receipt['selected_model_id'] == '1'
    assert [e['status'] for e in receipt['model_candidates']] == ['valid', 'valid']
    assert set(frames[0]) == {'name', 't_s', 'camera_id', 'camera_to_sfm_world'}
    assert len(frames[0]['camera_to_sfm_world']) == 16
    pose = np.array(frames[0]['camera_to_sfm_world']).reshape((4, 4), order='F')
    assert np.array_equal(pose[3], [0, 0, 0, 1])


def test_bad_first_model_isolated_and_preserved(tmp_path, monkeypatch):
    sparse, out, stamps = _fixture(tmp_path, monkeypatch,
        {0: ['frame_00000.png', 'frame_00001.png', 'frame_00002.png'],
         1: ['frame_00000.png', 'frame_00001.png', 'frame_00002.png']},
        broken=(0,))
    receipt = {}
    model, _, count = video_sfm.select_sparse_model(sparse, out, stamps, receipt)
    assert model.name == 'text-model-1' and count == 3
    assert receipt['model_candidates'][0]['status'] == 'rejected'
    assert 'intentional' in receipt['model_candidates'][0]['reason']


def test_unknown_frame_and_bad_camera_rejected_independently(tmp_path, monkeypatch):
    sparse, out, stamps = _fixture(tmp_path, monkeypatch,
        {0: ['frame_00000.png', 'other.png', 'frame_00002.png'],
         1: ['frame_00000.png', 'frame_00001.png', 'frame_00002.png']},
        camera_ids={1: 9})
    receipt = {}
    with pytest.raises(DataError, match='No valid COLMAP model'):
        video_sfm.select_sparse_model(sparse, out, stamps, receipt)
    assert len(receipt['model_candidates']) == 2
    assert 'missing or duplicated' in receipt['model_candidates'][0]['reason']
    assert 'missing camera' in receipt['model_candidates'][1]['reason']


def test_tie_uses_smallest_numeric_id(tmp_path, monkeypatch):
    sparse, out, stamps = _fixture(tmp_path, monkeypatch,
        {10: ['frame_00000.png', 'frame_00001.png', 'frame_00002.png'],
         2: ['frame_00000.png', 'frame_00001.png', 'frame_00002.png']})
    receipt = {}
    model, _, _ = video_sfm.select_sparse_model(sparse, out, stamps, receipt)
    assert model.name == 'text-model-2'
    assert [e['model_id'] for e in receipt['model_candidates']] == ['2', '10']


def test_insufficient_cameras_cannot_pass(tmp_path, monkeypatch):
    sparse, out, stamps = _fixture(tmp_path, monkeypatch,
        {0: ['frame_00000.png', 'frame_00001.png']})
    receipt = {}
    with pytest.raises(DataError, match='No valid COLMAP model'):
        video_sfm.select_sparse_model(sparse, out, stamps, receipt)
    assert receipt['model_candidates'][0]['status'] == 'rejected'
    assert 'fewer than three' in receipt['model_candidates'][0]['reason']


def test_invalid_rotation_is_not_published(tmp_path, monkeypatch):
    sparse, out, stamps = _fixture(tmp_path, monkeypatch,
        {0: ['frame_00000.png', 'frame_00001.png', 'frame_00002.png']},
        bad_rotation=(0,))
    receipt = {}
    with pytest.raises(DataError, match='No valid COLMAP model'):
        video_sfm.select_sparse_model(sparse, out, stamps, receipt)
    assert receipt['model_candidates'][0]['status'] == 'rejected'


def test_too_many_disconnected_models_fail_closed(tmp_path, monkeypatch):
    sparse, out, stamps = _fixture(tmp_path, monkeypatch, {i: [] for i in range(33)})
    with pytest.raises(DataError, match='more than 32'):
        video_sfm.select_sparse_model(sparse, out, stamps, {})


@pytest.mark.parametrize('bad', [
    '1 SIMPLE_RADIAL 640 480 -1 319.5 239.5 0.01\n',
    '1 PINHOLE 640 480 525 -1 319.5 239.5\n',
    '1 SIMPLE_RADIAL 640 480 nan 319.5 239.5 0.01\n',
    '1 PINHOLE 640 480 525 525 319.5 239.5\n1 PINHOLE 640 480 525 525 319.5 239.5\n',
])
def test_bad_intrinsics_rejected(tmp_path, bad):
    file = tmp_path / 'cameras.txt'
    file.write_text(bad)
    with pytest.raises(DataError):
        video_sfm._model_camera_ids(file)


def test_existing_distorted_sfm_default_is_not_silently_disabled(tmp_path):
    file = tmp_path / 'cameras.txt'
    file.write_text('1 SIMPLE_RADIAL 640 480 525 319.5 239.5 0.01\n')
    assert video_sfm._model_camera_ids(file) == {1}
