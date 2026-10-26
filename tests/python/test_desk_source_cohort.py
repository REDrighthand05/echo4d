"""Known-answer source availability tests; no network, depth pixels or solver."""
import argparse
import hashlib
import io
from pathlib import Path
import shutil
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from desk_source_cohort import extract_time_indexes, plan_cohort
from echo4d.model import DataError
import tum_desk_full_mesh as runner
from tum_desk_sfm import ROOT, prepare_video


def indexes(root, count=613, missing=(0,)):
    root.mkdir(parents=True)
    rgb = ''.join(f'{i * .1:.6f} rgb/{i:06d}.png\n' for i in range(count))
    depth = ''.join(f'{i * .1 + .01:.6f} depth/{i:06d}.png\n'
                    for i in range(count) if i not in missing)
    (root / 'rgb.txt').write_text(rgb)
    (root / 'depth.txt').write_text(depth)
    return rgb, depth


def test_v1_keeps_first_missing_source_and_v2_uses_first_legal_schedule(tmp_path):
    indexes(tmp_path / 'indexes')
    original = plan_cohort(tmp_path / 'indexes', 'v1-frozen')
    new = plan_cohort(tmp_path / 'indexes', 'v2-source-paired')
    assert [x['source_rgb_index'] for x in original['selected_frames']] == list(range(0, 600, 8))
    assert original['selected_frames'][0]['source_depth_path'] is None
    assert [x['source_rgb_index'] for x in new['selected_frames']] == list(range(1, 601, 8))
    assert new['rejected_before_selection'] == [
        {'source_rgb_index': 0, 'rgb': 'rgb/000000.png',
         'original_rgb_time_s': 0., 'reason': 'no_depth_within_30ms'}]
    assert new['selection_complete'] is True
    assert all(abs(x['timestamp_delta_seconds'] - .01) < 1e-10 for x in new['selected_frames'])
    assert new['depth_pixels_or_GT_read_for_selection'] is False
    assert not list((tmp_path / 'indexes').glob('*.png'))


def test_too_few_legal_pairs_is_explicit_not_a_smaller_experiment(tmp_path):
    indexes(tmp_path / 'indexes', count=590)
    plan = plan_cohort(tmp_path / 'indexes', 'v2-source-paired')
    assert len(plan['selected_frames']) == 74
    assert plan['selection_complete'] is False
    assert plan['required_frames'] == 75


@pytest.mark.parametrize('bad', ['nan', '1.0', '-1'])
def test_invalid_or_nonmonotonic_timestamps_rejected(tmp_path, bad):
    indexes(tmp_path / 'indexes')
    with (tmp_path / 'indexes/depth.txt').open('a') as stream:
        stream.write(f'{bad} depth/999999.png\n')
    with pytest.raises(DataError, match='timestamps must strictly increase'):
        plan_cohort(tmp_path / 'indexes', 'v2-source-paired')


def test_duplicate_original_depth_identity_is_rejected(tmp_path):
    indexes(tmp_path / 'indexes')
    path = tmp_path / 'indexes/depth.txt'
    path.write_text(path.read_text().replace('depth/000002.png', 'depth/000001.png'))
    with pytest.raises(DataError, match='reused depth'):
        plan_cohort(tmp_path / 'indexes', 'v2-source-paired')


def archive(path, rgb, depth, extra=None):
    with tarfile.open(path, 'w:gz') as output:
        for name, text in [('rgb.txt', rgb), ('depth.txt', depth)]:
            data = text.encode()
            member = tarfile.TarInfo(ROOT + '/' + name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
        if extra is not None:
            member = tarfile.TarInfo(extra)
            member.size = 1
            output.addfile(member, io.BytesIO(b'x'))


def test_archive_scanner_only_materializes_original_indexes(tmp_path):
    rgb, depth = indexes(tmp_path / 'indexes')
    archive(tmp_path / 'source.tgz', rgb, depth, ROOT + '/depth/unread-pixels.png')
    hashes = extract_time_indexes(tmp_path / 'source.tgz', tmp_path / 'out')
    assert set(x.name for x in (tmp_path / 'out').iterdir()) == {'rgb.txt', 'depth.txt'}
    assert hashes['depth.txt'] == hashlib.sha256(depth.encode()).hexdigest()


def test_archive_scanner_rejects_traversal_even_for_unselected_members(tmp_path):
    rgb, depth = indexes(tmp_path / 'indexes')
    archive(tmp_path / 'source.tgz', rgb, depth, ROOT + '/../escape.png')
    with pytest.raises(DataError, match='Invalid TUM archive member'):
        extract_time_indexes(tmp_path / 'source.tgz', tmp_path / 'out')


def test_failed_v1_pairing_never_encodes_or_runs_sfm(tmp_path, monkeypatch):
    rgb, depth = indexes(tmp_path / 'indexes')
    source = tmp_path / 'source.tgz'
    archive(source, rgb, depth)
    monkeypatch.setattr(runner, 'PINNED_BYTES', source.stat().st_size)
    monkeypatch.setattr(runner, 'PINNED_SHA256', hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(runner, 'get_archive', lambda target, record, offline: shutil.copyfile(source, target))
    def forbidden(*args, **kwargs):
        raise AssertionError('Expensive processing must not run on an impossible frozen cohort')
    monkeypatch.setattr(runner, 'prepare_video', forbidden)
    monkeypatch.setattr(runner, 'extract_selected', forbidden)
    monkeypatch.setattr(runner, 'invoke', forbidden)
    report = tmp_path / 'report'
    report.mkdir()
    args = argparse.Namespace(work=tmp_path / 'work', report_dir=report,
                              archive=source, cohort='v1-frozen')
    with pytest.raises(DataError, match='before encoding/SfM'):
        runner.run(args, {})
    assert (report / 'source-cohort-plan.json').is_file()
    assert (report / 'sensor-pairing-preflight.json').is_file()


def test_explicit_encoding_cannot_replace_original_source_identity(tmp_path):
    indexes(tmp_path / 'indexes', missing=())
    chosen = [['0.0', 'rgb/000000.png']] * 75
    with pytest.raises(DataError, match='original RGB index'):
        prepare_video(tmp_path / 'indexes', tmp_path, {}, chosen_rows=chosen)


def test_audit_reports_original_rgb_index_not_renumbered_cohort_position(tmp_path):
    indexes(tmp_path / 'indexes')
    source = plan_cohort(tmp_path / 'indexes', 'v2-source-paired')['selected_frames']
    audit = runner.sensor_pairing_diagnostics(tmp_path / 'indexes', source,
                                               runner.frozen_protocol(source))
    assert audit['pairs'][0]['source_frame_index'] == 0
    assert audit['pairs'][0]['source_rgb_index'] == 1
    assert audit['all_pairs_within_gap_and_unique']
