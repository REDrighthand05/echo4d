"""The alternate TUM scene must fail closed on archive tampering and bad inputs."""
from __future__ import annotations
import io
import tarfile
from pathlib import Path
import pytest

from echo4d.model import DataError
import tum_desk_sfm as bench
from tum_desk_sfm import (ROOT, validate_archive_name, extract_selected,
                          get_archive, MAX_ARCHIVE_BYTES)


def test_valid_member_paths():
    assert validate_archive_name(f'{ROOT}/rgb/1.png', 'file', 10)[1:] == ('rgb', '1.png')
    assert validate_archive_name(f'{ROOT}/rgb.txt', 'file', 10)[1:] == ('rgb.txt',)


@pytest.mark.parametrize('name', ['/tmp/danger.txt', '../evil', f'{ROOT}/../escape',
                                   f'{ROOT}/rgb/../../escape', f'other_dataset/rgb/a.png'])
def test_reject_directory_traversal_or_wrong_scene(name):
    with pytest.raises(DataError):
        validate_archive_name(name, 'file', 10)


def test_reject_links_and_excessive_file():
    with pytest.raises(DataError):
        validate_archive_name(f'{ROOT}/rgb/1.png', 'link', 10)
    with pytest.raises(DataError):
        validate_archive_name(f'{ROOT}/rgb/1.png', 'file', 51*1024**2)


def _tar(path: Path, count: int = 600, evil: bool = False) -> None:
    with tarfile.open(path, 'w:gz') as t:
        for name in ('rgb.txt', 'groundtruth.txt'):
            data=b'# fake TUM index\n'
            info=tarfile.TarInfo(f'{ROOT}/{name}')
            info.size=len(data)
            t.addfile(info, io.BytesIO(data))
        for i in range(count):
            data=b'fake rgb bytes'
            info=tarfile.TarInfo(f'{ROOT}/rgb/{i:06}.png')
            info.size=len(data)
            t.addfile(info, io.BytesIO(data))
        if evil:
            info=tarfile.TarInfo(f'{ROOT}/../out.txt')
            info.size=1
            t.addfile(info,io.BytesIO(b'1'))


def test_extract_selected_ignores_depth_and_bounds(tmp_path):
    path=tmp_path/'in.tgz'
    _tar(path)
    result=extract_selected(path,tmp_path/'out')
    assert result['rgb_images_extracted']==600
    assert result['index_files_extracted']==2
    assert result['depth_images_extracted']==0
    assert (tmp_path/'out'/ROOT/'rgb'/'000000.png').is_file()
    assert not (tmp_path/'out'/ROOT/'depth').exists()


def test_archive_path_escape_not_extracted(tmp_path):
    path=tmp_path/'bad.tgz'
    _tar(path,count=2,evil=True)
    with pytest.raises(DataError,match='Invalid TUM archive member'):
        extract_selected(path,tmp_path/'out')
    assert not (tmp_path/'out'/'out.txt').exists()


def test_too_few_frames_fails(tmp_path):
    path=tmp_path/'short.tgz'
    _tar(path,count=3)
    with pytest.raises(DataError,match='Missing TUM RGB'):
        extract_selected(path,tmp_path/'out')


def test_offline_source_pin_is_checked_against_known_bytes(monkeypatch,tmp_path):
    path=tmp_path/'sample.tgz'
    _tar(path,count=600)
    from hashlib import sha256
    actual_sha = sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(bench,'PINNED_SHA256',actual_sha)
    monkeypatch.setattr(bench,'PINNED_BYTES',path.stat().st_size)
    rec={}
    get_archive(tmp_path/'saved.tgz',rec,path)
    assert rec['archive']['bytes']==(tmp_path/'saved.tgz').stat().st_size
    assert rec['archive']['sha256']==actual_sha
    assert rec['archive']['sha256_pinned_before_run'] is True
    assert 'matched independently pre-observed' in rec['archive']['source_integrity_claim']


def test_offline_missing_file_fails(tmp_path):
    with pytest.raises(DataError):
        get_archive(tmp_path/'saved.tgz',{},tmp_path/'not-there.tgz')


def test_pinned_archive_sha_mismatch_fails(monkeypatch,tmp_path):
    path=tmp_path/'different.tgz'
    _tar(path,count=600)
    monkeypatch.setattr(bench,'PINNED_SHA256','0'*64)
    monkeypatch.setattr(bench,'PINNED_BYTES',path.stat().st_size)
    with pytest.raises(DataError,match='differs from previously observed'):
        get_archive(tmp_path/'out.tgz',{},path)
