"""Cross-scene **real RGB-only** COLMAP SfM test on TUM Freiburg1 Desk.

Intended as an independent contrast with the previous Freiburg1 XYZ result.
TUM sensor depths and true poses never enter COLMAP. Poses are loaded only
AFTER it has returned a valid estimated track, and Sim(3) is evaluation-only.
The source archive SHA is observed on first run then pinned before replication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from echo4d.model import DataError, file_sha256
from echo4d.sfm_eval import evaluate_sfm_poses
from tum_public_roundtrip import invoke
from tum_rgb_sfm import VIDEO_FPS, SOURCE_STRIDE, gt_after_sfm
from echo4d.tum import read_rows
from echo4d.model import safe_path

FRAME_COUNT = 75  # 613-frame desk sequence: indices 0,8,...,592

URL = 'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz'
ROOT = 'rgbd_dataset_freiburg1_desk'
# Run 1 observes source bytes. Once observed they MUST be pinned and used
# for run 2: never silently treat the first observed source as upstream-signed.
PINNED_SHA256: str | None = None
PINNED_BYTES: int | None = None
MAX_ARCHIVE_BYTES = 700 * 1024**2
MAX_UNPACKED_BYTES = 3 * 1024**3
ALLOWED_HOSTS = frozenset({'cvg.cit.tum.de', 'webshare.cvg.cit.tum.de'})


def validate_archive_name(name: str, kind: str, size: int) -> tuple[str, ...]:
    """Reject links and path traversal, including in members we do not extract."""
    path = PurePosixPath(name)
    parts = path.parts
    if (path.is_absolute() or not parts or parts[0] != ROOT
            or '..' in parts or not kind in {'file', 'dir'}
            or size < 0 or size > 50 * 1024**2):
        raise DataError(f'Invalid TUM archive member: {name[:140]}')
    return parts


def extract_selected(archive: Path, target: Path) -> dict:
    """Extract RGB and reference pose index, but never depth images.

    Reference pose exists on disk only for a later post-SfM evaluation step.
    It is never passed as an input to the feature matcher or mapper.
    """
    count = total = rgb_count = indexes = 0
    with tarfile.open(archive, mode='r:gz') as source:
        for item in source:
            parts = validate_archive_name(item.name, 'dir' if item.isdir() else
                                           'file' if item.isfile() else 'other', item.size)
            if item.isdir():
                continue
            count += 1
            total += item.size
            if count > 25000 or total > MAX_UNPACKED_BYTES:
                raise DataError('TUM archive extraction limits exceeded')
            relative = parts[1:]
            is_index = len(relative) == 1 and relative[0] in {'rgb.txt', 'groundtruth.txt'}
            is_rgb = (len(relative) == 2 and relative[0] == 'rgb'
                      and relative[1].lower().endswith('.png'))
            if not (is_rgb or is_index):
                continue
            out = target.joinpath(*parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            incoming = source.extractfile(item)
            if incoming is None:
                raise DataError('Unable to extract selected TUM file')
            with incoming, out.open('wb') as sink:
                shutil.copyfileobj(incoming, sink)
            if out.stat().st_size != item.size:
                raise DataError('Short read while extracting TUM data')
            indexes += int(is_index)
            rgb_count += int(is_rgb)
    if indexes != 2 or rgb_count < FRAME_COUNT * SOURCE_STRIDE:
        raise DataError('Missing TUM RGB/groundtruth index or enough real RGB images')
    return {'archive_files': count, 'declared_unpacked_bytes': total,
            'rgb_images_extracted': rgb_count, 'index_files_extracted': indexes,
            'depth_images_extracted': 0}


def get_archive(path: Path, receipt: dict, offline: Path | None = None) -> None:
    record = {'requested_official_url': URL, 'method': 'HTTPS official source' if offline is None else
              'user-provided offline archive (origin not independently verified)', 'bytes': 0}
    receipt['archive'] = record
    if offline is not None:
        if not offline.is_file() or not 1000 <= offline.stat().st_size <= MAX_ARCHIVE_BYTES:
            raise DataError('Offline archive missing or oversized')
        shutil.copyfile(offline, path)
        record['bytes'] = path.stat().st_size
        record['sha256'] = file_sha256(path)
    else:
        temp = path.with_suffix('.part')
        try:
            req = Request(URL, headers={'User-Agent': 'echo4d-cross-scene/1.0'})
            with urlopen(req, timeout=90) as response:
                redirected = response.geturl()
                resolved = urlsplit(redirected)
                if resolved.scheme != 'https' or resolved.hostname not in ALLOWED_HOSTS:
                    raise DataError('Dataset redirected outside allowed official HTTPS domains')
                record['resolved_url'] = redirected
                record['status_code'] = response.status
                expected_length = response.headers.get('Content-Length')
                if expected_length and int(expected_length) > MAX_ARCHIVE_BYTES:
                    raise DataError('Dataset exceeds download resource budget')
                digest = hashlib.sha256()
                with temp.open('wb') as sink:
                    while chunk := response.read(1024 * 1024):
                        record['bytes'] += len(chunk)
                        if record['bytes'] > MAX_ARCHIVE_BYTES:
                            raise DataError('Dataset download exceeds size limit')
                        digest.update(chunk)
                        sink.write(chunk)
                record['sha256'] = digest.hexdigest()
            if record['bytes'] < 1000:
                raise DataError('Dataset archive unexpectedly empty')
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    pinned = PINNED_SHA256 is not None and PINNED_BYTES is not None
    record['sha256_pinned_before_run'] = pinned
    if pinned and (record['sha256'] != PINNED_SHA256 or record['bytes'] != PINNED_BYTES):
        raise DataError('Downloaded data differs from previously observed fixed TUM archive')
    record['source_integrity_claim'] = ('matched independently pre-observed local pin' if pinned
                                        else 'first SHA-256 observation only; must pin and rerun')


def prepare_video(tum: Path, work: Path, r: dict) -> list[dict]:
    """Exactly the previous 8-frame stride and FFmpeg encoding, 75 images.

    Unlike xyz, desk has about 613 images: 90 images at stride 8 would
    exceed its duration, so 75 and 8 are locked *before* first run.
    """
    rows = read_rows(tum / 'rgb.txt', 2)
    chosen = rows[::SOURCE_STRIDE][:FRAME_COUNT]
    if len(chosen) != FRAME_COUNT:
        raise DataError(f'Expected {FRAME_COUNT} frames at stride {SOURCE_STRIDE}; got {len(chosen)}')
    image_dir = work / 'rgb-input'
    image_dir.mkdir()
    evidence=[]
    for i, (ts, path) in enumerate(chosen):
        original=safe_path(tum,path)
        dest=image_dir / f'frame_{i:05d}.png'
        shutil.copyfile(original,dest)
        evidence.append({'name':dest.name,'t_tum_rgb_s':float(ts),
                         'original_tum_path':path,'source_rgb_sha256':file_sha256(dest)})
    video = work / 'test-video-from-real-tum-rgb.mp4'
    invoke('encode-real-TUM-desk-RGB-into-test-MP4',[
        'ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y',
        '-framerate',str(VIDEO_FPS),'-start_number','0',
        '-i',str(image_dir / 'frame_%05d.png'),'-frames:v',str(FRAME_COUNT),
        '-c:v','libx264','-preset','ultrafast','-crf','15',
        '-pix_fmt','yuv420p',str(video)],r,timeout=180)
    r['test_video']={'from_real_TUM_RGB':True,'native_phone_video':False,
                     'artificial_timebase':True,'original_scene':'freiburg1_desk',
                     'selected_frames':FRAME_COUNT,'original_rgb_index_stride':SOURCE_STRIDE,
                     'fps':VIDEO_FPS,'bytes':video.stat().st_size,
                     'sha256':file_sha256(video),'rgb_frame_manifest':evidence}
    return evidence


def experiment(args: argparse.Namespace, r: dict) -> None:
    work = args.work.resolve()
    report = args.report_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / 'source.tgz'
    get_archive(archive, r, args.archive.resolve() if args.archive else None)
    r['extraction'] = extract_selected(archive, work / 'dataset')
    tum = work / 'dataset' / ROOT
    # Re-use the SAME selection rule, encoder and SfM runner as the xyz baseline.
    rgb_frames = prepare_video(tum, work, r)
    assert len(rgb_frames) == FRAME_COUNT
    output = work / 'sfm-out'
    cmd = [sys.executable, '-m', 'echo4d.video_sfm',
           '--video', str(work / 'test-video-from-real-tum-rgb.mp4'),
           '--output', str(output), '--fps', str(VIDEO_FPS),
           '--max-frames', str(FRAME_COUNT), '--camera-model', 'PINHOLE',
           '--camera-params', '525,525,319.5,239.5']
    try:
        invoke('run-actual-RGB-only-COLMAP-SfM-on-new-scene', cmd, r, timeout=1800)
    except Exception as failed:
        nested = output / 'report.json'
        if nested.is_file():
            r['failed_nested_sfm'] = json.loads(nested.read_text(encoding='utf-8'))
            shutil.copyfile(nested, report / 'nested-sfm-failure.json')
            failure = r['failed_nested_sfm'].get('error', {}).get('message', 'unknown error')
            raise DataError('New scene failed during COLMAP: ' + failure) from failed
        raise
    r['sfm'] = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    if not r['sfm']['status'].startswith('passed'):
        raise DataError('Real COLMAP did not return a successful track')
    track = json.loads((output / 'trajectory-scale-unknown.json').read_text(encoding='utf-8'))
    if track.get('units') != 'unknown' or track.get('metric_scale_estimated') is not False:
        raise DataError('RGB-only trajectory incorrectly claims true metric scale')
    n = r['sfm']['result']['frames_registered']
    if r['sfm']['result']['frames_extracted'] != FRAME_COUNT:
        raise DataError('Expected locked source frame count; cannot silently reduce denominator')
    r['registration'] = {'attempted': FRAME_COUNT, 'registered': n,
                         'ratio': n / FRAME_COUNT, 'min_frames': 10, 'min_ratio': .35,
                         'gate_passed': n >= 10 and n / FRAME_COUNT >= .35}
    # Only after the complete SfM pipeline do we read the known camera path.
    truth, unavailable = gt_after_sfm(tum, rgb_frames)
    r['gt_evaluation_only'] = {'file_sha256': file_sha256(tum / 'groundtruth.txt'),
                                'truth_poses': len(truth), 'no_gt_for': unavailable,
                                'note': 'Only opened after running RGB-only mapper.'}
    r['posthoc_sim3_trajectory_evaluation'] = evaluate_sfm_poses(track, truth)
    for name in ('trajectory-scale-unknown.json', 'registered_timestamps.csv'):
        shutil.copyfile(output / name, report / name)
    model_dir = output / r['sfm']['result']['sfm_model_dir']
    for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
        filename = model_dir / name
        if filename.is_file() and filename.stat().st_size <= 10 * 1024**2:
            shutil.copyfile(filename, report / name)
    r['saved_file_sha256'] = {f.name: file_sha256(f) for f in report.iterdir() if f.is_file()}
    if not r['registration']['gate_passed']:
        raise DataError('COLMAP registered fewer than 35% of 75 new-scene frames')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    p.add_argument('--archive', type=Path, help='Reuse downloaded archive after verifying its hash')
    args = p.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    r = {'experiment': 'ECHO4D-TUM-freiburg1_desk-real-RGB-COLMAP-cross-scene-v1',
         'started_utc': datetime.now(timezone.utc).isoformat(),
         'git_sha': os.getenv('ECHO_HEAD_SHA') or os.getenv('GITHUB_SHA'),
         'github_run_url': (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                            f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv('GITHUB_RUN_ID') else None),
         'status': 'running', 'source': URL, 'commands': [],
         'parameters': {'scene': 'freiburg1_desk', 'frames': FRAME_COUNT,
                        'rgb_selection_stride': SOURCE_STRIDE, 'encoded_fps': VIDEO_FPS,
                        'camera': 'PINHOLE 525,525,319.5,239.5', 'cpu_colmap': True},
         'limits': ['Real TUM office/desk RGB re-encoded as MP4; not native smartphone video.',
                    'SfM gets only RGB with assumed calibration, no depth, no ground truth.',
                    'True poses are loaded only after SfM; Sim3 is fit AND evaluated on the same poses, thus optimistic.',
                    'Even success gives neither actual metric scale nor dense depth, mesh or scene accuracy.',
                    'Initial download checksum is recorded but not pre-pinned until a second independent run.']}
    start = time.perf_counter()
    exit_code = 0
    try:
        experiment(args, r)
        r['status'] = ('passed_cross_scene_sfm_with_pinned_archive' if PINNED_SHA256
                       else 'passed_cross_scene_sfm_first_unpinned_observation')
    except Exception as exc:
        exit_code = 1
        r['status'] = 'failed'
        r['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        print(f'NEW_SCENE_SFM_FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
    finally:
        r['duration_s'] = round(time.perf_counter() - start, 3)
        (args.report_dir / 'report.json').write_text(
            json.dumps(r, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        score = r.get('posthoc_sim3_trajectory_evaluation', {})
        registration = r.get('registration', {})
        (args.report_dir / 'SUMMARY.md').write_text('\n'.join([
            '# TUM Freiburg1 Desk RGB-only cross-scene baseline', '',
            f"- Status: {r['status']}",
            f"- [GitHub Run]({r['github_run_url']})",
            f"- Archive SHA-256: {r.get('archive', {}).get('sha256', 'not downloaded')}",
            f"- Source pinned before run: {r.get('archive', {}).get('sha256_pinned_before_run', False)}",
            f"- Registered RGB images: {registration.get('registered', 'N/A')}/{FRAME_COUNT}",
            f"- Sparse points: {r.get('sfm', {}).get('result', {}).get('sparse_points', 'N/A')}",
            f"- Post-hoc Sim3 aligned ATE RMSE m: {score.get('ate_rmse_m_after_sim3', 'N/A')}",
            f"- Post-hoc median orientation degrees: {score.get('rotation_median_degrees_after_sim3', 'N/A')}",
            '- RGB-only real metres recovered: NO',
            f"- Failure: {r.get('error', {}).get('message', 'none')}", '',
            'A clean execution is not an independent mesh/metric depth accuracy certification.'
        ]) + '\n', encoding='utf-8')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
