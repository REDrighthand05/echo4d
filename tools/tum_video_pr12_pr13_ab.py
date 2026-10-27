"""Frozen same-video PR #12 vs PR #13 RGB SfM comparison.

Real TUM RGB source, one encoded CFR video (90 decoded frames at 4fps).
Both modes use the same image cohort, PINHOLE measured intrinsics and CPU
COLMAP flags. PR #13 is checked out at the pinned unchanged source commit;
a transparent COLMAP executable wrapper supplies its missing CLI intrinsics
option without editing the PR #13 Python implementation.

Do not evaluate camera accuracy unless image hashes and timestamps agree.
Ground truth is read only after both RGB-only reconstructions have run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

from echo4d.model import DataError, file_sha256
from echo4d.sfm_eval import evaluate_sfm_poses
from tum_public_roundtrip import download, extract, ROOT, PINNED_ARCHIVE_SHA256
from tum_rgb_sfm import prepare_video, gt_after_sfm, FRAME_COUNT, VIDEO_FPS

PR13_SHA = 'dd1b4be5d8b85b4bdf1fe11661876c8e15d0cdb0'
CAMERA_PARAMS = '525,525,319.5,239.5'


def run_command(cmd: list[str], label: str, report: dict, *, env=None, timeout=1800) -> bool:
    item = {'label': label, 'command': cmd, 'status': 'running'}
    report.setdefault('commands', []).append(item)
    start = time.monotonic()
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, errors='replace',
                                   env=env, timeout=timeout, check=False)
        item.update({'exit_code': completed.returncode,
                     'stdout_tail': completed.stdout[-2500:],
                     'stderr_tail': completed.stderr[-5500:]})
        item['status'] = 'passed' if completed.returncode == 0 else 'failed'
        return completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        item.update({'status': 'failed',
                     'error': f'{type(exc).__name__}: {str(exc)[:600]}'})
        return False
    finally:
        item['seconds'] = round(time.monotonic() - start, 3)


def write_colmap_adapter(path: Path) -> None:
    """Run original COLMAP with identical *declared* settings in both candidates.

    PR #13's CLI accepts --colmap but not --camera-params. This wrapper affects
    external COLMAP options only; its full executable is kept in the artifact.
    """
    code = """#!/usr/bin/env python3
import subprocess
import sys
argv = list(sys.argv[1:])
if argv and argv[0] == 'feature_extractor':
    argv += ['--ImageReader.camera_params', '525,525,319.5,239.5',
             '--SiftExtraction.max_num_features', '4096',
             '--SiftExtraction.num_threads', '2']
if argv and argv[0] == 'sequential_matcher':
    argv += ['--SequentialMatching.overlap', '10',
             '--SiftMatching.num_threads', '2']
raise SystemExit(subprocess.call(['colmap'] + argv))
"""
    path.write_text(code, encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def read_sample_times(csv_path: Path) -> list[float]:
    with csv_path.open(newline='', encoding='utf-8') as handle:
        entries = list(csv.DictReader(handle))
    return [float(entry['t']) for entry in entries]


def compare_inputs(a: Path, b: Path, report: dict) -> None:
    left = sorted((a / 'images').glob('*.png'))
    right = sorted((b / 'images').glob('*.png'))
    lh = [file_sha256(p) for p in left]
    rh = [file_sha256(p) for p in right]
    t12 = read_sample_times(a / 'all_input_timestamps.csv')
    pr13 = json.loads((b / 'report.json').read_text(encoding='utf-8'))
    t13 = [x['relative_pts_seconds'] for x in pr13['sampling']['samples']]
    report['same_input'] = {
        'pr12_count': len(lh), 'pr13_count': len(rh),
        'matching_image_sha256_by_index': lh == rh,
        'pr12_images_sha256': lh, 'pr13_images_sha256': rh,
        'pr12_relative_seconds': t12, 'pr13_relative_seconds': t13,
        'maximum_absolute_timing_delta_s':
            max((abs(x-y) for x,y in zip(t12,t13)), default=None)
            if len(t12) == len(t13) else None,
        'same_camera_model': 'PINHOLE',
        'camera_params_supplied_to_both': CAMERA_PARAMS,
        'sampling_policy': 'all 90 decoded frames of fixed 4fps CFR MP4',
        'colmap_settings': 'both fixed 4096 SIFT features, overlap 10, CPU/two threads',
    }
    if len(lh) != FRAME_COUNT or lh != rh:
        raise DataError('Same-frame SHA256 gate failed: cannot compare camera accuracy')
    if len(t12) != len(t13) or any(abs(x-y) > 0.001 for x,y in zip(t12,t13)):
        raise DataError('Same-video timestamp gate failed: cannot compare camera accuracy')


def compare_poses(a: Path, b: Path, truth: dict, report: dict) -> None:
    track12 = json.loads((a / 'trajectory-scale-unknown.json').read_text())
    track13 = json.loads((b / 'trajectory.json').read_text())
    poses13 = []
    for p in track13['poses']:
        idx = int(Path(p['name']).stem) - 1
        name = f'frame_{idx:05d}.png'
        matrix = p['camera_to_world_row_major']
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise DataError('Invalid PR13 pose dimensions')
        flat_column_major = [matrix[row][col] for col in range(4) for row in range(4)]
        poses13.append({'name': name, 'camera_to_sfm_world': flat_column_major})
    compatible13 = {'units': 'unknown', 'metric_scale_estimated': False, 'frames': poses13}
    report['comparison'] = {
        'pr12': {'registration_count': len(track12['frames']),
                 'registration_fraction': len(track12['frames']) / FRAME_COUNT,
                 'evaluation_only': evaluate_sfm_poses(track12, truth)},
        'pr13': {'registration_count': len(poses13),
                 'registration_fraction': len(poses13) / FRAME_COUNT,
                 'evaluation_only': evaluate_sfm_poses(compatible13, truth)},
    }
    for label, path, original in (
        ('pr12', a, json.loads((a/'report.json').read_text())),
        ('pr13', b, json.loads((b/'report.json').read_text()))):
        selected = (path / original['result']['sfm_model_dir']
                    if label == 'pr12' else
                    path / ('model_txt_' + original['selected_model_id']))
        report['comparison'][label]['model_cameras_txt_sha256'] = file_sha256(selected/'cameras.txt')
        report['comparison'][label]['model_images_txt_sha256'] = file_sha256(selected/'images.txt')


def experiment(args, report: dict) -> None:
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    if not (args.pr13_source / 'echo4d' / 'video_sfm.py').is_file():
        raise DataError('Missing pinned PR13 checkout src/echo4d/video_sfm.py')
    pinned = subprocess.run(
        ['git', '-C', str(args.pr13_source), 'rev-parse', 'HEAD'],
        capture_output=True, text=True, check=False)
    report['pr13_actual_checkout_sha'] = pinned.stdout.strip()
    if pinned.returncode != 0 or pinned.stdout.strip() != PR13_SHA:
        raise DataError('PR13 comparison source is not the frozen expected commit')
    archive = work / 'source.tgz'
    download(archive, report)
    report['archive_sha256'] = file_sha256(archive)
    if report['archive_sha256'] != PINNED_ARCHIVE_SHA256:
        raise DataError('TUM archive SHA does not match pre-registered bytes')
    extract(archive, work / 'dataset')
    tum = work / 'dataset' / ROOT
    source = prepare_video(tum, work, report)
    video = work / 'test-video-from-real-tum-rgb.mp4'
    report['single_video_sha256'] = file_sha256(video)
    wrapper = work / 'pr13-fixed-colmap-wrapper'
    write_colmap_adapter(wrapper)
    report['pr13_colmap_wrapper_sha256'] = file_sha256(wrapper)
    a, b = work/'pr12', work/'pr13'
    a_command = [sys.executable, '-m', 'echo4d.video_sfm',
                 '--video', str(video), '--output', str(a), '--fps', str(VIDEO_FPS),
                 '--max-frames', str(FRAME_COUNT), '--camera-model', 'PINHOLE',
                 '--camera-params', CAMERA_PARAMS]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(args.pr13_source.resolve())
    b_command = [sys.executable, '-m', 'echo4d.video_sfm', str(video),
                 '--output', str(b), '--max-frames', str(FRAME_COUNT),
                 '--colmap', str(wrapper)]
    a_ok = run_command(a_command, 'PR12-modified', report)
    b_ok = run_command(b_command, 'PR13-pinned', report, env=env)
    for label, path in (('pr12', a), ('pr13', b)):
        for src, dst in (
            ('report.json', f'{label}_report.json'),
            ('trajectory-scale-unknown.json', f'{label}_trajectory-scale-unknown.json'),
            ('trajectory.json', f'{label}_trajectory.json'),
            ('all_input_timestamps.csv', f'{label}_all_input_timestamps.csv'),
            ('registered_timestamps.csv', f'{label}_registered_timestamps.csv'),
        ):
            if (path / src).is_file():
                shutil.copyfile(path / src, args.report_dir / dst)
    if not (a_ok and b_ok):
        raise DataError('One or both RGB-only COLMAP runs failed; see raw reports')
    compare_inputs(a, b, report)
    # Scoring truth must not be loaded until BOTH SfM solvers complete.
    truth, excluded = gt_after_sfm(tum, source)
    report['posthoc_groundtruth_exclusions'] = excluded
    compare_poses(a, b, truth, report)
    for label, path in (('pr12', a), ('pr13', b)):
        record = json.loads((path / 'report.json').read_text(encoding='utf-8'))
        model = (path / record['result']['sfm_model_dir'] if label == 'pr12'
                 else path / ('model_txt_' + record['selected_model_id']))
        for name in ('images.txt', 'cameras.txt', 'points3D.txt'):
            file = model / name
            if file.is_file() and file.stat().st_size <= 8 * 1024**2:
                shutil.copyfile(file, args.report_dir / f'{label}_{name}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--report-dir', required=True, type=Path)
    parser.add_argument('--pr13-source', required=True, type=Path)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'experiment': 'PR12-vs-PR13-same-CFR-video-and-intrinsics',
        'pr13_pinned_source_sha': PR13_SHA,
        'source': 'TUM Freiburg1 XYZ, RGB ONLY for SfM',
        'groundtruth_used_in_sfm': False,
        'physical_scale_obtained': False,
        'commands': [],
        'status': 'running',
        'locked': {'frame_count': FRAME_COUNT, 'fps': VIDEO_FPS,
                   'camera_model': 'PINHOLE', 'camera_params': CAMERA_PARAMS,
                   'dataset_sha256': PINNED_ARCHIVE_SHA256},
        'github_run_url': ('https://github.com/' + os.environ.get('GITHUB_REPOSITORY','')
                           + '/actions/runs/' + os.environ.get('GITHUB_RUN_ID',''))
                           if os.environ.get('GITHUB_RUN_ID') else None,
    }
    start = time.monotonic()
    try:
        experiment(args, report)
        report['status'] = 'passed_same_input_pose_comparison_unknown_scale'
        code = 0
    except Exception as exc:
        report['status'] = 'failed'
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
        print(f"FROZEN_PR12_PR13_AB_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        code = 1
    finally:
        report['elapsed_seconds'] = round(time.monotonic()-start,3)
        (args.report_dir/'report.json').write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n',
            encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
