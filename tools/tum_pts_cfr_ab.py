"""Paired CFR test: one pinned real-TUM RGB MP4, two true COLMAP runs.

This compares input identity, registration and evaluation-only camera errors.
It cannot establish metric scale, surface accuracy or VFR camera timestamp truth.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

from echo4d.model import DataError, file_sha256
from echo4d.sfm_eval import evaluate_sfm_poses
from tum_public_roundtrip import download, extract, ROOT, URL, invoke
from tum_rgb_sfm import prepare_video, gt_after_sfm, FRAME_COUNT, VIDEO_FPS


def experiment(work: Path, out: Path, receipt: dict) -> None:
    archive = work / 'source.tgz'
    download(archive, receipt)
    receipt['dataset'] = {'sha256': file_sha256(archive), 'url': URL}
    receipt['extraction'] = extract(archive, work / 'dataset')
    tum = work / 'dataset' / ROOT
    frame_sources = prepare_video(tum, work, receipt)
    video = work / 'test-video-from-real-tum-rgb.mp4'
    receipt['same_mp4_sha256'] = file_sha256(video)
    receipt['fixed_settings'] = {
        'frames': FRAME_COUNT, 'fps': VIDEO_FPS,
        'camera_model': 'PINHOLE', 'camera_params': '525,525,319.5,239.5',
        'no_gt_used_for_colmap': True}
    checked = {}
    for mode in ('resampled', 'source-pts'):
        folder = work / f'sfm-{mode}'
        cmd = [sys.executable, '-m', 'echo4d.video_sfm',
               '--video', str(video), '--output', str(folder),
               '--fps', str(VIDEO_FPS), '--max-frames', str(FRAME_COUNT),
               '--camera-model', 'PINHOLE',
               '--camera-params', '525,525,319.5,239.5',
               '--timestamp-mode', mode]
        try:
            invoke('sfm-' + mode, cmd, receipt, timeout=1800)
        except Exception:
            failure = folder / 'report.json'
            if failure.is_file():
                shutil.copyfile(failure, out / f'{mode}-failure.json')
            raise
        result = json.loads((folder / 'report.json').read_text())
        if not result['status'].startswith('passed'):
            raise DataError(f'{mode}: real COLMAP did not pass')
        if result['result']['frames_extracted'] != FRAME_COUNT:
            raise DataError(f'{mode}: extracted cohort changed')
        track = json.loads((folder / 'trajectory-scale-unknown.json').read_text())
        if track['version'] != 'echo-sfm-trajectory-1' or track['units'] != 'unknown':
            raise DataError(f'{mode}: downstream SfM trajectory contract changed')
        hashes = {f'frame_{i:05d}.png':
                  file_sha256(folder / 'images' / f'frame_{i:05d}.png')
                  for i in range(FRAME_COUNT)}
        checked[mode] = (result, track, hashes)
        for name in ('report.json', 'trajectory-scale-unknown.json',
                     'registered_timestamps.csv', 'all_input_timestamps.csv'):
            shutil.copyfile(folder / name, out / f'{mode}-{name}')

    one, two = checked['resampled'], checked['source-pts']
    receipt['same_input'] = {
        'identical_90_png_sha256': one[2] == two[2],
        'per_view': [{'name': name, 'resampled_sha256': value,
                      'source_pts_sha256': two[2][name]}
                     for name, value in one[2].items()]}
    if not receipt['same_input']['identical_90_png_sha256']:
        raise DataError('Same-input A/B invalid: decoded RGB images differ')
    pts = two[0].get('source_pts_sampling')
    if not pts or [x['decoded_frame_index'] for x in pts['frame_map']] != list(range(FRAME_COUNT)):
        raise DataError('Source-PTS identity mapping is not the frozen 90 frames')

    # GT is only opened after BOTH genuine COLMAP runs have completed.
    gt, excluded = gt_after_sfm(tum, frame_sources)
    receipt['gt_evaluation_only'] = {
        'sha256': file_sha256(tum / 'groundtruth.txt'),
        'matched': len(gt), 'excluded': excluded}
    receipt['comparison'] = {}
    for mode, (run, track, _) in checked.items():
        ev = evaluate_sfm_poses(track, gt)
        receipt['comparison'][mode] = {
            'registration': run['result'],
            'posthoc_pose_evaluation': ev}
    receipt['interpretation'] = (
        'Exact same CFR video, camera intrinsics and decoded PNGs: differences '
        'in COLMAP registration may be run-to-run variation rather than PTS. '
        'GT Sim3 is for score only, never physical recovered scale.')
    receipt['status'] = 'passed_same_video_two_real_colmap_runs'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--report-dir', required=True, type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    out = args.report_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'commands': [],
              'experiment': 'pr34-b-paired-CFR-colmap',
              'head_sha': os.environ.get('ECHO_HEAD_SHA', os.environ.get('GITHUB_SHA')),
              'run_url': (f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
                          if os.environ.get('GITHUB_RUN_ID') else None),
              'utc': datetime.now(timezone.utc).isoformat()}
    begin = time.perf_counter()
    try:
        work.mkdir(parents=True, exist_ok=False)
        experiment(work, out, report)
        code = 0
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        print(f'PAIRED_COLMAP_FAILED: {exc}', file=sys.stderr)
        code = 1
    finally:
        report['seconds'] = round(time.perf_counter() - begin, 3)
        (out / 'report.json').write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
        summary = ['# PR34 B — same-video PTS / resampled real COLMAP',
                   f"- Status: {report['status']}",
                   f"- Run: {report['run_url']}",
                   f"- Input video SHA: {report.get('same_mp4_sha256', 'unavailable')}",
                   f"- Identical PNG input: {report.get('same_input', {}).get('identical_90_png_sha256', 'unverified')}"]
        for mode, row in report.get('comparison', {}).items():
            summary.append(f"- {mode}: registered {row['registration']['frames_registered']}/{row['registration']['frames_extracted']}; posthoc Sim3 ATE {row['posthoc_pose_evaluation'].get('ate_rmse_m_after_sim3')} m")
        summary.append('- Evaluation-only alignment, no recovered physical scale or dense geometry.')
        if report.get('error'):
            summary.append(f"- Failure: {report['error']['message']}")
        (out / 'SUMMARY.md').write_text('\n'.join(summary) + '\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
