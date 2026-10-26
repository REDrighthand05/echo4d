"""Issue #47/B: true TUM RGB images, deliberately VFR re-encoded MP4, real COLMAP.

Checks source RGB-index -> decoded-frame PTS -> reconstructed SfM t_s and
posthoc evaluation-only pose metrics. This tests encoded VFR behavior, NOT
original phone-camera hardware timestamp fidelity or metric surface accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from echo4d.model import DataError, file_sha256
from echo4d.video_pts import decoded_frame_pts
from echo4d.sfm_eval import evaluate_sfm_poses
from tum_public_roundtrip import download, extract, invoke, PINNED_ARCHIVE_SHA256, ROOT
from tum_rgb_sfm import prepare_video, gt_after_sfm, FRAME_COUNT


def run_vfr_trial(work: Path, out: Path, ledger: dict) -> None:
    archive = work / 'source.tgz'
    download(archive, ledger)
    if file_sha256(archive) != PINNED_ARCHIVE_SHA256:
        raise DataError('TUM source archive SHA is not the preregistered archive')
    ledger['archive_sha256'] = file_sha256(archive)
    extract(archive, work / 'dataset')
    tum = work / 'dataset' / ROOT
    original = prepare_video(tum, work, ledger)
    if len(original) != FRAME_COUNT:
        raise DataError('TUM source frame cohort changed')
    source_times = [item['t_tum_rgb_s'] for item in original]
    if any(not math.isfinite(b-a) or b-a <= 0
           for a, b in zip(source_times, source_times[1:])):
        raise DataError('TUM original RGB timeline is not strictly increasing')

    # FFconcat treats durations as an encoding request, not a guarantee of
    # exact preservation of original capture times. FFprobe verifies the result.
    manifest = work / 'source-times.ffconcat'
    lines = ['ffconcat version 1.0']
    for i, item in enumerate(original):
        path = work / 'rgb-input' / item['name']
        if "'" in str(path):
            raise DataError('Cannot safely quote input path for FFconcat')
        lines.append(f"file '{path}'")
        if i < FRAME_COUNT-1:
            lines.append(f"duration {source_times[i+1]-source_times[i]:.9f}")
    manifest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    video = work / 'reencoded-vfr-from-real-tum-rgb.mp4'
    invoke('encode-vfr-from-original-rgb-times', [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
        '-safe', '0', '-f', 'concat', '-i', str(manifest),
        '-fps_mode', 'vfr', '-c:v', 'libx264', '-preset', 'ultrafast',
        '-crf', '15', '-pix_fmt', 'yuv420p', str(video)
    ], ledger, timeout=180)
    ledger['encoded_vfr'] = {
        'sha256': file_sha256(video), 'ffconcat_sha256': file_sha256(manifest),
        'original_rgb_sha256': [item['source_rgb_sha256'] for item in original],
        'origin': 'public TUM true RGB frames, re-encoded with VFR durations',
        'not_native_camera_video': True,
        'not_original_hardware_pts': True,
    }
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'frame=best_effort_timestamp_time',
                 '-of', 'json', str(video)]
    result = subprocess.run(probe_cmd, capture_output=True, text=True,
                            errors='replace', check=False, timeout=180)
    ledger['probe_exit_code'] = result.returncode
    if result.returncode:
        raise DataError('ffprobe failed for re-encoded VFR source: '
                        + result.stderr[-400:])
    pts = decoded_frame_pts(result.stdout)
    if len(pts) != FRAME_COUNT:
        raise DataError(f'VFR decoded frame cohort is {len(pts)}, expected {FRAME_COUNT}')
    relative_video = [v - pts[0] for v in pts]
    relative_original = [v - source_times[0] for v in source_times]
    timing_error = [relative_video[i]-relative_original[i]
                    for i in range(FRAME_COUNT)]
    max_abs = max(abs(v) for v in timing_error)
    ledger['source_time_comparison'] = {
        'frames': FRAME_COUNT,
        'original_tum_rgb_relative_s': relative_original,
        'decoded_video_relative_pts_s': relative_video,
        'encoded_pts_minus_original_tum_s': timing_error,
        'max_abs_seconds': max_abs,
        'original_pts_preserved_exactly': max_abs < 1e-8,
        'note': 'Encoding may quantize source capture times; use decoded video PTS '
                'for video-frame time, original TUM capture time for sensor sync.'
    }
    if max_abs > 0.08:
        raise DataError('VFR encoding drifted >80ms against original TUM RGB times')
    sfm = work / 'sfm-source-pts'
    cmd = [sys.executable, '-m', 'echo4d.video_sfm',
           '--video', str(video), '--output', str(sfm),
           '--max-frames', str(FRAME_COUNT),
           '--timestamp-mode', 'source-pts',
           '--camera-model', 'PINHOLE',
           '--camera-params', '525,525,319.5,239.5']
    try:
        invoke('real-colmap-VFR-source-pts', cmd, ledger, timeout=1800)
    except Exception:
        nested = sfm / 'report.json'
        if nested.is_file():
            shutil.copyfile(nested, out / 'failed-colmap-report.json')
        raise
    nested = json.loads((sfm / 'report.json').read_text(encoding='utf-8'))
    if not nested['status'].startswith('passed'):
        raise DataError('Real COLMAP returned nonpassing report')
    if nested['result']['frames_extracted'] != FRAME_COUNT:
        raise DataError('Real VFR extraction did not preserve 90 frames')
    mapping = nested['source_pts_sampling']['frame_map']
    if [m['decoded_frame_index'] for m in mapping] != list(range(FRAME_COUNT)):
        raise DataError('VFR extracted frame identity does not match decoded indices')
    if any(abs(m['relative_pts_seconds']-relative_video[i]) > 1e-8
           for i, m in enumerate(mapping)):
        raise DataError('Video PTS cannot be traced to the extracted frame names')
    track = json.loads((sfm / 'trajectory-scale-unknown.json').read_text())
    if (track['version'] != 'echo-sfm-trajectory-1'
            or track['units'] != 'unknown'
            or track['metric_scale_estimated'] is not False):
        raise DataError('VFR SfM publication violates unknown-scale downstream contract')
    if any(abs(pose['t_s']-relative_video[int(pose['name'][6:11])]) > 1e-8
           for pose in track['frames']):
        raise DataError('Registered trajectory lost its original decoded PTS')

    # GT must not influence extraction, model selection or camera estimation.
    truth, excluded = gt_after_sfm(tum, original)
    evaluation = evaluate_sfm_poses(track, truth)
    ledger['posthoc_evaluation_only'] = {
        'groundtruth_sha256': file_sha256(tum / 'groundtruth.txt'),
        'not_interpolatable': excluded, 'values': evaluation}
    ledger['result'] = nested['result']
    ledger['model_candidates'] = nested.get('model_candidates')
    ledger['all_extracted_png_sha256'] = {
        f'frame_{i:05d}.png':
            file_sha256(sfm / 'images' / f'frame_{i:05d}.png')
        for i in range(FRAME_COUNT)}
    for name in ('trajectory-scale-unknown.json', 'all_input_timestamps.csv',
                 'registered_timestamps.csv', 'report.json'):
        shutil.copyfile(sfm / name, out / ('vfr-' + name))
    model = sfm / nested['result']['sfm_model_dir']
    for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
        file = model / name
        if file.is_file() and file.stat().st_size < 8 * 1024**2:
            shutil.copyfile(file, out / ('vfr-selected-' + name))
    ledger['status'] = 'passed_real_rgb_reencoded_vfr_source_pts_not_metric'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.report_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ledger = {
        'experiment': 'issue47-B-real-TUM-frames-reencoded-VFR-PTS',
        'status': 'running', 'commands': [],
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'git_sha': os.environ.get('GITHUB_SHA'),
        'run_url': (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
                    if os.getenv('GITHUB_RUN_ID') else None),
    }
    started = time.monotonic()
    try:
        work = args.work.resolve()
        work.mkdir(parents=True, exist_ok=False)
        run_vfr_trial(work, out, ledger)
        code = 0
    except Exception as exc:
        ledger['status'] = 'failed'
        ledger['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        print(f'ISSUE47_B_VFR_FAILED: {type(exc).__name__}: {exc}',
              file=sys.stderr)
        code = 1
    finally:
        ledger['seconds'] = round(time.monotonic()-started, 3)
        (out / 'report.json').write_text(
            json.dumps(ledger, ensure_ascii=False, allow_nan=False, indent=2) + '\n',
            encoding='utf-8')
        result = ledger.get('result', {})
        ev = ledger.get('posthoc_evaluation_only', {}).get('values', {})
        summary = [
            '# Issue #47 B — actual RGB / re-encoded VFR / real COLMAP',
            f"- Status: {ledger['status']}", f"- Run: {ledger['run_url']}",
            f"- Registered: {result.get('frames_registered','not measured')}/"
            f"{result.get('frames_extracted','not measured')}",
            f"- Maximum encoded PTS drift from original TUM RGB: "
            f"{ledger.get('source_time_comparison',{}).get('max_abs_seconds','not measured')} s",
            f"- Evaluation-only aligned ATE RMSE: "
            f"{ev.get('ate_rmse_m_after_sim3','not measured')} m",
            '- No original phone hardware PTS, independent surface truth, or RGB-only physical scale.',
            f"- Error: {ledger.get('error','none')}",
        ]
        (out / 'SUMMARY.md').write_text('\n'.join(summary)+'\n', encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
