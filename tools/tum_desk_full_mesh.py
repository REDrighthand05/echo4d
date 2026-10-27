"""Frozen second-scene desk RGB SfM -> measured scale -> sensor TSDF -> withheld Z.

Different physical scene from freiburg1_xyz. No GT pose or geometric holdout
depth is used to estimate camera motion, scale, mesh, or training bounds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

import numpy as np

from echo4d.cli import reconstruct_command
from echo4d.metric_anchor import calibrate
from echo4d.metric_holdout import evaluate_sensor_metric_holdouts
from echo4d.metric_rgbd import convert_metric_sensor
from echo4d.model import DataError, file_sha256, safe_path
from echo4d.tum import nearest_timestamp, read_rows
from tum_desk_sfm import (FRAME_COUNT, SOURCE_STRIDE, PINNED_SHA256, PINNED_BYTES, ROOT, URL,
                          extract_selected, get_archive, prepare_video,
                          validate_archive_name)
from tum_public_roundtrip import invoke
from desk_source_cohort import MODES, extract_time_indexes, plan_cohort
from tum_rgb_sfm import VIDEO_FPS, gt_after_sfm
from tum_sensor_metric_mesh import (BBOX_MARGIN_M, BBOX_PERCENTILES, DOWNSAMPLE,
                                    MAX_DEPTH_M, MAX_VOXELS, MIN_OBSERVATIONS,
                                    VOXEL_M, depth_only_training_bounds)

# PREDECLARED on PR #34 D before executing second-scene Mesh experiment.
GEOMETRY_SELECTED = tuple(range(60))
GEOMETRY_HELDOUT = (4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59)
SCALE_ANCHORS = (15, 45, 65)
SCALE_DIAGNOSTICS = (60, 61, 62, 63, 64, 66, 67)
MAX_GAP_S = .03
DEPTH_UNITS_PER_M = 5000


def frozen_protocol(source: list[dict]) -> dict:
    if (len(source) != FRAME_COUNT or
            [x.get('name') for x in source] !=
            [f'frame_{i:05d}.png' for i in range(FRAME_COUNT)]):
        raise DataError('Desk source must have exactly the frozen 75 ordered RGB names')
    selected, held = set(GEOMETRY_SELECTED), set(GEOMETRY_HELDOUT)
    anchors, checks = set(SCALE_ANCHORS), set(SCALE_DIAGNOSTICS)
    if (len(selected) != 60 or len(held) != 12 or len(anchors) != 3 or
            len(checks) != 7 or not held.issubset(selected) or
            held & anchors or held & checks or anchors & checks or
            checks & selected or
            any(i < 0 or i >= FRAME_COUNT for i in selected | anchors | checks)):
        raise DataError('Invalid registered predeclared scene roles')
    train = [source[i]['name'] for i in GEOMETRY_SELECTED if i not in held]
    withheld = [source[i]['name'] for i in GEOMETRY_HELDOUT]
    if len(train) != 48 or len(set(train + withheld)) != 60:
        raise DataError('Invalid desk 48/12 geometry split')
    return {
        'version': 'desk-metric-mesh-v1-frozen',
        'source': 'TUM freiburg1_desk, 75 real RGB images re-encoded as 4fps MP4',
        'selected_indices': GEOMETRY_SELECTED, 'holdout_indices': GEOMETRY_HELDOUT,
        'scale_anchor_indices': SCALE_ANCHORS,
        'scale_check_indices': SCALE_DIAGNOSTICS,
        'train_names': train, 'holdout_names': withheld,
        'max_rgb_depth_gap_s': MAX_GAP_S,
        'depth_numeric_units_per_meter': DEPTH_UNITS_PER_M,
        'voxel_m': VOXEL_M, 'max_voxels': MAX_VOXELS,
        'max_depth_m': MAX_DEPTH_M, 'min_observations': MIN_OBSERVATIONS,
        'downsample': DOWNSAMPLE,
        'bounds_method': {'percentiles': BBOX_PERCENTILES,
                          'margin_m': BBOX_MARGIN_M, 'source': 'training depth only'},
        'screening_gates': {'min_coverage': .80, 'max_overlap_rmse_m': .15},
    }


def extract_sensor_depths(archive: Path, dataset: Path) -> dict:
    """Safely extract sensor depth only AFTER SfM; never replace RGB/GT files."""
    count = total = extracted = index_count = 0
    seen: set[str] = set()
    with tarfile.open(archive, mode='r:gz') as stream:
        for item in stream:
            parts = validate_archive_name(
                item.name,
                'dir' if item.isdir() else 'file' if item.isfile() else 'other',
                item.size)
            if item.isdir():
                continue
            count += 1
            total += item.size
            if count > 25000 or total > 3 * 1024**3:
                raise DataError('Desk dataset extraction resource limit exceeded')
            rel = parts[1:]
            is_index = len(rel) == 1 and rel[0] == 'depth.txt'
            is_depth = (len(rel) == 2 and rel[0] == 'depth'
                        and rel[1].lower().endswith('.png'))
            if not (is_index or is_depth):
                continue
            if item.name in seen:
                raise DataError('Duplicate selected sensor member in archive')
            seen.add(item.name)
            dest = dataset.joinpath(*parts)
            if dest.exists():
                raise DataError('Refusing to overwrite an extracted sensor member')
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = stream.extractfile(item)
            if data is None:
                raise DataError('Unable to read real sensor member')
            with data, dest.open('xb') as sink:
                shutil.copyfileobj(data, sink)
            if dest.stat().st_size != item.size:
                raise DataError('Short sensor member read')
            index_count += int(is_index)
            extracted += int(is_depth)
    if index_count != 1 or extracted < FRAME_COUNT:
        raise DataError('TUM desk archive lacks depth.txt or enough measured PNGs')
    return {'depth_index_count': index_count, 'depth_pngs_extracted': extracted,
            'depth_index_sha256': file_sha256(dataset / ROOT / 'depth.txt')}



def sensor_pairing_diagnostics(tum: Path, source: list[dict],
                               protocol: dict) -> dict:
    """Explain nearest raw RGB/depth timestamp gaps WITHOUT opening depth pixels.

    This inventory is informational only. It never changes the frozen 75 RGB
    source identities, the 48/12 split, the 30ms gate, or depth associations.
    Calling match_sensor_inputs afterwards remains mandatory.
    """
    depth_rows = read_rows(tum / 'depth.txt', 2)
    times = np.asarray([float(row[0]) for row in depth_rows], dtype=np.float64)
    if (not len(times) or not np.isfinite(times).all() or
            np.any(np.diff(times) <= 0)):
        raise DataError('Desk depth index timestamps invalid')
    roles = sorted(set(GEOMETRY_SELECTED) | set(SCALE_ANCHORS) |
                   set(SCALE_DIAGNOSTICS))
    if (len(source) != FRAME_COUNT or len(protocol['train_names']) != 48 or
            len(protocol['holdout_names']) != 12):
        raise DataError('Frozen desk source or split changed before audit')
    rows, first_seen = [], {}
    for index in roles:
        name = source[index].get('name')
        if name != f'frame_{index:05d}.png':
            raise DataError('Frozen desk RGB frame order changed')
        t = float(source[index]['t_tum_rgb_s'])
        if not math.isfinite(t):
            raise DataError('Nonfinite source RGB timestamp')
        # No maximum-gap filter in this DIAGNOSTIC: record the closest actual
        # measurement even if the still-frozen 30ms acceptance gate rejects it.
        nearest = nearest_timestamp(times, t, float('inf'))
        if nearest is None:
            raise DataError('No sensor timestamps available')
        dt = float(times[nearest] - t)
        name_depth = depth_rows[nearest][1]
        duplicate = name_depth in first_seen
        if not duplicate:
            first_seen[name_depth] = name
        role = ('geometry_holdout' if index in GEOMETRY_HELDOUT else
                'scale_anchor' if index in SCALE_ANCHORS else
                'scale_diagnostic' if index in SCALE_DIAGNOSTICS else
                'geometry_training')
        rows.append({'name': name, 'source_frame_index': index,
                     'source_rgb_index': source[index].get('source_rgb_index', index), 'role': role,
                     'source_rgb_time_s': t,
                     'nearest_depth_time_s': float(times[nearest]),
                     'nearest_depth_path': name_depth, 'signed_gap_s': dt,
                     'absolute_gap_s': abs(dt),
                     'within_frozen_30ms': abs(dt) <= MAX_GAP_S,
                     'duplicate_nearest_depth': duplicate,
                     'first_used_by': first_seen[name_depth] if duplicate else None})
    gaps = np.asarray([row['absolute_gap_s'] for row in rows], dtype=float)
    failures = [row for row in rows if not row['within_frozen_30ms']]
    duplicates = [row for row in rows if row['duplicate_nearest_depth']]
    return {'version': 'desk-rgb-depth-index-time-audit-1',
            'diagnostic_only_no_depth_pixels_opened': True,
            'source_selection_and_geometry_holdouts_unchanged': True,
            'no_time_shift_or_threshold_fitting': True,
            'max_allowed_gap_s': MAX_GAP_S, 'frozen_rgb_roles': len(roles),
            'depth_index_rows': len(depth_rows),
            'within_gap_count': len(rows) - len(failures),
            'outside_gap_count': len(failures),
            'duplicate_nearest_depth_count': len(duplicates),
            'first_outside_gap': failures[0] if failures else None,
            'median_abs_gap_s': float(np.median(gaps)),
            'p95_abs_gap_s': float(np.percentile(gaps, 95)),
            'max_abs_gap_s': float(gaps.max()),
            'all_pairs_within_gap_and_unique': not (failures or duplicates),
            'pairs': rows}


def match_sensor_inputs(tum: Path, source: list[dict],
                        protocol: dict) -> dict[str, dict]:
    """Validate identities and timestamps; do not read geometric holdout pixels."""
    depth_rows = read_rows(tum / 'depth.txt', 2)
    timestamps = np.asarray([float(row[0]) for row in depth_rows], dtype=float)
    if (not len(timestamps) or not np.isfinite(timestamps).all() or
            np.any(np.diff(timestamps) <= 0)):
        raise DataError('Desk depth.txt has non-monotonic/invalid timestamps')
    roles = set(GEOMETRY_SELECTED) | set(SCALE_ANCHORS) | set(SCALE_DIAGNOSTICS)
    result: dict[str, dict] = {}
    used_paths: set[str] = set()
    for index in sorted(roles):
        entry = source[index]
        name = entry['name']
        t = float(entry['t_tum_rgb_s'])
        if not math.isfinite(t):
            raise DataError('Nonfinite source RGB timestamp')
        at = nearest_timestamp(timestamps, t, MAX_GAP_S)
        if at is None:
            raise DataError(f'Missing sensor Z within 30ms for fixed {name}')
        depth_name = depth_rows[at][1]
        depth_path = safe_path(tum, depth_name)
        if depth_name in used_paths or not depth_path.is_file():
            raise DataError(f'Duplicate or absent depth source for {name}')
        used_paths.add(depth_name)
        result[name] = {'name': name, 'depth': depth_name,
                        'timestamp_delta_seconds': float(timestamps[at] - t),
                        'source_rgb_time_s': t,
                        'source_depth_time_s': float(timestamps[at])}
    if len(result) != len(roles) or len(protocol['train_names']) != 48:
        raise DataError('Incomplete predeclared desk sensor cohort')
    return result


def persist(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')
    return file_sha256(path)


def run(args: argparse.Namespace, record: dict) -> None:
    work, report = args.work.resolve(), args.report_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / 'source.tgz'
    get_archive(archive, record, args.archive.resolve() if args.archive else None)
    if (PINNED_SHA256 is None or PINNED_BYTES is None or
            file_sha256(archive) != PINNED_SHA256 or
            archive.stat().st_size != PINNED_BYTES):
        raise DataError('Pre-pinned TUM desk archive bytes/SHA mismatch')
    record['source_archive_sha256'] = file_sha256(archive)
    record['source_archive_bytes'] = archive.stat().st_size
    # Check source association before video encoding/COLMAP. This opens only
    # original text indexes, never holdout depth pixels or GT poses.
    index_root = work / 'source-time-indexes'
    record['preflight_source_index_sha256'] = extract_time_indexes(archive, index_root)
    mode = getattr(args, 'cohort', 'v1-frozen')
    plan = plan_cohort(index_root, mode)
    record['source_cohort_plan_sha256'] = persist(report / 'source-cohort-plan.json', plan)
    if not plan['selection_complete']:
        raise DataError('Insufficient legal source pairs for the predeclared 75-frame cohort')
    planned_source = plan['selected_frames']
    protocol = frozen_protocol(planned_source)
    if mode == 'v2-source-paired':
        protocol['version'] = 'desk-metric-mesh-v2-source-paired'
    protocol['source_cohort_mode'] = mode
    protocol['source_cohort_plan_sha256'] = record['source_cohort_plan_sha256']
    protocol['original_source_indices'] = [x['source_rgb_index'] for x in planned_source]
    record['protocol_sha256'] = persist(report / 'frozen-protocol.json', protocol)
    preflight = sensor_pairing_diagnostics(index_root, planned_source, protocol)
    record['sensor_pairing_preflight_sha256'] = persist(
        report / 'sensor-pairing-preflight.json', preflight)
    if not preflight['all_pairs_within_gap_and_unique']:
        raise DataError('Frozen desk source pairing failed before encoding/SfM; see preflight')

    record['rgb_extraction'] = extract_selected(archive, work / 'dataset')
    tum = work / 'dataset' / ROOT
    chosen = [[str(x['t_tum_rgb_s']), x['original_tum_path']] for x in planned_source]
    source = prepare_video(tum, work, record, chosen_rows=chosen)
    record['test_video']['source_cohort_mode'] = mode
    if mode == 'v2-source-paired':
        record['test_video']['original_rgb_index_stride'] = None
        record['test_video']['eligible_rgb_index_stride'] = SOURCE_STRIDE
    if [(x['original_tum_path'], x['t_tum_rgb_s']) for x in source] != [
            (x['original_tum_path'], x['t_tum_rgb_s']) for x in planned_source]:
        raise DataError('Encoded source cohort differs from pre-SfM frozen identities')

    # RGB-only solver: no depth inputs and no GT poses.
    sfm = work / 'sfm-out'
    command = [sys.executable, '-m', 'echo4d.video_sfm',
               '--video', str(work / 'test-video-from-real-tum-rgb.mp4'),
               '--output', str(sfm), '--fps', str(VIDEO_FPS),
               '--max-frames', str(FRAME_COUNT), '--camera-model', 'PINHOLE',
               '--camera-params', '525,525,319.5,239.5']
    try:
        invoke('desk-75-rgb-only-colmap', command, record, timeout=1800)
    finally:
        nested = sfm / 'report.json'
        if nested.is_file():
            shutil.copyfile(nested, report / 'sfm-run-report.json')
    sfm_report = json.loads((sfm / 'report.json').read_text(encoding='utf-8'))
    record['registration'] = {
        'attempted': FRAME_COUNT,
        'registered': sfm_report['result']['frames_registered']}
    if (not sfm_report['status'].startswith('passed') or
            sfm_report['result']['frames_extracted'] != FRAME_COUNT or
            sfm_report['result']['frames_registered'] != FRAME_COUNT):
        raise DataError('Desk requires all 75 fixed RGB views; no cherry-picking')
    sfm_track = sfm / 'trajectory-scale-unknown.json'
    track = json.loads(sfm_track.read_text(encoding='utf-8'))
    if track.get('units') != 'unknown' or track.get('metric_scale_estimated') is not False:
        raise DataError('SfM track falsely declares a real scale')
    model = sfm / sfm_report['result']['sfm_model_dir']
    for filename in ('trajectory-scale-unknown.json', 'registered_timestamps.csv'):
        shutil.copyfile(sfm / filename, report / filename)
    for filename in ('cameras.txt', 'images.txt', 'points3D.txt'):
        current = model / filename
        if not current.is_file():
            raise DataError(f'Missing registered COLMAP model {filename}')
        if current.stat().st_size <= 25 * 1024**2:
            shutil.copyfile(current, report / filename)

    # Physical depth appears ONLY after the locked RGB SfM is complete.
    record['depth_extraction'] = extract_sensor_depths(archive, work / 'dataset')
    # Preserve original 30ms hard stop, but record the nearest sensor time for
    # every fixed RGB identity first; this does not inspect withheld Z pixels.
    time_audit = sensor_pairing_diagnostics(tum, source, protocol)
    record['sensor_pairing_preflight_sha256'] = persist(
        report / 'sensor-pairing-preflight.json', time_audit)
    record['sensor_pairing_summary'] = {
        k: v for k, v in time_audit.items() if k not in ('pairs', 'first_outside_gap')}
    record['sensor_pairing_first_outside_gap'] = time_audit['first_outside_gap']
    measurements = match_sensor_inputs(tum, source, protocol)
    calibration_spec = {
        'source_kind': 'measured_registered_z_depth',
        'source_description': 'TUM Freiburg1 Desk measured RGB-D sensor camera-Z, not RGB inference',
        'source_dataset_url': URL,
        'depth_index_sha256': file_sha256(tum / 'depth.txt'),
        'depth_numeric_units_per_meter': DEPTH_UNITS_PER_M,
        'anchors': [measurements[source[i]['name']] for i in SCALE_ANCHORS],
        'holdouts': [measurements[source[i]['name']] for i in SCALE_DIAGNOSTICS],
    }
    depth_spec = report / 'measured-depth-evidence.json'
    record['depth_spec_sha256'] = persist(depth_spec, calibration_spec)
    metric = calibrate(sfm_track, model, tum, depth_spec)
    metric_path = report / 'metric-trajectory-from-sensor.json'
    record['metric_sha256'] = persist(metric_path, metric)
    record['measured_m_per_sfm_unit'] = metric['scale_m_per_sfm_unit']
    record['anchor_diagnostics'] = metric['anchor_diagnostics']
    record['scale_check_diagnostics'] = metric['heldout_depth_diagnostics']

    split = {
        'source_kind': 'measured_registered_z_depth',
        'source_description': 'TUM Freiburg1 Desk sensor-measured camera-Z; not RGB-only depth',
        'depth_numeric_units_per_meter': DEPTH_UNITS_PER_M,
        'train': [measurements[name] for name in protocol['train_names']],
        'holdout': protocol['holdout_names'],
        'protocol': protocol,
    }
    split_path = report / 'desk-48train-12holdout.json'
    record['split_sha256'] = persist(split_path, split)
    input_dir = work / 'desk-sensor-recording'
    conversion = convert_metric_sensor(
        metric_path, sfm_track, model, sfm / 'images', tum, split_path,
        input_dir, downsample=DOWNSAMPLE)
    record['conversion'] = conversion
    record['input_manifest_sha256'] = file_sha256(input_dir / 'recording.json')
    record['sensor_provenance_sha256'] = file_sha256(input_dir / 'sensor-provenance.json')
    bounds = depth_only_training_bounds(input_dir / 'recording.json')
    record['training_depth_only_bounds_m'] = bounds
    output = report / 'mesh'
    mesh_result = reconstruct_command(argparse.Namespace(
        recording=input_dir / 'recording.json', output=output,
        bounds=bounds, voxel=VOXEL_M, truncation=None, max_depth=MAX_DEPTH_M,
        max_voxels=MAX_VOXELS, min_observations=MIN_OBSERVATIONS,
        stride=1, limit=None, reference=None, threshold=.05,
        no_scene=False, save_volume=False, quiet=True))
    record['mesh'] = mesh_result
    if (mesh_result['frames'] != 48 or mesh_result['triangles'] <= 0 or
            mesh_result['vertices'] <= 0):
        raise DataError('Desk 48 measured training frames produced no valid Mesh')

    # First opening of withheld geometry depth pixels, after Mesh creation.
    score = evaluate_sensor_metric_holdouts(
        output / 'mesh.ply', input_dir / 'recording.json', metric,
        split, source, tum, model, downsample=DOWNSAMPLE,
        max_gap_s=MAX_GAP_S, max_depth_m=MAX_DEPTH_M)
    if ([x['name'] for x in score['per_frame']] != protocol['holdout_names'] or
            score['gt_camera_poses_loaded'] is not False or
            score['holdout_frames'] != 12):
        raise DataError('Desk score violated frozen independent holdout contract')
    record['holdout_sha256'] = persist(report / 'desk-heldout-metrics.json', score)
    record['heldout_summary'] = {
        k: v for k, v in score.items() if k not in ('per_frame', 'limitations')}

    # GT only for late availability audit. It cannot modify the product.
    truth, missing = gt_after_sfm(tum, source)
    record['gt_audit_only'] = {
        'file_sha256': file_sha256(tum / 'groundtruth.txt'),
        'matched': len(truth), 'missing': missing,
        'not_used_for_scale_fusion_or_depth_scoring': True}
    if missing or len(truth) != FRAME_COUNT:
        raise DataError('Frozen desk scene missing reference GT poses')
    if not score['screening_gate_passed']:
        raise DataError('Desk holdout predeclared 80% coverage / 15cm RMSE screen failed')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    p.add_argument('--archive', type=Path,
                   help='Offline official archive; pinned SHA and bytes still checked')
    p.add_argument('--cohort', choices=MODES, default='v1-frozen',
                   help='V1 preserves failed source list; V2 is a new source-paired experiment')
    args = p.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        'experiment': 'echo4d-tum-freiburg1-desk-full-mesh-' + args.cohort,
        'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'tested_checkout_sha': os.getenv('ECHO_TESTED_CHECKOUT_SHA'),
        'head_sha': os.getenv('ECHO_HEAD_SHA'), 'git_sha': os.getenv('GITHUB_SHA'),
        'run_url': (f'https://github.com/{os.getenv("GITHUB_REPOSITORY")}/actions/runs/'
                    f'{os.getenv("GITHUB_RUN_ID")}') if os.getenv('GITHUB_RUN_ID') else None,
        'declared_limits': [
            'Desk is a different physical scene from freiburg1_xyz.',
            'TUM real RGB re-encoded as test video, not a native phone video.',
            'Dense geometry uses real sensor Z, not RGB-only estimated depth.',
            'Measured scale never uses GT Sim3 or scored geometric holdout depth.',
            'Withheld consistency is same-sensor, not scanned-surface geometric truth.',
            'Frozen research thresholds; failures and raw measurements are preserved.'],
        'commands': []}
    started = time.perf_counter()
    code = 0
    try:
        run(args, receipt)
        receipt['status'] = 'passed_desk_sensor_mesh_screen_not_independent_surface_truth'
    except Exception as exc:
        code = 1
        receipt['status'] = 'failed'
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        print(f'DESK_FULL_MESH_FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
    finally:
        receipt['duration_s'] = round(time.perf_counter()-started, 3)
        persist(args.report_dir / 'report.json', receipt)
        scores = receipt.get('heldout_summary', {})
        lines = [
            '# TUM Freiburg1 Desk — frozen 48/12 real-sensor geometry experiment', '',
            f'- Status: {receipt["status"]}',
            f'- Actual Actions run: {receipt["run_url"]}',
            f'- Official archive SHA256: {receipt.get("source_archive_sha256", "n/a")}',
            f'- Registered RGB: {receipt.get("registration", {}).get("registered", "n/a")}/75',
            f'- Measured scale m/SfM: {receipt.get("measured_m_per_sfm_unit", "n/a")}',
            f'- Original Mesh SHA256: {receipt.get("mesh", {}).get("mesh_sha256", "n/a")}',
            f'- Mesh triangles: {receipt.get("mesh", {}).get("triangles", "n/a")}',
            f'- Heldout valid pixels: {scores.get("valid_measured_pixels", "n/a")}',
            f'- Heldout coverage: {scores.get("coverage", "n/a")}',
            f'- Heldout MAE m: {scores.get("mae_m", "n/a")}',
            f'- Heldout overlap RMSE m: {scores.get("rmse_m", "n/a")}',
            f'- Heldout P95 m: {scores.get("p95_abs_m", "n/a")}',
            f'- Failure: {receipt.get("error", {}).get("message", "none")}', '',
            'Real measured sensor depth; not RGB-only. Same-sensor comparison is NOT',
            'an independent surface scan or proof of smartphone-video generality.']
        (args.report_dir / 'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        all_files = sorted(x for x in args.report_dir.rglob('*') if x.is_file())
        persist(args.report_dir / 'file-sha256.json',
                {str(x.relative_to(args.report_dir)): file_sha256(x)
                 for x in all_files if x.name != 'file-sha256.json'})
    return code


if __name__ == '__main__':
    raise SystemExit(main())

