"""Desk V2: frozen train-only alignment against TUM motion-capture poses.

Research diagnostic, NOT a mesh fix or independent surface ground truth.
Original TUM and frozen desk ZIP bytes are checked before accessing GT.
No held-out pose participates in the rigid alignment. No sensor-depth pixels
are read, no reconstruction is rerun, and the frozen 15 cm gate is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from echo4d.model import DataError, rigid_matrix
from echo4d.tum import interpolate_pose

if not __debug__:
    raise RuntimeError('Audit requires assertions; Python -O is forbidden')

TUM_ARCHIVE_SHA = 'e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c'
EVIDENCE_ZIP_SHA = 'df328812537ea4e4667f36ec7cdd9f901ff5c7fe25a1b37022daf81c20538104'
MESH_SHA = '320da61fcd421c0e55aa6fc014750aeaa3e6e2d98b607e9feaae268f83d2a939'
METRIC_TRACK_SHA = '5d60549557d38ef2e7a3f864c4f6dac52ccb7e88574a9fe97c9f1080b80af392'
RGB_INDEX_SHA = 'd1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284'
SOURCE_COHORT_SHA = '368546ed7f3137078992289b0d4f23ce2ac03bf471e87ddedbe325bbdf2e95f2'
ROOT = 'rgbd_dataset_freiburg1_desk/'
HOLDOUT_INDICES = (4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59)
GT_BRACKET_MAX_S = 0.1  # predeclared; no posthoc tuning


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_index(data: bytes, columns: int) -> list[list[str]]:
    if len(data) > 64 * 1024 * 1024:
        raise DataError('Index too large')
    rows, last = [], -math.inf
    for line in data.decode('utf-8').splitlines():
        fields = line.partition('#')[0].strip().split()
        if not fields:
            continue
        if len(fields) != columns:
            raise DataError('Unexpected source index columns')
        t = float(fields[0])
        if not math.isfinite(t) or t <= last:
            raise DataError('Nonmonotonic/nonfinite source index time')
        last = t
        rows.append(fields)
    if not rows:
        raise DataError('Empty source index')
    return rows


def validated_source_cohort(rgb: list[list[str]], plan: dict) -> list[dict]:
    """Verify the frozen V2 name/index/path/time ledger against original rgb.txt.

    V2 samples every eighth *eligible* RGB frame, not raw rgb[::8].
    The caller pins the plan bytes to the original archived SHA256.
    """
    if (not isinstance(plan, dict) or plan.get('mode') != 'v2-source-paired'
            or plan.get('stride_domain') != 'eligible_rgb_index'
            or plan.get('selection_complete') is not True
            or plan.get('source_stride') != 8 or plan.get('required_frames') != 75
            or not isinstance(plan.get('source_index_sha256'), dict)
            or plan['source_index_sha256'].get('rgb.txt') != RGB_INDEX_SHA):
        raise DataError('Wrong frozen V2 source-cohort metadata')
    selected = plan.get('selected_frames')
    if not isinstance(selected, list) or len(selected) != 75:
        raise DataError('Wrong frozen V2 75-frame source cohort')
    previous = -1
    for i, row in enumerate(selected):
        name = f'frame_{i:05d}.png'
        if not isinstance(row, dict) or row.get('name') != name:
            raise DataError(f'Frozen source-cohort name/order mismatch at {name}')
        index = row.get('source_rgb_index')
        if type(index) is not int or index <= previous or index >= len(rgb):
            raise DataError(f'Frozen source RGB index invalid/reused at {name}')
        path, stamp = row.get('original_tum_path'), row.get('t_tum_rgb_s')
        if (not isinstance(path, str) or not path.startswith('rgb/')
                or type(stamp) not in (int, float) or not math.isfinite(stamp)
                or path != rgb[index][1] or float(stamp) != float(rgb[index][0])):
            raise DataError(f'Frozen source RGB index/path/time mismatch at {name}')
        previous = index
    return selected


def reference_poses(rgb: list[list[str]], gt: list[list[str]], names: list[str],
                    source_plan: dict) -> tuple[dict, list[dict]]:
    """Original V2 source-cohort association, never synthetic MP4 or raw stride."""
    selected = validated_source_cohort(rgb, source_plan)
    a = np.asarray(gt, dtype=float)
    if not np.isfinite(a).all() or np.any(abs(np.linalg.norm(a[:, 4:8], axis=1) - 1) > .01):
        raise DataError('Nonfinite or non-unit TUM mocap quaternion')
    rotations = Rotation.from_quat(a[:, 4:8])
    out, ledger = {}, []
    for name in names:
        match = re.fullmatch(r'frame_(\d{5})\.png', name)
        if not match:
            raise DataError('Noncanonical frozen frame name')
        i = int(match.group(1))
        if i >= 60:
            raise DataError('Outside original first-60 frozen geometry cohort')
        source = selected[i]
        t = float(source['t_tum_rgb_s'])
        at = int(np.searchsorted(a[:, 0], t))
        exact = at < len(a) and abs(a[at, 0] - t) < 1e-9
        if not exact and (at == 0 or at == len(a) or a[at, 0] - a[at-1, 0] > GT_BRACKET_MAX_S):
            raise DataError(f'No bracketed mocap pose within fixed 100ms for {name}')
        p = interpolate_pose(a[:, 0], a[:, 1:4], rotations, t, GT_BRACKET_MAX_S)
        if p is None:
            raise DataError(f'No synchronized mocap pose for {name}')
        out[name] = p
        ledger.append({'name': name, 'source_rgb_index': source['source_rgb_index'],
                       'original_rgb_timestamp_s': t,
                       'original_rgb_path': source['original_tum_path'],
                       'gt_low_timestamp_s': float(a[at if exact else at-1, 0]),
                       'gt_high_timestamp_s': float(a[at, 0]),
                       'interpolated': not exact})
    return out, ledger


def align_train_only(estimated: dict[str, np.ndarray], reference: dict[str, np.ndarray],
                     train: list[str], holdout: list[str]) -> dict:
    """Fit proper SE(3) on training centers ONLY; metric scale remains 1."""
    if (len(train) != 48 or len(holdout) != 12 or len(set(train)) != 48 or
            len(set(holdout)) != 12 or set(train) & set(holdout)):
        raise DataError('Expected disjoint frozen 48/12 cohort')
    if not set(train + holdout).issubset(estimated) or not set(train + holdout).issubset(reference):
        raise DataError('Missing estimated/reference pose for frozen frame')
    x = np.stack([estimated[n][:3, 3] for n in train])
    y = np.stack([reference[n][:3, 3] for n in train])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise DataError('Nonfinite pose center')
    xc, yc = x-x.mean(axis=0), y-y.mean(axis=0)
    if np.linalg.matrix_rank(xc, tol=1e-8) < 2 or np.linalg.matrix_rank(yc, tol=1e-8) < 2:
        raise DataError('Degenerate train-only pose alignment')
    u, _, vt = np.linalg.svd(yc.T @ xc)
    fix = np.diag([1., 1., 1. if np.linalg.det(u @ vt) > 0 else -1.])
    rot = u @ fix @ vt
    shift = y.mean(axis=0) - rot @ x.mean(axis=0)
    if np.linalg.det(rot) < 0.999:
        raise DataError('Reflected alignment forbidden')
    def score(names):
        rows = []
        for name in names:
            est, gt = estimated[name], reference[name]
            pos = float(np.linalg.norm(rot @ est[:3, 3] + shift - gt[:3, 3]))
            delta = gt[:3, :3].T @ rot @ est[:3, :3]
            angle = float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1))))
            rows.append({'name': name, 'position_error_m': pos, 'rotation_error_degrees': angle})
        return {'count': len(rows), 'position_rmse_m': float(np.sqrt(np.mean([r['position_error_m']**2 for r in rows]))),
                'position_p95_m': float(np.percentile([r['position_error_m'] for r in rows], 95)),
                'rotation_median_degrees': float(np.median([r['rotation_error_degrees'] for r in rows])),
                'per_frame': rows}
    return {'fit_scope': '48 TRAIN camera centers only; NO holdout fit; fixed metric scale 1',
            'scale': 1.0, 'rotation_rowmajor': rot.flatten().tolist(),
            'translation_m': shift.tolist(), 'train': score(train), 'holdout': score(holdout)}


def run(tum_archive: Path, evidence_zip: Path) -> dict:
    if sha_file(tum_archive) != TUM_ARCHIVE_SHA:
        raise DataError('Official TUM archive SHA mismatch')
    if sha_file(evidence_zip) != EVIDENCE_ZIP_SHA:
        raise DataError('Frozen desk experiment ZIP SHA mismatch')
    with tarfile.open(tum_archive, 'r:gz') as t:
        def read(name):
            f = t.extractfile(ROOT + name)
            if f is None:
                raise DataError(f'Missing TUM {name}')
            with f:
                data = f.read(64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise DataError('Oversized TUM index')
            return data
        rgb_bytes, gt_bytes = read('rgb.txt'), read('groundtruth.txt')
    if sha(rgb_bytes) != RGB_INDEX_SHA:
        raise DataError('Unexpected original RGB index SHA')
    rgb, gt = parse_index(rgb_bytes, 2), parse_index(gt_bytes, 8)
    with zipfile.ZipFile(evidence_zip) as z:
        if z.testzip() is not None:
            raise DataError('Corrupt frozen experiment ZIP')
        mesh = z.read('frozen-inputs/mesh/mesh.ply')
        track = z.read('frozen-inputs/metric-trajectory-from-sensor.json')
        split_bytes = z.read('frozen-inputs/desk-48train-12holdout.json')
        plan_bytes = z.read('frozen-inputs/source-cohort-plan.json')
    if sha(mesh) != MESH_SHA or sha(track) != METRIC_TRACK_SHA:
        raise DataError('Frozen Mesh or metric trajectory changed')
    if sha(plan_bytes) != SOURCE_COHORT_SHA:
        raise DataError('Frozen V2 source-cohort plan SHA mismatch')
    source_plan = json.loads(plan_bytes)
    split = json.loads(split_bytes)
    def names(role):
        return [v['name'] if isinstance(v, dict) else v for v in split[role]]
    train, held = names('train'), names('holdout')
    expected = {f'frame_{i:05d}.png' for i in range(60)}
    if (len(train) != 48 or len(held) != 12 or set(train + held) != expected or
            set(held) != {f'frame_{i:05d}.png' for i in HOLDOUT_INDICES}):
        raise DataError('Frozen 48/12 names or holdout indices changed')
    traj = json.loads(track)
    est = {p['name']: rigid_matrix(p['camera_to_metric_sfm_world']) for p in traj['frames']}
    if len(est) != len(traj['frames']):
        raise DataError('Duplicate estimated frame names')
    ref, ledger = reference_poses(rgb, gt, train + held, source_plan)
    by_name = {row['name']: row for row in source_plan['selected_frames']}
    for row in split['train'] + split['holdout']:
        if isinstance(row, dict):
            source = by_name[row['name']]
            if ('source_rgb_time_s' in row and
                    float(row['source_rgb_time_s']) != float(source['t_tum_rgb_s'])):
                raise DataError('Frozen split/source-cohort timestamp mismatch')
            if ('depth' in row and row['depth'] != source['source_depth_path']):
                raise DataError('Frozen split/source-cohort depth identity mismatch')
    comparison = align_train_only(est, ref, train, held)
    return {'version': 'desk-v2-external-mocap-pose-diagnostic-v1',
            'status': 'POSE_DIAGNOSTIC_ONLY_NOT_MESH_ACCEPTANCE',
            'tum_archive_sha256': TUM_ARCHIVE_SHA, 'rgb_index_sha256': sha(rgb_bytes),
            'mocap_index_sha256': sha(gt_bytes), 'frozen_experiment_zip_sha256': EVIDENCE_ZIP_SHA,
            'mesh_sha256': MESH_SHA, 'metric_trajectory_sha256': METRIC_TRACK_SHA,
            'split_sha256': sha(split_bytes), 'source_cohort_plan_sha256': sha(plan_bytes),
            'source_timestamps': ledger,
            'max_mocap_bracket_span_s': GT_BRACKET_MAX_S,
            'comparison': comparison,
            'limitations': ['Independent TUM mocap trajectory is not independent surface truth.',
                            'No heldout depth pixels read; no Mesh or 15cm gate improvement measured.',
                            'TUM mocap optical-frame calibration and timestamp offset are inherited from TUM; not independently recalibrated.',
                            'Training-only SE3 alignment does not alter the frozen sensor-derived metric scale.',
                            'No GT-derived correction is applied to production reconstruction.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-tum-archive', type=Path, required=True)
    parser.add_argument('--frozen-evidence-zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.official_tum_archive, args.frozen_evidence_zip)
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as exc:
        # Persist a structured failure without fabricating a numerical score.
        result = {'version': 'desk-v2-external-mocap-pose-diagnostic-v1',
                  'status': 'BLOCKED_NO_VALIDATED_POSE_REFERENCE',
                  'failure_type': type(exc).__name__, 'failure_reason': str(exc),
                  'geometry_accepted': False, 'holdout_pose_score_available': False}
        args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8')
        print('BLOCKED; failure receipt', args.output, file=sys.stderr)
        sys.exit(2)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('report_sha256', sha_file(args.output))
    print('train_position_rmse_m', result['comparison']['train']['position_rmse_m'])
    print('holdout_position_rmse_m', result['comparison']['holdout']['position_rmse_m'])
    print('not a mesh or depth accuracy PASS')
