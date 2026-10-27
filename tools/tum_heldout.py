"""Pre-registered TUM RGB-D holdout experiment with independently withheld depth frames.

Data/source contract reuses the pinned official TUM download from Issue #7.
Geometry is compared with truly withheld sensor images, not copied mesh outputs.
These are independent observations, NOT laser-scan surface ground truth.
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

import numpy as np
from PIL import Image

from echo4d.holdout import read_echo_ply, evaluate_frames, rasterize_mesh_depth
from echo4d.model import DataError, file_sha256, load_recording
from tum_public_roundtrip import (download, extract, invoke, URL, ROOT, BOUNDS,
                                  VOXEL, PINNED_ARCHIVE_SHA256, PINNED_ARCHIVE_BYTES)

HOLDOUT_PERIOD = 5            # indices 0,5,...,55, predeclared; 12 of 60
FRAMES = 60
DOWNSAMPLE = 4
MAX_DEPTH = 4.0


def build_training_split(import_dir: Path, output: Path) -> dict:
    source = import_dir / 'recording.json'
    base = json.loads(source.read_text(encoding='utf-8'))
    frames = base['frames']
    if len(frames) != FRAMES:
        raise DataError(f'Expected exactly {FRAMES} pinned input frames; received {len(frames)}')
    heldout = list(range(0, FRAMES, HOLDOUT_PERIOD))
    train = [i for i in range(len(frames)) if i not in heldout]
    assert len(train) == 48 and len(heldout) == 12
    training_record = dict(base)
    training_record['name'] = base['name'] + '-train-48-of-60'
    training_record['frames'] = [frames[i] for i in train]
    training_record['notes'] = ('TUM frames at indices [0,5,...,55] are excluded from TSDF '
                                'and reserved for out-of-fit depth evaluation.')
    # Save manifest in same folder so relative source paths still resolve.
    output.write_text(json.dumps(training_record, ensure_ascii=False,
                                 allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')
    checked = load_recording(output)
    if len(checked) != 48:
        raise DataError('Split manifest is malformed')
    original = load_recording(source)
    actual_training_times = {entry['t'] for entry in checked.entries}
    actual_holdout_times = {original.entries[i]['t'] for i in heldout}
    if actual_training_times & actual_holdout_times:
        raise DataError('Data leakage: held-out time occurs in the training manifest')
    # Preserve source image hashes so the withheld observations can be audited.
    heldout_sha = []
    for i in heldout:
        entry = original.entries[i]
        heldout_sha.append({'index': i, 't': entry['t'],
                            'depth_sha256': file_sha256(import_dir / entry['depth']),
                            'rgb_sha256': file_sha256(import_dir / entry['color'])})
    return {'original_manifest_sha256': file_sha256(source),
            'training_manifest_sha256': file_sha256(output),
            'training_count': len(train), 'holdout_count': len(heldout),
            'train_indices': train, 'heldout_indices': heldout,
            'heldout_source_sha256': heldout_sha,
            'time_overlap_count': 0}


def save_depth_triptych(recording, index: int, mesh_vertices: np.ndarray,
                        mesh_faces: np.ndarray, path: Path) -> None:
    """Diagnostic pixel-level visual; RGB photo not needed for this report."""
    target = None
    for i, frame, _ in recording.iter_frames():
        if i == index:
            target = frame
            break
    if target is None:
        raise DataError('Unknown heldout diagnostic frame')
    predicted, _ = rasterize_mesh_depth(mesh_vertices, mesh_faces, recording.camera,
                                        target.camera_to_world, far=MAX_DEPTH)
    measured = target.depth
    def gray(depth):
        valid = depth > 0
        out = np.zeros((*depth.shape, 3), np.uint8)
        val = np.clip(255 - 220 * depth / MAX_DEPTH, 0, 255).astype(np.uint8)
        out[valid] = val[valid, None]
        return out
    observed = gray(measured)
    modeled = gray(predicted)
    error = np.zeros_like(observed)
    overlap = (measured >= .15) & (measured <= MAX_DEPTH) & (predicted >= .15) & (predicted <= MAX_DEPTH)
    if target.excluded is not None:
        overlap &= ~target.excluded
    error[overlap, 0] = np.clip(np.abs(predicted[overlap] - measured[overlap]) / .2 * 255, 0, 255).astype('u1')
    # Uncovered measured pixels appear blue, to avoid hiding their absence.
    missing = (measured >= .15) & (measured <= MAX_DEPTH) & ~overlap
    error[missing, 2] = 255
    tile = np.concatenate([observed, modeled, error], axis=1)
    Image.fromarray(tile).resize((tile.shape[1] * 3, tile.shape[0] * 3),
                                  resample=Image.Resampling.NEAREST).save(path)


def experiment(args: argparse.Namespace, r: dict) -> None:
    work = args.work.resolve()
    report = args.report_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / 'source.tgz'
    if args.archive:
        provided = args.archive.resolve()
        if not provided.is_file():
            raise DataError('Offline archive missing')
        shutil.copyfile(provided, archive)
        r['download'] = {'method': 'provided offline file, origin not independently authenticated',
                         'bytes': archive.stat().st_size, 'sha256': file_sha256(archive)}
        if (r['download']['bytes'] != PINNED_ARCHIVE_BYTES or
                r['download']['sha256'] != PINNED_ARCHIVE_SHA256):
            raise DataError('Offline data differs from the pinned TUM public archive')
    else:
        download(archive, r)
    r['extraction'] = extract(archive, work / 'dataset')
    imported = work / 'imported'
    invoke('import-TUM-true-pose-60', [sys.executable, '-m', 'echo4d', 'import-tum',
           str(work / 'dataset' / ROOT), '--output', str(imported), '--intrinsics',
           '525', '525', '319.5', '239.5', '--stride', '12', '--limit', '60',
           '--downsample', '4', '--world-up', 'z'], r)
    split = build_training_split(imported, imported / 'recording.train.json')
    r['split'] = split
    train_recon = work / 'train-fused-48'
    invoke('tsdf-train-48-only', [sys.executable, '-m', 'echo4d', 'reconstruct',
           str(imported / 'recording.train.json'), '--output', str(train_recon),
           '--bounds', *BOUNDS, '--voxel', VOXEL, '--max-depth', str(MAX_DEPTH),
           '--min-observations', '2', '--quiet', '--no-scene'], r)
    r['mesh'] = {'sha256': file_sha256(train_recon / 'mesh.ply'),
                 'report_sha256': file_sha256(train_recon / 'report.json')}
    reconstructed_report = json.loads((train_recon / 'report.json').read_text())
    r['mesh']['geometry'] = reconstructed_report['mesh']
    if reconstructed_report['frames_integrated'] != 48:
        raise DataError('Data leakage: expected exactly 48 fused frames')
    used_indices = [entry['frame'] for entry in reconstructed_report['integration']]
    if used_indices != list(range(48)):
        raise DataError('TSDF integration report indicates unexpected frame set')
    r['mesh']['integrated_frame_count'] = len(used_indices)
    vertices, faces = read_echo_ply(train_recon / 'mesh.ply')
    full_recording = load_recording(imported / 'recording.json')
    r['heldout_depth_evaluation'] = evaluate_frames(vertices, faces, full_recording,
                                                    split['heldout_indices'], max_depth=MAX_DEPTH)
    # The numerical check is not an unearned accuracy PASS: only enforce basic observability.
    if r['heldout_depth_evaluation']['overlap_pixels'] < 100:
        raise DataError('Too little independent overlap for a defensible score')
    worst = max(r['heldout_depth_evaluation']['per_frame'],
                key=lambda x: x['score']['mae_m'])['index']
    r['visual_diagnostic_frame_index'] = worst
    save_depth_triptych(full_recording, worst, vertices, faces,
                        report / 'holdout-depth-comparison.png')
    shutil.copyfile(train_recon / 'mesh.ply', report / 'train-only-mesh.ply')
    r['claim'] = ('Held-out real sensor depth view-consistency (estimated surface with known TUM poses); '
                  'NOT independent laser-scan surface accuracy; missing areas are separately counted.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--archive', type=Path, help='Offline re-run on pinned dataset tarball')
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    r = {'experiment': 'TUM-freiburg1-xyz-12-view-heldout-depth-v1',
         'started_utc': datetime.now(timezone.utc).isoformat(),
         'status': 'running',
         'repo_sha': os.environ.get('ECHO_HEAD_SHA', os.environ.get('GITHUB_SHA')),
         'actions_url': (f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/"
                         f"{os.environ.get('GITHUB_RUN_ID')}"
                         if os.environ.get('GITHUB_RUN_ID') else None),
         'dataset_url': URL,
         'pinned_dataset_sha256': PINNED_ARCHIVE_SHA256,
         'pinned_dataset_bytes': PINNED_ARCHIVE_BYTES,
         'parameters': {'total_imported_frames': FRAMES, 'train_frames': 48,
                        'heldout_frames': 12, 'holdout_rule': 'index % 5 == 0',
                        'import_stride': 12, 'downsample': DOWNSAMPLE,
                        'camera': 'TUM fr1 known groundtruth',
                        'voxel_m': float(VOXEL),
                        'tsdf_bounds_m': [float(v) for v in BOUNDS]},
         'method_qualifications': [
             'No heldout RGB-D depth samples are integrated into the train-only TSDF.',
             'The TUM reference pose and RGB-D are shared-sensor observations, not independently measured 3D surfaces.',
             '12 temporal heldout views are correlated with 48 neighbouring training views.',
             'This holdout evaluates known-pose, measured-depth TSDF generalization, not ordinary video SfM or monocular metric depth.',
             'Conditional depth error is measured only where mesh and held-out depth overlap.',
             'Missing predictions are included in measured-depth coverage, never silently called correct.',
         ], 'commands': []}
    start = time.perf_counter()
    code = 0
    try:
        experiment(args, r)
        r['status'] = 'passed_execution_not_accuracy_threshold'
    except Exception as exc:
        code = 1
        r['status'] = 'failed'
        r['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        print(f'HOLDOUT_EXPERIMENT_FAILED {type(exc).__name__}: {exc}', file=sys.stderr)
    finally:
        r['elapsed_seconds'] = round(time.perf_counter() - start, 3)
        (args.report_dir / 'report.json').write_text(
            json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
        scored = r.get('heldout_depth_evaluation', {})
        summary = [
            '# TUM held-out real depth validation', '',
            f"- Experiment status: **{r['status']}** (execution only, NOT accuracy certification)",
            f"- [GitHub Actions run]({r['actions_url']})",
            f"- Repo commit: {r['repo_sha']}",
            f"- TUM source SHA-256: {r.get('download', {}).get('sha256','not_downloaded')}",
            f"- Training / never-fused heldout: {r.get('split',{}).get('training_count','N/A')} / {r.get('split',{}).get('holdout_count','N/A')}",
            f"- Mesh vertices: {r.get('mesh',{}).get('geometry',{}).get('vertices','N/A')}",
            f"- Mesh triangles: {r.get('mesh',{}).get('geometry',{}).get('triangles','N/A')}",
            f"- Measured sensor pixels: {scored.get('measured_valid_pixels','N/A')}",
            f"- Coverage of observed depths: {scored.get('observed_depth_coverage','N/A')}",
            f"- Conditional MAE m: {scored.get('mae_m','N/A')}",
            f"- Conditional RMSE m: {scored.get('rmse_m','N/A')}",
            f"- Conditional p95 absolute m: {scored.get('p95_abs_m','N/A')}",
            f"- Error: {r.get('error',{}).get('message','none')}", '',
            '**This is held-out depth consistency, not ground-truth mesh-to-laser-scanner accuracy.**',
            'Review report.json for each heldout frame, overlap, errors, input hashes and commands.',
            'Diagnostic PNG: measured Z / rendered Z / red overlap errors and blue missing-model pixels.'
        ]
        (args.report_dir / 'SUMMARY.md').write_text('\n'.join(summary) + '\n', encoding='utf8')
        print(json.dumps({'status': r['status'], 'report': str(args.report_dir / 'report.json')},
                         ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
