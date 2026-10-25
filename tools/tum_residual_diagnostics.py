"""Post-hoc analysis of TUM RGB-D holdout residual tails, never a retraining loop.

Run the locked public-data 48/12 experiment first, then evaluate whether a
small number of pixels or measured depth discontinuities account for most of
the squared error. This labels correlations, not proven physical causes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from echo4d.holdout import read_echo_ply, rasterize_mesh_depth
from echo4d.model import DataError, file_sha256, load_recording
from echo4d.residual_diagnostics import diagnose_frames
from tum_heldout import (experiment as original_experiment, URL,
                         PINNED_ARCHIVE_SHA256, MAX_DEPTH)

EXPECTED_HELDOUT = list(range(0, 60, 5))


def analyze(args: argparse.Namespace, report: dict) -> None:
    work = args.work.resolve()
    root = args.report_dir.resolve()
    baseline_dir = root / 'original-holdout'
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline = {'experiment': 'TUM holdout original known-pose real RGB-D reference',
                'status': 'running', 'commands': []}
    original_experiment(argparse.Namespace(work=work,
                                          report_dir=baseline_dir,
                                          archive=args.archive), baseline)
    baseline['status'] = 'completed'
    (baseline_dir / 'report.json').write_text(json.dumps(baseline, indent=2,
                           ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
    report['original_experiment'] = {
        'manifest_sha256': baseline['split']['original_manifest_sha256'],
        'training_manifest_sha256': baseline['split']['training_manifest_sha256'],
        'heldout_indices': baseline['split']['heldout_indices'],
        'training_frames': baseline['split']['training_count'],
        'mesh_sha256': baseline['mesh']['sha256'],
        'run_status': baseline['status'],
    }
    if baseline['split']['heldout_indices'] != EXPECTED_HELDOUT:
        raise DataError('Holdout split changed: cannot compare to registered baseline')
    mesh = work / 'train-fused-48' / 'mesh.ply'
    vertices, faces = read_echo_ply(mesh)
    recording = load_recording(work / 'imported' / 'recording.json')
    observations = []
    for index, frame, evidence in recording.iter_frames():
        if index not in EXPECTED_HELDOUT:
            continue
        pred, _ = rasterize_mesh_depth(vertices, faces, recording.camera,
                                       frame.camera_to_world, far=MAX_DEPTH)
        observations.append({'index': index, 'prediction': pred,
                             'measured': frame.depth, 'excluded': frame.excluded})
    if len(observations) != len(EXPECTED_HELDOUT):
        raise DataError('Some registered heldout frames are missing')
    diagnostic = diagnose_frames(observations, step_m=.10, band_pixels=2)
    prior = baseline['heldout_depth_evaluation']
    for ours, theirs in [('overlap_pixels', 'overlap_pixels'),
                          ('sensor_valid_pixels', 'measured_valid_pixels')]:
        if diagnostic[ours] != prior[theirs]:
            raise DataError('Diagnostic frame population differs from old frozen evaluation')
    for ours, theirs in [('coverage', 'observed_depth_coverage'),
                          ('mae_m', 'mae_m'), ('rmse_m', 'rmse_m'),
                          ('p95_abs_m', 'p95_abs_m'), ('median_abs_m', 'median_abs_m')]:
        if abs(diagnostic[ours] - prior[theirs]) > 1e-10:
            raise DataError(f'Diagnostic scoring changed the frozen {ours} metric')
    report['verification'] = {'original_holdout_metrics_matched': True,
                              'threshold_difference_absolute': 1e-10}
    report['diagnostic'] = diagnostic
    report['raw_mesh_sha256'] = file_sha256(mesh)
    report['pinned_public_archive_sha256'] = PINNED_ARCHIVE_SHA256
    report['interpretation_rules'] = [
        'A large fraction of RMSE may come from very few large residual pixels.',
        'Association with held-out depth discontinuities is a correlation, not a causal attribution.',
        'Do not filter or remove any pixels when quoting the original baseline RMSE and coverage.',
        'Do not claim precision from unobserved scene surfaces or independently unknown poses.',
        'Do not use held-out pixels to train or tune the TSDF (post-hoc diagnostics only).',
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--report-dir', required=True, type=Path)
    parser.add_argument('--archive', type=Path,
                        help='Optional offline TUM tar archive (must match official pinned SHA)')
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'experiment': 'TUM-12-heldout-depth-error-tail-attribution-v1',
        'status': 'running',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_sha': os.environ.get('ECHO_HEAD_SHA', os.environ.get('GITHUB_SHA')),
        'github_run_url': (f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/"
                           f"{os.environ['GITHUB_RUN_ID']}"
                           if os.environ.get('GITHUB_RUN_ID') and os.environ.get('GITHUB_REPOSITORY')
                           else None),
        'source_url': URL, 'source_sha256': PINNED_ARCHIVE_SHA256,
        'preregistered_parameters': {
            'training_frames': 48, 'never_fused_frames': 12,
            'heldout_indices': EXPECTED_HELDOUT,
            'depth_discontinuity_threshold_m': .10,
            'region_dilation_radius_pixels': 2,
            'absolute_error_thresholds_m': [.03,.05,.10,.30,.50,1.0],
            'scoring': 'Use overlap pixels for residuals and all valid measured pixels for coverage',
            'no_fusion_change_or_hyperparameter_sweep': True,
        },
    }
    start = time.perf_counter()
    code = 0
    try:
        analyze(args, report)
        report['status'] = 'passed_execution_not_causal_proof'
    except Exception as exc:
        code = 1
        report['status'] = 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        print(f'RESIDUAL_DIAGNOSTIC_FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
    finally:
        report['elapsed_seconds'] = round(time.perf_counter()-start,3)
        (args.report_dir / 'report.json').write_text(json.dumps(report, indent=2,
                    ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
        d = report.get('diagnostic', {})
        summary = [
            '# TUM held-out RGB-D error-tail diagnostic (NOT a causal proof)', '',
            f"- Status: **{report['status']}**", f"- GitHub Run: {report['github_run_url']}",
            f"- Code: {report['git_sha']}",
            f"- Pinned official dataset SHA256: {PINNED_ARCHIVE_SHA256}",
            f"- Observed depth coverage: {d.get('coverage', 'N/A')}",
            f"- Original conditional MAE m: {d.get('mae_m', 'N/A')}",
            f"- Original conditional RMSE m: {d.get('rmse_m', 'N/A')}",
            f"- Original baseline metrics reproduced: {report.get('verification', {}).get('original_holdout_metrics_matched', False)}",
            '', '## Long-tail thresholds',
        ]
        for k,v in d.get('tails', {}).items():
            summary.append(f"- {k}: n={v['pixels']}, fraction={v['fraction_of_overlap']}, squared-error share={v['squared_error_share']}")
        summary.extend(['', '## Regions (mutually exclusive; descriptive only)'])
        for k,v in d.get('regions', {}).items():
            summary.append(f"- {k}: n={v['overlap_pixels']}, rmse_m={v['rmse_m']}, squared-error share={v['squared_error_share']}")
        summary.extend(['', 'Full per-frame values and commands: report.json; original reconstruction report and depth comparison in original-holdout/',
                        'These numbers DO NOT identify a proven physical cause or laser-scanned surface accuracy.'])
        (args.report_dir / 'SUMMARY.md').write_text('\n'.join(summary) + '\n',encoding='utf8')
        print(json.dumps({'status': report['status'], 'report': str(args.report_dir/'report.json')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())