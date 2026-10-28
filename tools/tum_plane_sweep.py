"""Real TUM known-pose plane-sweep MVS -> existing ECHO TSDF; no measured Z.

Distinct from rectified pairwise SGBM. Reuses the frozen 60/48/12 TUM source
selection, existing reconstruction CLI and five-view Mesh renderer. The only
sensor-depth input is depth.txt timestamps used for fixed frame selection.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from echo4d.model import DataError, file_sha256
from echo4d.known_pose_plane_sweep import infer_plane_sweep
from echo4d.known_pose_depth_consensus import regularize_observed_depth
from echo4d.known_pose_depth_fusion import fuse_reprojected_depths
from echo4d.known_pose_mvs_sources import select_plane_sweep_views
from tum_rgb_stereo import (HOLDOUT, TOTAL, rgb_pose_frames,
                            stereo_resolution, load_stereo_rgb,
                            archive_stereo_inputs, render_mesh)
from tum_public_roundtrip import (URL, ROOT, PINNED_ARCHIVE_SHA256,
                                  PINNED_ARCHIVE_BYTES, BOUNDS, download, extract)


def build_plane_sweep_recording(frames: list[dict], target: Path, report: dict,
                                *, max_anchors: int = 48, planes: int = 64,
                                max_sources: int = 4,
                                pairing: str = 'top2',
                                source_visibility: bool = False,
                                visibility_mode: str = 'hard',
                                window_mode: str = 'fixed',
                                depth_regularization: str = 'none',
                                crossview_fusion: bool = False,
                                crossview_local_surface: bool = False) -> Path:
    camera, _, _ = stereo_resolution(320)
    training = [i for i in range(TOTAL) if i not in HOLDOUT]
    selected, selection_stats = select_plane_sweep_views(
        frames, camera, training, max_anchors=max_anchors, max_sources=max_sources)
    report['source_selection'] = selection_stats
    report['selected_source_indices'] = [
        {'left_index': i, 'source_indices': js} for i, js in selected]
    if len(selected) < 6:
        raise DataError(f'Only {len(selected)} eligible train anchors with >=2 overlapping MVS sources')
    target.mkdir(parents=True, exist_ok=False)
    (target / 'color').mkdir()
    (target / 'depth').mkdir()
    cache: dict[int, np.ndarray] = {}
    def rgb(i: int) -> np.ndarray:
        if i not in cache:
            cache[i] = load_stereo_rgb(frames[i]['rgb'], 320)
        return cache[i]
    # First pass estimates source-camera Z from the SAME training RGB and
    # known poses. The second pass tests whether each plane is hidden behind
    # this estimated source surface. No holdout or sensor-depth PNG is read.
    prepass: dict[int, np.ndarray] = {}
    if source_visibility:
        first_pass_stats = []
        for left, partners in selected:
            z, stats = infer_plane_sweep(rgb(left),
                [(rgb(j), frames[j]['pose']) for j in partners],
                camera, frames[left]['pose'], planes=planes, pairing=pairing,
                window_mode=window_mode)
            prepass[left] = z
            first_pass_stats.append({'left_index': left,
                                     'valid_depth_pixels': stats['valid_depth_pixels']})
        report['source_visibility_prepass'] = {
            'method': 'train-only RGB-inferred Z, same pairing and 64-plane budget',
            'total_valid_pixels': sum(s['valid_depth_pixels'] for s in first_pass_stats),
            'per_anchor': first_pass_stats}
    missing_source_z = 0
    entries = []
    saved_diagnostics = []
    diagnostics = []
    for left, partners in selected:
        if left in HOLDOUT or any(j in HOLDOUT for j in partners):
            raise DataError('Holdout RGB cannot enter plane-sweep reconstruction')
        sources = [(rgb(j), frames[j]['pose']) for j in partners]
        if source_visibility:
            missing_source_z += sum(j not in prepass for j in partners)
            source_z = [prepass.get(j, np.zeros((camera.height, camera.width),
                                                dtype=np.float32)) for j in partners]
        else:
            source_z = None
        depth, stats = infer_plane_sweep(rgb(left), sources, camera,
                                        frames[left]['pose'], planes=planes, pairing=pairing,
                                        source_depths=source_z,
                                        visibility_mode=visibility_mode,
                                        window_mode=window_mode)
        if depth_regularization == 'rgb-consensus':
            stats['inference_before_regularization'] = stats.copy()
            depth, consensus = regularize_observed_depth(depth, rgb(left))
            stats['rgb_depth_consensus'] = consensus
            stats['valid_depth_pixels'] = consensus['remaining_valid_pixels']
            stats['unknown_pixels'] = int(depth.size - stats['valid_depth_pixels'])
            stats['valid_fraction'] = stats['valid_depth_pixels'] / depth.size
            stats['adaptive_statistics_stage'] = 'before_regularization'
        elif depth_regularization != 'none':
            raise DataError('Unknown depth regularization mode')
        stats = {'left_index': left, 'source_indices': partners, **stats}
        diagnostics.append(stats)
        if stats['valid_depth_pixels'] < 100:
            continue
        saved_diagnostics.append(stats)
        n = len(entries)
        Image.fromarray(rgb(left)).save(target / 'color' / f'{n:05d}.png')
        np.save(target / 'depth' / f'{n:05d}.npy', depth, allow_pickle=False)
        entries.append({'t': float(frames[left]['t_s'] - frames[0]['t_s']),
                        'cameraToWorld': frames[left]['pose'].flatten(order='F').tolist(),
                        'color': f'color/{n:05d}.png',
                        'depth': f'depth/{n:05d}.npy'})
    if crossview_fusion:
        # Only already saved TRAIN RGB-inferred depth and measured training poses.
        # Never read TUM sensor depth PNGs, holdout RGB, or prior Mesh.
        saved_depths = [np.load(target / e['depth'], allow_pickle=False) for e in entries]
        saved_rgbs = [np.asarray(Image.open(target / e['color']).convert('RGB'))
                      for e in entries]
        saved_poses = [np.asarray(e['cameraToWorld'], dtype=np.float64).reshape(4, 4, order='F')
                       for e in entries]
        # Preserve the existing no-keyword call for the default path and its
        # test doubles; only the opt-in path changes the fusion contract.
        fusion_kwargs = {'local_surface_support': True} if crossview_local_surface else {}
        fused_depths, fusion_receipts = fuse_reprojected_depths(
            saved_depths, saved_poses, saved_rgbs, camera, **fusion_kwargs)
        for e, fused, stats in zip(entries, fused_depths, saved_diagnostics):
            np.save(target / e['depth'], fused, allow_pickle=False)
            stats['inference_before_crossview_fusion'] = stats.copy()
            stats['valid_depth_pixels'] = int(np.count_nonzero(fused))
            stats['unknown_pixels'] = int(fused.size - stats['valid_depth_pixels'])
            stats['valid_fraction'] = stats['valid_depth_pixels'] / fused.size
        report['crossview_fusion'] = {
            'method': 'two independent RGB-consistent training depth projections',
            'input_depth_source': 'RGB-inferred, not measured',
            'train_only': True, 'source_views_per_anchor_max': 6,
            'local_surface_support': crossview_local_surface,
            'tolerance_m': 0.12, 'rgb_mae_threshold': 45.0,
            'newly_supported_depth_pixels': sum(x['added_pixels'] for x in fusion_receipts),
            'adjusted_depth_pixels': sum(x['adjusted_pixels'] for x in fusion_receipts),
            'per_anchor': fusion_receipts,
            'note': 'Newly filled pixels are reprojected inferred observations, not inpainted unknown geometry.'}
    report['plane_sweep'] = {'candidate_anchors': len(selected),
                             'saved_training_frames': len(entries),
                             'planes': planes, 'max_sources': max_sources, 'pairing': pairing,
                             'window_mode': window_mode,
                             'depth_regularization': depth_regularization,
                             'adaptive_statistics_stage': 'before_regularization',
                             'consensus_removed_pixels': sum(x.get('rgb_depth_consensus', {}).get('removed_isolated_pixels', 0) for x in diagnostics),
                             'consensus_adjusted_pixels': sum(x.get('rgb_depth_consensus', {}).get('adjusted_pixels', 0) for x in diagnostics),
                             'adaptive_reference_pixels': sum(x['adaptive_reference_pixels'] for x in diagnostics),
                             'adaptive_valid_depth_pixels': sum(x['adaptive_valid_depth_pixels'] for x in diagnostics),
                             'source_visibility': source_visibility,
                             'visibility_mode': visibility_mode if source_visibility else 'disabled',
                             'penalized_photo_samples': sum(x['penalized_photo_samples'] for x in diagnostics),
                             'source_depth_maps_missing': missing_source_z,
                             'occluded_photo_samples': sum(x['occluded_photo_samples'] for x in diagnostics),
                             'per_anchor': diagnostics,
                             'no_sensor_depth_pixels': True,
                             'train_only': True}
    if len(entries) < 6:
        raise DataError(f'Only {len(entries)} usable RGB plane-sweep depth frames; require >=6')
    data = {'version': 'echo-dataset-1', 'source': 'recorded', 'units': 'meters',
            'name': 'TUM known-pose plane-sweep RGB inferred Z',
            'duration': float(max(entries[-1]['t'], .001)), 'depth_scale': 1,
            'intrinsics': camera.to_dict(), 'frames': entries,
            'notes': 'Depth inferred from RGB ZNCC MVS, not measured. Metres from TUM groundtruth camera poses; depth.txt used for timestamp selection only.'}
    manifest = target / 'recording.json'
    manifest.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf8')
    report['inferred_manifest_sha256'] = file_sha256(manifest)
    return manifest


def execute(args: argparse.Namespace, report: dict) -> None:
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / 'source.tgz'
    if args.archive:
        shutil.copyfile(args.archive, archive)
        report['download'] = {'bytes': archive.stat().st_size,
                              'sha256': file_sha256(archive), 'method': 'offline pinned'}
    else:
        download(archive, report)
    if (report['download']['sha256'] != PINNED_ARCHIVE_SHA256 or
            report['download']['bytes'] != PINNED_ARCHIVE_BYTES):
        raise DataError('Pinned official TUM archive mismatch')
    report['extraction'] = extract(archive, work / 'dataset')
    root = work / 'dataset' / ROOT
    frames = rgb_pose_frames(root)
    report['input'] = {'source_rgb_resolution': [640, 480],
                       'algorithm_resolution': [320, 240],
                       'total': TOTAL, 'train': TOTAL - len(HOLDOUT),
                       'holdout_indices': list(HOLDOUT),
                       'source_rgb_sha256': [f['rgb_sha256'] for f in frames],
                       'depth_png_files_opened': 0,
                       'pose_source': 'TUM groundtruth known metric poses'}
    manifest = build_plane_sweep_recording(frames, work / 'mvs', report,
                                           max_anchors=args.max_anchors, planes=args.planes,
                                           max_sources=args.max_sources, pairing=args.pairing,
                                           source_visibility=args.source_visibility,
                                           visibility_mode=args.visibility_mode,
                                           window_mode=args.window_mode,
                                           depth_regularization=args.depth_regularization,
                                           crossview_fusion=args.crossview_fusion,
                                           crossview_local_surface=args.crossview_local_surface)
    report['archived_inputs'] = archive_stereo_inputs(
        manifest, frames, root, args.report_dir / 'inputs')
    target = args.report_dir / 'reconstruction'
    cmd = [sys.executable, '-m', 'echo4d', 'reconstruct', str(manifest),
           '--output', str(target), '--bounds', *BOUNDS, '--voxel', '0.05',
           '--max-depth', '4', '--max-voxels', '4000000',
           '--min-observations', '1', '--quiet', '--no-scene']
    report['reconstruct_command'] = cmd
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    report['reconstruct_returncode'] = run.returncode
    report['reconstruct_stdout_tail'] = run.stdout[-2000:]
    report['reconstruct_stderr_tail'] = run.stderr[-2000:]
    mesh = target / 'mesh.ply'
    if run.returncode or not mesh.is_file():
        raise DataError('Plane-sweep TSDF reconstruction failed; see stderr')
    report['mesh_sha256'] = file_sha256(mesh)
    report['mesh_output'] = json.loads(run.stdout.strip().splitlines()[-1])
    camera, _, _ = stereo_resolution(320)
    render_mesh(mesh, frames, args.report_dir / 'visual', report, camera)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    p.add_argument('--archive', type=Path)
    p.add_argument('--max-anchors', type=int, default=48)
    p.add_argument('--max-sources', type=int, default=4)
    p.add_argument('--planes', type=int, default=64)
    p.add_argument('--pairing', choices=('top2', 'diverse'), default='top2')
    p.add_argument('--source-visibility', action='store_true',
                   help='Two-pass TRAIN-only RGB depth occlusion test; no measured Z')
    p.add_argument('--visibility-mode', choices=('hard', 'soft'), default='hard',
                   help='Hard veto or bounded soft penalty for RGB-inferred occluders')
    p.add_argument('--window-mode', choices=('fixed', 'adaptive'), default='fixed',
                   help='Use observed 11x11 RGB patches only where 5x5 has weak texture')
    p.add_argument('--depth-regularization', choices=('none', 'rgb-consensus'), default='none',
                   help='Only regularize observed inferred Z; no hole filling or measured depth')
    p.add_argument('--crossview-fusion', action='store_true',
                   help='Reproject matching RGB-inferred training surfaces into holes; require two views')
    p.add_argument('--crossview-local-surface', action='store_true',
                   help='Require adjacent RGB-inferred Z support before source reprojection')
    args = p.parse_args(argv)
    if args.crossview_local_surface and not args.crossview_fusion:
        p.error('--crossview-local-surface requires --crossview-fusion')
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {'experiment': 'tum-xyz-known-pose-cpu-plane-sweep-no-measured-Z',
              'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'checkout_sha': os.environ.get('GITHUB_SHA'), 'source_url': URL,
              'inputs_retained': ['original 640x480 RGB', 'known metric poses',
                                  'camera intrinsics', 'depth.txt timestamp selection'],
              'inputs_removed': ['sensor depth PNG pixels', 'prior Mesh'],
              'source_visibility_enabled': args.source_visibility,
              'visibility_mode': args.visibility_mode if args.source_visibility else 'disabled',
              'window_mode': args.window_mode,
              'depth_regularization': args.depth_regularization,
              'crossview_fusion_enabled': args.crossview_fusion,
              'crossview_local_surface_enabled': args.crossview_local_surface,
              'evidence_level': 'real data experiment; no independent surface truth or #74 acceptance'}
    start = time.perf_counter()
    try:
        execute(args, report)
        report['status'] = 'mesh_generated_unreviewed'
    except Exception as exc:
        report['status'] = 'failed_or_blocked'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        report['elapsed_seconds'] = round(time.perf_counter() - start, 3)
        (args.report_dir / 'report.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf8')
        print(json.dumps({'status': report['status'],
                          'mesh_sha256': report.get('mesh_sha256'),
                          'error': report.get('error')}))
    return 0 if report['status'] == 'mesh_generated_unreviewed' else 1

if __name__ == '__main__':
    raise SystemExit(main())
