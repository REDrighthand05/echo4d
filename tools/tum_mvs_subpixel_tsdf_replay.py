"""Real TUM A/B: refine TSDF projection sampling, reuse frozen RGB-inferred Z.

Control is the exact old Mesh in Actions #40692184301. The experiment uses
the same 48 inferred 320x240 train maps and known metric poses. No MVS rerun,
no measured Z, no prior Mesh during reconstruction. Subpixel Z is interpolated
ONLY between agreeing, already inferred source samples; where it cannot,\nold nearest-neighbor sampling is preserved. No new sensor measurements.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from echo4d.model import Camera, file_sha256
from echo4d.known_pose_subpixel_fusion import (
    upsampled_camera, upsample_observed_depth, upsample_observed_rgb)
from tum_public_roundtrip import ROOT, BOUNDS
from tum_rgb_stereo import (HOLDOUT, rgb_pose_frames, load_stereo_rgb,
                            archive_stereo_inputs, render_mesh)


def replay(old: Path, source_root: Path, target: Path) -> dict:
    old_manifest_path = old / 'inputs/recording.json'
    old_manifest = json.loads(old_manifest_path.read_text())
    entries = old_manifest['frames']
    camera = Camera.from_dict(old_manifest['intrinsics'])
    if (len(entries) != 48 or camera.width != 320 or camera.height != 240
            or old_manifest['depth_scale'] != 1 or old_manifest['units'] != 'meters'):
        raise ValueError('Expected frozen 48x320x240 metric RGB-inferred MVS recording')
    if not (old / 'reconstruction/mesh.ply').is_file():
        raise FileNotFoundError('Frozen real control Mesh is required for A/B')
    if target.exists():
        raise FileExistsError(f'Output already exists: {target}')
    (target / 'recording/color').mkdir(parents=True)
    (target / 'recording/depth').mkdir()
    frames = rgb_pose_frames(source_root / ROOT)
    original_valid = interpolated = 0
    receipts = []
    for i, entry in enumerate(entries):
        absolute_t = frames[0]['t_s'] + float(entry['t'])
        matches = [f for f in frames if f['index'] not in HOLDOUT
                   and abs(f['t_s'] - absolute_t) < 1e-5]
        if len(matches) != 1:
            raise ValueError('Could not uniquely match training RGB/pose; no holdout allowed')
        depth = np.load(old / 'inputs' / entry['depth'], allow_pickle=False)
        if depth.shape != (240, 320) or depth.dtype != np.float32:
            raise ValueError('Frozen depth format changed')
        rgb = load_stereo_rgb(matches[0]['rgb'], 320)
        refined, stats = upsample_observed_depth(depth, fallback='nearest')
        rgb_refined = upsample_observed_rgb(rgb)
        np.save(target / 'recording' / entry['depth'], refined, allow_pickle=False)
        Image.fromarray(rgb_refined).save(target / 'recording' / entry['color'])
        original_valid += stats['original_valid_pixels']
        interpolated += stats['interpolated_pixels_not_independent_measurements']
        receipts.append({'frame': i, 'source_rgb_sha256': matches[0]['rgb_sha256'], **stats})
    manifest = dict(old_manifest)
    manifest['intrinsics'] = upsampled_camera(camera).to_dict()
    manifest['name'] = 'TUM MVS same 48 RGB-inferred Z maps, conservative 2x-1 TSDF sampling'
    manifest['notes'] = ('Same original inferred Z at even pixels; odd sites interpolate only '
                         'mutually agreeing RGB-inferred Z, with exact old nearest-depth fallback at edges/unknown. They are not independent observations. '
                         'Measured metric poses retained; no measured depth or prior Mesh.')
    manifest_path = target / 'recording/recording.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    archive = archive_stereo_inputs(manifest_path, frames, source_root / ROOT,
                                    target / 'inputs')
    cmd = [sys.executable, '-m', 'echo4d', 'reconstruct', str(manifest_path),
           '--output', str(target / 'reconstruction'), '--bounds', *BOUNDS,
           '--voxel', '0.05', '--max-depth', '4', '--max-voxels', '4000000',
           '--min-observations', '1', '--quiet', '--no-scene']
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    mesh = target / 'reconstruction/mesh.ply'
    if proc.returncode or not mesh.is_file():
        raise RuntimeError(f'Actual subpixel TSDF failed: {proc.stderr[-2500:]}')
    old_report = json.loads((old / 'report.json').read_text())
    report = {
        'experiment': 'same-frozen-MVS-depth-subpixel-TSDF-projection-ab',
        'checkout_sha': os.environ.get('GITHUB_SHA'),
        'control_mesh_sha256': file_sha256(old / 'reconstruction/mesh.ply'),
        'control_inferred_manifest_sha256': file_sha256(old_manifest_path),
        'control_inference_run': 40692184301,
        'source_input': old_report.get('input'),
        'inferred_depth_source': 'RGB plane-sweep, NOT measured sensor Z',
        'metric_pose_source': 'TUM supplied measured camera trajectory',
        'sensor_depth_png_files_opened': 0,
        'prior_mesh_used_for_reconstruction': False,
        'training_frames': len(entries),
        'original_depth_resolution': [320, 240],
        'fusion_sampling_resolution': [639, 479],
        'original_inferred_valid_pixels': original_valid,
        'interpolated_subpixel_samples_not_new_measurements': interpolated,\n        'edge_fallback': 'nearest old inferred depth, never new measurement',
        'per_frame': receipts,
        'input_archive': archive,
        'inferred_manifest_sha256': file_sha256(manifest_path),
        'reconstruct_command': cmd,
        'reconstruct_returncode': proc.returncode,
        'mesh_output': json.loads(proc.stdout.strip().splitlines()[-1]),
        'mesh_sha256': file_sha256(mesh),
        'status': 'mesh_generated_unreviewed',
        'visual_acceptance': 'NOT AUTOMATICALLY PASSED',
    }
    render_mesh(mesh, frames, target / 'visual', report, camera)
    (target / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return {'mesh_sha256': report['mesh_sha256'], 'triangles': report['mesh_output']['triangles'],
            'interpolated_samples': interpolated, 'status': report['status']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.old, args.source_root, args.target)))
