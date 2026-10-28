"""Rebuild a real TUM Mesh from archived RGB-inferred Z without rerunning MVS.

Frozen old inferred maps come from Actions #40692184301. Only the RGB-based
crossview fusion rule changes; measured depth is never opened here.
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

from echo4d.model import file_sha256
from echo4d.known_pose_depth_fusion import fuse_reprojected_depths
from tum_public_roundtrip import ROOT, BOUNDS
from tum_rgb_stereo import HOLDOUT, rgb_pose_frames, load_stereo_rgb, stereo_resolution, archive_stereo_inputs, render_mesh


def replay(old: Path, source_root: Path, target: Path) -> dict:
    frames = rgb_pose_frames(source_root / ROOT)
    old_manifest = json.loads((old / 'inputs/recording.json').read_text())
    entries = old_manifest['frames']
    if len(entries) != 48 or old_manifest['intrinsics']['width'] != 320:
        raise ValueError('Expected pinned 48-frame 320x240 RGB-inferred source')
    if target.exists():
        raise FileExistsError(f'Replay output must not exist: {target}')
    (target / 'recording/color').mkdir(parents=True)
    (target / 'recording/depth').mkdir(parents=True)
    camera, _, _ = stereo_resolution(320)
    poses, rgbs, depths = [], [], []
    for entry in entries:
        absolute_t = frames[0]['t_s'] + float(entry['t'])
        choices = [f for f in frames if f['index'] not in HOLDOUT
                   and abs(f['t_s'] - absolute_t) < 1e-5]
        if len(choices) != 1:
            raise ValueError('Could not match inferred depth to unique frozen training RGB')
        rgbs.append(load_stereo_rgb(choices[0]['rgb'], 320))
        poses.append(np.array(entry['cameraToWorld'], dtype=np.float64).reshape(4, 4, order='F'))
        depths.append(np.load(old / 'inputs' / entry['depth'], allow_pickle=False))
    fused, receipts = fuse_reprojected_depths(
        depths, poses, rgbs, camera, local_surface_support=True)
    for i, (entry, rgb, depth) in enumerate(zip(entries, rgbs, fused)):
        Image.fromarray(rgb).save(target / 'recording' / entry['color'])
        np.save(target / 'recording' / entry['depth'], depth, allow_pickle=False)
    manifest = dict(old_manifest)
    manifest['name'] = 'TUM known-pose RGB MVS with local-surface-supported crossview Z'
    manifest['notes'] = ('RGB-inferred depth only, measured metric poses. Local source Z '
                         'neighborhood support before crossview transfer. No sensor Z or prior Mesh.')
    path = target / 'recording/recording.json'
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    archived = archive_stereo_inputs(path, frames, source_root / ROOT, target / 'inputs')
    cmd = [sys.executable, '-m', 'echo4d', 'reconstruct', str(path),
           '--output', str(target / 'reconstruction'), '--bounds', *BOUNDS,
           '--voxel', '0.05', '--max-depth', '4', '--max-voxels', '4000000',
           '--min-observations', '1', '--quiet', '--no-scene']
    process = subprocess.run(cmd, text=True, capture_output=True, timeout=900)
    mesh = target / 'reconstruction/mesh.ply'
    if process.returncode or not mesh.is_file():
        raise RuntimeError(f'Replay Mesh reconstruction failed: {process.stderr[-2000:]}')
    old_report = json.loads((old / 'report.json').read_text())
    report = {'experiment': 'frozen-inference-local-surface-crossview-replay',
              'checkout_sha': os.environ.get('GITHUB_SHA'),
              'input': old_report['input'],
              'inference_source_checkout_sha': old_report.get('checkout_sha'),
              'inference_source_mesh_sha256': old_report.get('mesh_sha256'),
              'inferred_manifest_sha256': file_sha256(path),
              'archived_inputs': archived,
              'status': 'mesh_generated_unreviewed',
              'mesh_sha256': file_sha256(mesh),
              'method': 'opt-in local surface support on crossview RGB inferred Z',
              'inference_reused': 'old/inputs/depth from real Actions 40692184301',
              'reconstruct_command': cmd,
              'reconstruct_returncode': process.returncode,
              'mesh_output': json.loads(process.stdout.strip().splitlines()[-1]),
              'reconstruct_stdout_tail': process.stdout[-2000:],
              'reconstruct_stderr_tail': process.stderr[-2000:],
              'crossview_fusion': {
                  'local_surface_support': True,
                  'newly_supported_depth_pixels': sum(x['added_pixels'] for x in receipts),
                  'adjusted_depth_pixels': sum(x['adjusted_pixels'] for x in receipts),
                  'per_anchor': receipts,
                  'no_measured_depth_pixels': True,
                  'train_only': True}}
    render_mesh(mesh, frames, target / 'visual', report, camera)
    (target / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return {'mesh_sha256': report['mesh_sha256'],
            'added_inferred_pixels': report['crossview_fusion']['newly_supported_depth_pixels'],
            'mesh': str(mesh), 'input_manifest_sha256': file_sha256(path)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(replay(a.old, a.source_root, a.target)))
