"""Post-hoc TUM sensor-Z diagnosis of TWO already reconstructed RGB-only-depth meshes.

This is an evaluation-only script: measured depth is opened AFTER both meshes
and all inferred training depths exist. No reference pixel can modify inference,
mesh fusion, pose selection, or the fixed 48/12 split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from echo4d.holdout import read_echo_ply, rasterize_mesh_depth
from echo4d.model import file_sha256, safe_path
from echo4d.tum import read_rows, nearest_timestamp
from tum_public_roundtrip import ROOT, PINNED_ARCHIVE_SHA256
from tum_rgb_stereo import HOLDOUT, rgb_pose_frames, stereo_resolution


def score_pair(old: np.ndarray, new: np.ndarray, measured: np.ndarray) -> dict:
    """Separate newly covered real surfaces from newly covered incorrect surfaces.

    No score calls a predicted pixel wrong when the sensor has no valid return.
    Common-pixel errors are compared on EXACTLY the same mask.
    """
    if (old.shape != new.shape or old.shape != measured.shape or old.ndim != 2
            or any(not np.isfinite(x).all() or np.any(x < 0) for x in (old, new, measured))):
        raise ValueError('Expected finite nonnegative depth arrays of identical shape')
    truth = (measured >= .15) & (measured <= 4.0)
    a = (old >= .15) & (old <= 4.0)
    b = (new >= .15) & (new <= 4.0)
    common = truth & a & b
    gained = truth & ~a & b
    lost = truth & a & ~b

    def errors(pred, mask):
        if not np.any(mask):
            return {'pixels': 0, 'mae_m': None, 'rmse_m': None,
                    'within_10cm_fraction': None, 'over_30cm_fraction': None,
                    'bias_m': None}
        delta = (pred[mask] - measured[mask]).astype(np.float64)
        err = np.abs(delta)
        return {'pixels': int(mask.sum()), 'mae_m': float(err.mean()),
                'rmse_m': float(np.sqrt(np.mean(delta ** 2))),
                'within_10cm_fraction': float(np.mean(err <= .10)),
                'over_30cm_fraction': float(np.mean(err > .30)),
                'bias_m': float(delta.mean())}

    return {
        'sensor_valid_pixels': int(truth.sum()),
        'old_covered_sensor_pixels': int((truth & a).sum()),
        'new_covered_sensor_pixels': int((truth & b).sum()),
        'old_predicted_pixels': int(a.sum()),
        'new_predicted_pixels': int(b.sum()),
        'gained_sensor_covered_pixels': int(gained.sum()),
        'lost_sensor_covered_pixels': int(lost.sum()),
        'new_only_sensor_agreement': errors(new, gained),
        'old_only_lost_sensor_agreement': errors(old, lost),
        'common_old_error': errors(old, common),
        'common_new_error': errors(new, common),
        'old_all_error': errors(old, truth & a),
        'new_all_error': errors(new, truth & b),
        'sensor_unknown_new_only_pixels': int((~truth & ~a & b).sum()),
        'warning': 'Sensor-invalid pixels are unknown, not evidence of incorrect geometry.'
    }


def sensor_depth(root: Path, t: float, depth_rows: list, depth_times: np.ndarray) -> tuple[np.ndarray, dict]:
    index = nearest_timestamp(depth_times, t, .03)
    if index is None:
        raise ValueError(f'No TUM measured-depth timestamp within 30ms of {t}')
    rel = depth_rows[index][1]
    path = safe_path(root, rel)
    with Image.open(path) as image:
        pixels = np.asarray(image)
    if pixels.shape != (480, 640) or pixels.dtype != np.uint16:
        raise ValueError('TUM measured depth must be native uint16 640x480')
    # TUM sensor depth scale is 5000 units per meter. Use the SAME even-pixel
    # sampling as the 320x240 RGB estimator, without interpolating invalid Z.
    depth = (pixels[::2, ::2].astype(np.float32) / 5000.0)
    return depth, {'timestamp': float(depth_times[index]), 'source_sha256': file_sha256(path),
                   'path': rel, 'scale_units_per_meter': 5000,
                   'sampling': 'native[::2,::2], no interpolation'}


def error_image(old: np.ndarray, new: np.ndarray, truth: np.ndarray) -> Image.Image:
    valid = (truth >= .15) & (truth <= 4)
    oa = (old >= .15) & (old <= 4)
    nb = (new >= .15) & (new <= 4)
    gained = valid & ~oa & nb
    img = np.zeros((*truth.shape, 3), dtype=np.uint8)
    img[valid] = (48, 48, 48)
    img[valid & oa & nb] = (120, 120, 120)
    img[gained & (np.abs(new - truth) <= .10)] = (20, 180, 70)
    img[gained & (np.abs(new - truth) > .10) & (np.abs(new - truth) <= .30)] = (245, 170, 30)
    img[gained & (np.abs(new - truth) > .30)] = (220, 40, 50)
    img[valid & oa & ~nb] = (45, 100, 230)
    return Image.fromarray(img)


def execute(old: Path, new: Path, source_root: Path, output: Path) -> dict:
    root = source_root / ROOT
    if not root.is_dir():
        raise FileNotFoundError(f'Pinned TUM extraction missing: {root}')
    frames = rgb_pose_frames(root)
    depth_rows = read_rows(root / 'depth.txt', 2)
    depth_times = np.array([float(x[0]) for x in depth_rows], dtype=np.float64)
    old_manifest = json.loads((old / 'inputs/recording.json').read_text())
    new_manifest = json.loads((new / 'inputs/recording.json').read_text())
    if (len(old_manifest['frames']) != 48 or len(new_manifest['frames']) != 48 or
            [x['t'] for x in old_manifest['frames']] != [x['t'] for x in new_manifest['frames']]):
        raise ValueError('The compared inferred-depth bundles must share 48 training timestamps')
    old_report = json.loads((old / 'report.json').read_text())
    new_report = json.loads((new / 'report.json').read_text())
    if (old_report['input']['holdout_indices'] != list(HOLDOUT)
            or new_report['input']['holdout_indices'] != list(HOLDOUT)):
        raise ValueError('Frozen holdout cohort changed')
    camera, _, _ = stereo_resolution(320)
    output.mkdir(parents=True, exist_ok=True)
    train = []
    for i, (a, b) in enumerate(zip(old_manifest['frames'], new_manifest['frames'])):
        z0 = np.load(old / 'inputs' / a['depth'], allow_pickle=False)
        z1 = np.load(new / 'inputs' / b['depth'], allow_pickle=False)
        sensor, provenance = sensor_depth(root, frames[0]['t_s'] + float(a['t']), depth_rows, depth_times)
        train.append({'training_index': i, 't_relative_s': a['t'],
                      'sensor': provenance, 'comparison': score_pair(z0, z1, sensor)})
    meshes = {label: read_echo_ply(folder / 'reconstruction/mesh.ply')
              for label, folder in [('old', old), ('new', new)]}
    heldout, montage_rows = [], []
    for idx in (4, 24, 55):
        pose = frames[idx]['pose']
        z0, _ = rasterize_mesh_depth(*meshes['old'], camera, pose, far=4.0)
        z1, _ = rasterize_mesh_depth(*meshes['new'], camera, pose, far=4.0)
        sensor, provenance = sensor_depth(root, frames[idx]['t_s'], depth_rows, depth_times)
        result = score_pair(z0, z1, sensor)
        heldout.append({'index': idx, 'sensor': provenance, 'comparison': result})
        with Image.open(frames[idx]['rgb']) as photo:
            rgb = photo.convert('RGB').resize((320, 240), Image.Resampling.BILINEAR)
        def gray(z):
            valid = (z >= .15) & (z <= 4)
            arr = np.zeros(z.shape, dtype=np.uint8)
            arr[valid] = np.clip(255 - (z[valid] - .15) * 254 / 3.85, 1, 255).astype(np.uint8)
            return Image.fromarray(arr).convert('RGB')
        montage_rows.append([rgb, gray(sensor), gray(z0), gray(z1), error_image(z0, z1, sensor)])
    canvas = Image.new('RGB', (1600, 3 * 270 + 35), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((5, 8), 'Original RGB | measured Z (DIAG ONLY) | old Mesh | new Mesh | new-only: green <=10cm, amber <=30cm, red >30cm, blue lost', fill='black')
    for i, row in enumerate(montage_rows):
        y = 35 + 270 * i
        for j, image in enumerate(row):
            canvas.paste(image, (j * 320, y))
        d.text((5, y + 242), f'heldout {heldout[i]["index"]}: 640x480 source RGB; geometry at 320x240', fill='black')
    canvas.save(output / 'sensor-error-3-heldout.png')
    def aggregate(rows):
        scores = [r['comparison'] for r in rows]
        keys = ['sensor_valid_pixels', 'old_covered_sensor_pixels',
                'new_covered_sensor_pixels', 'gained_sensor_covered_pixels',
                'lost_sensor_covered_pixels', 'sensor_unknown_new_only_pixels']
        totals = {k: sum(s[k] for s in scores) for k in keys}
        for tag in ('new_only_sensor_agreement', 'common_old_error', 'common_new_error'):
            n = sum(s[tag]['pixels'] for s in scores)
            totals[tag] = {'pixels': n,
                           'weighted_mae_m': (sum(s[tag]['pixels'] * s[tag]['mae_m']
                                                  for s in scores if s[tag]['pixels']) / n) if n else None,
                           'fraction_over_30cm': (sum(s[tag]['pixels'] * s[tag]['over_30cm_fraction']
                                                    for s in scores if s[tag]['pixels']) / n) if n else None}
        return totals
    result = {
        'status': 'posthoc_diagnosis_only_no_new_mesh',
        'source': 'TUM fr1_xyz RGB-D, J. Sturm et al., CC BY 4.0',
        'pinned_archive_sha256': PINNED_ARCHIVE_SHA256,
        'source_resolution': [640, 480], 'diagnostic_resolution': [320, 240],
        'sensor_depth_used_for_reconstruction': False,
        'measured_depth_used_after_mesh_generation_only': True,
        'metric_scale_from_known_poses': True,
        'old_mesh_sha256': file_sha256(old / 'reconstruction/mesh.ply'),
        'new_mesh_sha256': file_sha256(new / 'reconstruction/mesh.ply'),
        'train_depth_diagnostic': {'aggregate': aggregate(train), 'frames': train},
        'heldout_mesh_diagnostic': {'aggregate': aggregate(heldout), 'frames': heldout},
        'error_montage_sha256': file_sha256(output / 'sensor-error-3-heldout.png'),
        'limitations': [
            'Sensor-invalid pixels cannot establish correctness or error.',
            'Training sensor depth is used only for post-hoc error localization, not training or tuning.',
            'TUM RGB-D depth and RGB have known registration imperfections.',
            'The three heldout views use fixed measured camera poses; no unknown-pose capability.',
            'This diagnostic neither generates nor improves a mesh; visual acceptance remains FAIL.'
        ]
    }
    (output / 'diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', required=True, type=Path)
    parser.add_argument('--new', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    r = execute(args.old, args.new, args.source_root, args.output)
    print(json.dumps({'status': r['status'],
                      'train': r['train_depth_diagnostic']['aggregate'],
                      'heldout': r['heldout_mesh_diagnostic']['aggregate']}))
