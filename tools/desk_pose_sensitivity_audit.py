"""Read-only, preregistered desk V2 camera-local pose sensitivity (#15).

Run from repository root with PYTHONPATH=src; never use the result for fitting.
"""
import argparse
import hashlib
import io
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from echo4d.holdout import read_echo_ply, rasterize_mesh_depth
from echo4d.model import Camera, rigid_matrix

# Input and baseline integrity guards below use assert; never permit -O to disable them.
if not __debug__:
    raise RuntimeError('Evidence audit must not run with Python -O (asserts disabled)')

ZIP_SHA = 'df328812537ea4e4667f36ec7cdd9f901ff5c7fe25a1b37022daf81c20538104'
MESH_SHA = '320da61fcd421c0e55aa6fc014750aeaa3e6e2d98b607e9feaae268f83d2a939'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def perturb(axis, kind, sign):
    d = np.eye(4)
    if kind == 'translation':
        d[axis, 3] = sign * .01
    else:
        angle = sign * math.pi / 360
        c, s = math.cos(angle), math.sin(angle)
        a, b = [i for i in range(3) if i != axis]
        orientation = -1 if axis == 1 else 1
        d[a, a] = d[b, b] = c
        d[a, b], d[b, a] = -orientation * s, orientation * s
    return d


def run(archive):
    assert sha(archive.read_bytes()) == ZIP_SHA, 'Wrong original experiment ZIP'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'Bad artifact CRC'
        mesh = z.read('frozen-inputs/mesh/mesh.ply')
        assert sha(mesh) == MESH_SHA, 'Changed Mesh'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mesh.ply'
            path.write_bytes(mesh)
            vertices, faces = read_echo_ply(path)
        split = json.loads(z.read('frozen-inputs/desk-48train-12holdout.json'))
        held = [x['name'] if isinstance(x, dict) else x for x in split['holdout']]
        assert len(held) == 12 and len(split['train']) == 48
        traj = json.loads(z.read('frozen-inputs/metric-trajectory-from-sensor.json'))
        poses = {x['name']: rigid_matrix(x['camera_to_metric_sfm_world']) for x in traj['frames']}
        prior = json.loads(z.read('residual-report.json'))
        assert [f['name'] for f in prior['per_frame']] == held
        c = z.read('frozen-inputs/cameras.txt').decode().splitlines()[-1].split()
        assert c[:4] == ['1', 'PINHOLE', '640', '480']
        camera = Camera(160, 120, *[float(v) / 4 for v in c[4:]])
        frames = []
        for f in prior['per_frame']:
            name = f['name']
            raw = z.read(name + '.npz')
            assert sha(raw) == f['arrays_sha256'], 'Changed original depth arrays'
            with np.load(io.BytesIO(raw), allow_pickle=False) as a:
                measured, original = a['measured_z_m'], a['predicted_z_m']
            valid = (measured >= .15) & (measured <= 4.)
            original_mask = valid & (original >= .15) & (original <= 4.)
            assert int(original_mask.sum()) == f['total']['overlap_pixels']
            frames.append((name, poses[name], measured, original, valid, original_mask))

    cases = [('baseline', None, None, 0)]
    cases += [(f'{kind}-{axis}-{sign:+d}', axis, kind, sign)
              for kind in ('translation', 'rotation') for axis in range(3) for sign in (-1, 1)]
    all_results = []
    total_valid = sum(int(f[4].sum()) for f in frames)
    for label, axis, kind, sign in cases:
        rows = []
        for name, pose, measured, original, valid, original_mask in frames:
            moved = pose if axis is None else pose @ perturb(axis, kind, sign)
            predicted, _ = rasterize_mesh_depth(vertices, faces, camera, moved, far=4.)
            if axis is None:
                assert np.array_equal(predicted, original), 'Baseline raster differs from frozen original'
            mask = valid & (predicted >= .15) & (predicted <= 4.)
            common = mask & original_mask
            e = predicted[mask].astype(np.float64) - measured[mask].astype(np.float64)
            ec = predicted[common].astype(np.float64) - measured[common].astype(np.float64)
            eb = original[common].astype(np.float64) - measured[common].astype(np.float64)
            rows.append({'name': name, 'overlap': int(mask.sum()),
                         'original_retained': int(common.sum()),
                         'original_lost': int((original_mask & ~mask).sum()),
                         'newly_covered': int((mask & ~original_mask).sum()),
                         'sse_m2': float(e @ e),
                         'common_sse_m2': float(ec @ ec),
                         'baseline_common_sse_m2': float(eb @ eb),
                         'paired_common_sse_delta_m2': float(ec @ ec - eb @ eb),
                         'tail_ge_30cm': int(np.count_nonzero(np.abs(e) >= .30))})
        keys = [k for k in rows[0] if k != 'name']
        t = {k: sum(row[k] for row in rows) for k in keys}
        t['coverage'] = t['overlap'] / total_valid
        t['rmse_m'] = math.sqrt(t['sse_m2'] / t['overlap'])
        all_results.append({'case': label, 'total': t, 'per_frame': rows})
    baseline = all_results[0]['total']
    assert baseline['overlap'] == 158851
    assert abs(baseline['sse_m2'] - 4471.624055396125) < 1e-5
    target_sse = baseline['overlap'] * .15 ** 2
    return {'version': 'desk-v2-preregistered-camera-pose-sensitivity-1',
            'evidence_zip_sha256': ZIP_SHA, 'mesh_sha256': MESH_SHA,
            'pose_rule': 'camera_to_world @ local_delta; +/-0.01m XYZ, +/-0.5deg XYZ, one axis at a time',
            'fixed_15cm_target_sse_m2': target_sse,
            'original_fixed_mask_required_sse_reduction_m2': baseline['sse_m2'] - target_sse,
            'all_cases_fail_even_if_lost_original_pixels_have_zero_error': all(
                r['total']['common_sse_m2'] > target_sse for r in all_results[1:]),
            'cases': all_results,
            'limitations': 'Diagnostic on frozen held-out sensor depth; not new independent surface truth or pose calibration.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-zip', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.evidence_zip)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('report_sha256', sha(args.output.read_bytes()))
    for case in result['cases']:
        t = case['total']
        print(case['case'], f"coverage={t['coverage']:.6f}", f"rmse_cm={t['rmse_m']*100:.4f}",
              f"lost={t['original_lost']}", f"common_sse_m2={t['common_sse_m2']:.3f}")
