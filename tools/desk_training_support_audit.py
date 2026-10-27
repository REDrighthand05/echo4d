"""Read-only desk V2 training-view support diagnostic, pre-registered in #15.

Nearest-integer projection (np.rint), same frozen Mesh/estimated poses,
48 train views, z-buffer |delta Z| <= .05m; original overlap pixels only.
No GT poses, train sensor depth, reconstruction, or gate changes.
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

ZIP_SHA = 'df328812537ea4e4667f36ec7cdd9f901ff5c7fe25a1b37022daf81c20538104'
MESH_SHA = '320da61fcd421c0e55aa6fc014750aeaa3e6e2d98b607e9feaae268f83d2a939'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def run(archive):
    assert sha(archive.read_bytes()) == ZIP_SHA
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        mesh_bytes = z.read('frozen-inputs/mesh/mesh.ply')
        assert sha(mesh_bytes) == MESH_SHA
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mesh.ply'
            path.write_bytes(mesh_bytes)
            vertices, faces = read_echo_ply(path)
        traj = json.loads(z.read('frozen-inputs/metric-trajectory-from-sensor.json'))
        split = json.loads(z.read('frozen-inputs/desk-48train-12holdout.json'))
        prior = json.loads(z.read('residual-report.json'))
        c = z.read('frozen-inputs/cameras.txt').decode().splitlines()[-1].split()
        assert c[:4] == ['1', 'PINHOLE', '640', '480']
        camera = Camera(160, 120, *[float(v) / 4 for v in c[4:]])
        poses = {x['name']: rigid_matrix(x['camera_to_metric_sfm_world'])
                 for x in traj['frames']}
        train = [x['name'] for x in split['train']]
        held = [x['name'] if isinstance(x, dict) else x for x in split['holdout']]
        assert len(train) == 48 and len(held) == 12 and len(set(train + held)) == 60
        assert [f['name'] for f in prior['per_frame']] == held
        pts, errs, frame_ids = [], [], []
        for j, f in enumerate(prior['per_frame']):
            name = f['name']
            data = z.read(name + '.npz')
            assert sha(data) == f['arrays_sha256']
            with np.load(io.BytesIO(data), allow_pickle=False) as a:
                measured, predicted = a['measured_z_m'], a['predicted_z_m']
            assert measured.shape == (120, 160) and predicted.shape == measured.shape
            overlap = ((measured >= .15) & (measured <= 4) &
                       (predicted >= .15) & (predicted <= 4))
            assert int(overlap.sum()) == f['total']['overlap_pixels']
            y, x = np.nonzero(overlap)
            d = predicted[overlap].astype(np.float64)
            xyz = np.column_stack(((x-camera.cx)/camera.fx*d,
                                   (y-camera.cy)/camera.fy*d, d))
            p = poses[name]
            pts.append(xyz @ p[:3, :3].T + p[:3, 3])
            errs.append(d - measured[overlap].astype(np.float64))
            frame_ids.append(np.full(len(d), j, dtype=np.uint8))
        sse = float(sum(np.dot(e, e) for e in errs))
        pts, errs, frame_ids = map(np.concatenate, (pts, errs, frame_ids))
        assert len(errs) == 158851 and abs(sse - 4471.624055396125) < 1e-5
        support = np.zeros(len(pts), dtype=np.uint8)
        for name in train:
            p = poses[name]
            zbuf, _ = rasterize_mesh_depth(vertices, faces, camera, p, far=4.0)
            q = (pts - p[:3, 3]) @ p[:3, :3]
            with np.errstate(divide='ignore', invalid='ignore'):
                u = np.rint(camera.fx*q[:, 0]/q[:, 2] + camera.cx)
                v = np.rint(camera.fy*q[:, 1]/q[:, 2] + camera.cy)
            inside = ((q[:, 2] > 0) & np.isfinite(u) & np.isfinite(v) &
                      (u >= 0) & (u < camera.width) &
                      (v >= 0) & (v < camera.height))
            ids = np.flatnonzero(inside)
            uu, vv = u[ids].astype(int), v[ids].astype(int)
            seen = ((zbuf[vv, uu] > 0) &
                    (np.abs(zbuf[vv, uu].astype(np.float64)-q[ids, 2]) <= .05))
            support[ids[seen]] += 1
        def stats(mask):
            e = errs[mask]
            tail = np.abs(e) >= .30
            return {'n':len(e), 'tail_n':int(tail.sum()),
                    'tail_rate':float(tail.mean()) if len(e) else None,
                    'sse_m2':float(np.dot(e, e)),
                    'tail_sse_m2':float(np.dot(e[tail], e[tail]))}
        bins = [support == 0, (support >= 1) & (support <= 2), support >= 3]
        keys = ('0', '1-2', '>=3')
        result = {'version':'desk-v2-48-training-view-support-1',
                  'evidence_zip_sha256':ZIP_SHA, 'mesh_sha256':MESH_SHA,
                  'source_pose':'estimated_RGB_COLMAP_sensor_scale_not_GT',
                  'support_rule':'nearest_integer_pixel_np_rint; z>0; in_frame; abs(train_zbuffer_Z - projected_Z)<=0.05m; original fixed Mesh',
                  'tail_rule':'abs(original_overlap_predicted_Z - original_measured_Z)>=0.30m',
                  'fixed_gate':'coverage>=80%; RMSE<=0.15m; original FAIL',
                  'overall':stats(np.ones(len(errs), dtype=bool)),
                  'bins':{k:stats(b) for k,b in zip(keys,bins)},
                  'per_frame':[]}
        for j, name in enumerate(held):
            result['per_frame'].append({'name':name,
                'bins':{k:stats((frame_ids == j) & b) for k,b in zip(keys,bins)}})
        assert sum(b['n'] for b in result['bins'].values()) == len(errs)
        assert sum(b['tail_n'] for b in result['bins'].values()) == result['overall']['tail_n']
        assert abs(sum(b['sse_m2'] for b in result['bins'].values()) - sse) < 1e-6
        assert abs(sum(b['sse_m2'] for f in result['per_frame']
                       for b in f['bins'].values()) - sse) < 1e-6
        result['sse_conservation_abs_error_m2'] = abs(
            sum(b['sse_m2'] for b in result['bins'].values()) - sse)
        result['original_rmse_m'] = math.sqrt(sse/len(errs))
        return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-zip', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.evidence_zip), indent=2,
                                   sort_keys=True, allow_nan=False)+'\n')
    print('report_sha256', sha(a.output.read_bytes()))
