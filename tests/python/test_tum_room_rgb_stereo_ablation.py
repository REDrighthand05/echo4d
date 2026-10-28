"""Synthetic CPU stereo unit checks; these do not constitute a room Mesh run."""
import numpy as np
import pytest
cv2 = pytest.importorskip('cv2', reason='Optional OpenCV stereo tested by dedicated real-room workflow')
from types import SimpleNamespace

from echo4d.model import DataError
from tum_room_rgb_stereo_ablation import (
    choose_stereo_partner, infer_rectified_stereo_depth)


def test_partner_requires_real_rightward_parallax():
    poses = [np.eye(4) for _ in range(3)]
    poses[1][0,3] = .12
    poses[2][1,3] = .12
    assert choose_stereo_partner(poses,0) == 1
    assert choose_stereo_partner([np.eye(4),np.eye(4)],0) is None


def test_stereo_recovers_metric_Z_from_known_baseline_without_sensor_Z():
    h,w = 120,256
    rng = np.random.default_rng(27)
    left = cv2.GaussianBlur(rng.integers(0,256,(h,w,3),dtype=np.uint8),(3,3),.4)
    right = np.zeros_like(left)
    disparity = 12
    right[:,:-disparity] = left[:,disparity:]
    camera = SimpleNamespace(width=w,height=h,fx=120.,fy=120.,cx=w/2,cy=h/2)
    p0,p1 = np.eye(4),np.eye(4)
    p1[0,3] = .10
    depth, receipt = infer_rectified_stereo_depth(left,right,camera,p0,p1)
    assert depth.shape == (h,w)
    assert receipt['valid_original_pixels'] > 1000
    assert .7 < np.median(depth[depth>0]) < 1.3
    assert receipt['metric_scale_source'].startswith('known TUM')


def test_rejects_unusable_baseline_and_mismatched_images():
    camera = SimpleNamespace(width=256,height=120,fx=120.,fy=120.,cx=128.,cy=60.)
    image = np.zeros((120,256,3),dtype=np.uint8)
    with pytest.raises(DataError):
        infer_rectified_stereo_depth(image,image,camera,np.eye(4),np.eye(4))
    right_pose = np.eye(4)
    right_pose[0,3] = .1
    with pytest.raises(DataError):
        infer_rectified_stereo_depth(image[:100],image,camera,np.eye(4),right_pose)


def test_replay_ledger_retains_exact_inputs_original_rgb_and_baseline(tmp_path):
    import hashlib
    import json
    from tum_room_rgb_stereo_ablation import preserve_replay_inputs, ROOT
    imported, inferred, baseline = [tmp_path/name for name in ('import', 'inferred', 'baseline')]
    for folder in (imported, inferred, baseline/'room-96-fine-wide'):
        folder.mkdir(parents=True)
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    association = {'frames': []}
    for i in range(4):
        raw = b'original rgb ' + bytes([i])
        relative = f'rgb/{i}.png'
        write(tmp_path/'source'/ROOT/relative, raw)
        association['frames'].append({'rgb_path': relative, 'rgb_sha256': hashlib.sha256(raw).hexdigest()})
    for name in ('rgb.txt', 'depth.txt', 'groundtruth.txt'):
        write(tmp_path/'source'/ROOT/name, name.encode())
    write(tmp_path/'source'/ROOT/'depth'/'private.png', b'sensor Z must not be copied')
    (imported/'recording.json').write_text('{}')
    (imported/'association.json').write_text(json.dumps(association))
    (inferred/'recording.json').write_text(json.dumps({'frames': [{'color':'color/0.png','depth':'depth/0.npy'}]}))
    write(inferred/'color/0.png', b'actual resized rgb')
    (inferred/'depth').mkdir()
    np.save(inferred/'depth/0.npy', np.array([[0., 1.]], dtype=np.float32))
    (baseline/'room-sparse-quality-report.json').write_text('{}')
    write(baseline/'room-96-fine-wide/mesh.ply', b'actual frozen control bytes')
    target = tmp_path/'retained'
    result = preserve_replay_inputs(imported, inferred, baseline, target,
                                    {'expanded_indices':[0], 'holdout_indices':[1,2,3]})
    ledger = json.loads((target/'ledger.json').read_text())
    assert ledger['original_rgb_indices'] == [0,1,2,3]
    assert not (target/'source/depth').exists()
    assert not result['sensor_depth_arrays_retained']
    assert result['files'] == 14
    assert result['ledger_sha256'] == hashlib.sha256((target/'ledger.json').read_bytes()).hexdigest()
    for entry in ledger['files']:
        payload = (target/entry['path']).read_bytes()
        assert len(payload) == entry['bytes']
        assert hashlib.sha256(payload).hexdigest() == entry['sha256']
    assert np.array_equal(np.load(target/'inferred/depth/0.npy'), [[0.,1.]])
    assert (target/'baseline/mesh.ply').read_bytes() == b'actual frozen control bytes'
    # A changed original cannot acquire a fresh trusted receipt.
    write(tmp_path/'source'/ROOT/'rgb/0.png', b'changed')
    with pytest.raises(DataError, match='checksum changed'):
        preserve_replay_inputs(imported,inferred,baseline,tmp_path/'bad',
                               {'expanded_indices':[0], 'holdout_indices':[1,2,3]})
