"""Known-answer checks; these do not substitute for real TUM sensor calibration."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from echo4d.metric_anchor import (calibrate,estimate_scale_from_three_anchors,
                                  describe_holdout,read_colmap_observations,
                                  sample_observed_point_depths)
from echo4d.model import DataError,file_sha256
from echo4d.colmap import Camera

NAMES=[f'frame_{i:05d}.png' for i in range(10)]


def fixture(root:Path):
    model=root/'model';model.mkdir()
    depths=root/'depths';depths.mkdir()
    # World points are in arbitrary SfM units, while measured sensor Z is 2 meters.
    points=[];features=[]
    for y in [10,17,24,31,38]:
        for x in [13,22,31,40,49]:
            pid=len(points)+1
            points.append(f'{pid} {(x-32)*5/80:.10f} {(y-24)*5/80:.10f} 5 10 20 30 0.1 1 0')
            features.extend([str(x),str(y),str(pid)])
    (model/'points3D.txt').write_text('\n'.join(points)+'\n')
    (model/'cameras.txt').write_text('1 PINHOLE 64 48 80 80 32 24\n')
    image_lines=[]
    for idx,name in enumerate(NAMES):
        image_lines.extend([f'{idx+1} 1 0 0 0 0 0 0 1 {name}',' '.join(features)])
        np.save(depths/f'frame_{idx:05d}.npy',np.full((48,64),2,dtype=np.float32))
    (model/'images.txt').write_text('\n'.join(image_lines)+'\n')
    record={'version':'echo-sfm-trajectory-1','units':'unknown','metric_scale_estimated':False,
            'model_images_txt_sha256':file_sha256(model/'images.txt'),
            'model_cameras_txt_sha256':file_sha256(model/'cameras.txt'),
            'frames':[{'name':n,'t_s':i*.25,'camera_id':1,
            'camera_to_sfm_world':np.eye(4).flatten(order='F').tolist()} for i,n in enumerate(NAMES)]}
    (root/'track.json').write_text(json.dumps(record))
    spec={'source_kind':'measured_registered_z_depth','source_description':'Synthetic 2 meter depth sensor for unit tests',
          'depth_numeric_units_per_meter':1,
          'anchors':[{'name':n,'depth':n.replace('.png','.npy'),'timestamp_delta_seconds':.006} for n in NAMES[:3]],
          'holdouts':[{'name':n,'depth':n.replace('.png','.npy'),'timestamp_delta_seconds':.006} for n in NAMES[3:]]}
    (root/'spec.json').write_text(json.dumps(spec))
    return model,depths


def run(root):
    model,depths=fixture(root)
    return calibrate(root/'track.json',model,depths,root/'spec.json')


def test_correct_external_metric_anchor(tmp_path):
    r=run(tmp_path)
    assert r['scale_m_per_sfm_unit']==pytest.approx(.4)
    assert len(r['frames'])==10
    assert all(x['used_correspondences']==25 for x in r['anchor_diagnostics'])
    assert all(x['mae_m']==pytest.approx(0) for x in r['heldout_depth_diagnostics'])
    assert r['scale_is_from_external_gt_poses'] is False


def test_perturbed_heldout_keeps_anchor_scale(tmp_path):
    model,depths=fixture(tmp_path)
    np.save(depths/'frame_00007.npy',np.full((48,64),2.10,dtype=np.float32))
    result=calibrate(tmp_path/'track.json',model,depths,tmp_path/'spec.json')
    assert result['scale_m_per_sfm_unit']==pytest.approx(.4)
    assert result['heldout_depth_diagnostics'][4]['mae_m']==pytest.approx(.10,abs=1e-6)


def test_disagreement_across_anchors_fails(tmp_path):
    model,depths=fixture(tmp_path)
    np.save(depths/'frame_00002.npy',np.full((48,64),4,dtype=np.float32))
    with pytest.raises(DataError,match='disagree'):
        calibrate(tmp_path/'track.json',model,depths,tmp_path/'spec.json')


def test_zero_depth_anchor_fails_not_fabricated(tmp_path):
    model,depths=fixture(tmp_path)
    np.save(depths/'frame_00002.npy',np.zeros((48,64),dtype=np.float32))
    with pytest.raises(DataError,match='lacks'):
        calibrate(tmp_path/'track.json',model,depths,tmp_path/'spec.json')


def test_wrong_input_model_hash_fails(tmp_path):
    model,_=fixture(tmp_path)
    (model/'cameras.txt').write_text('1 PINHOLE 64 48 100 100 32 24\n')
    with pytest.raises(DataError,match='hash mismatch'):
        calibrate(tmp_path/'track.json',model,tmp_path/'depths',tmp_path/'spec.json')


def test_unknown_metric_evidence_fails(tmp_path):
    model,depths=fixture(tmp_path)
    spec=json.loads((tmp_path/'spec.json').read_text());spec['source_kind']='predicted_relative_depth'
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    with pytest.raises(DataError,match='physical provenance'):
        calibrate(tmp_path/'track.json',model,depths,tmp_path/'spec.json')


def test_invalid_sensor_timestamp_fails(tmp_path):
    model,depths=fixture(tmp_path)
    spec=json.loads((tmp_path/'spec.json').read_text());spec['anchors'][0]['timestamp_delta_seconds']=.2
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    with pytest.raises(DataError,match='timestamp'):
        calibrate(tmp_path/'track.json',model,depths,tmp_path/'spec.json')


def test_unsafe_path_fails(tmp_path):
    model,depths=fixture(tmp_path)
    spec=json.loads((tmp_path/'spec.json').read_text());spec['anchors'][0]['depth']='../bad.npy'
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    with pytest.raises(DataError):
        calibrate(tmp_path/'track.json',model,depths,tmp_path/'spec.json')


def test_no_registered_anchor_fails(tmp_path):
    model,depths=fixture(tmp_path)
    spec=json.loads((tmp_path/'spec.json').read_text());spec['anchors'][0]['name']='missing.png'
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    with pytest.raises(DataError,match='not registered'):
        calibrate(tmp_path/'track.json',model,depths,tmp_path/'spec.json')


def test_depth_discontinuity_excluded(tmp_path):
    model,depths=fixture(tmp_path)
    img=np.full((48,64),2,dtype=np.float32);img[17,22]=2.7
    np.save(depths/'frame_00000.npy',img)
    r=calibrate(tmp_path/'track.json',model,depths,tmp_path/'spec.json')
    assert r['anchor_diagnostics'][0]['used_correspondences']==24
    assert r['scale_m_per_sfm_unit']==pytest.approx(.4)


def test_perspective_reprojection_mismatch_is_rejected():
    camera=Camera(64,48,80.,80.,32.,24.)
    point={1:np.array([0.,0.,5.])}
    obs=sample_observed_point_depths(np.eye(4),point,[(10.,10.,1)],np.full((48,64),2.),camera)
    assert len(obs['depth_m'])==0
    assert obs['counts']['failed_reprojection_check']==1


def test_outlier_has_small_median_effect():
    observations=[]
    for i in range(3):
        x=np.full(25,2.);x[0]=8.
        observations.append({'depth_m':x,'camera_z_sfm_units':np.full(25,5.)})
    scale,_=estimate_scale_from_three_anchors(observations)
    assert scale==pytest.approx(.4)


def test_holdout_does_not_hide_errors():
    r=describe_holdout({'depth_m':np.array([2.,2.5,2.]),
                        'camera_z_sfm_units':np.array([5.,5.,5.]),'counts':{}},.4)
    assert r['rmse_m']==pytest.approx(np.sqrt(.25/3))
    assert r['mae_m']==pytest.approx(.5/3)
