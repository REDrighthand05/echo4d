import json
from pathlib import Path
from argparse import Namespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from echo4d.sfm_eval import similarity_align, evaluate_sfm_poses
from echo4d.video_sfm import read_timestamps, _estimated_track, run_step, run_pipeline
from echo4d.model import DataError


def pose(rot, center):
    m=np.eye(4)
    m[:3,:3]=rot
    m[:3,3]=center
    return m.flatten(order='F').tolist()


def test_umeyama_scale_rotation_translation_not_reflection():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(17,3))
    r=Rotation.from_euler('xyz',[15,27,-8],degrees=True).as_matrix()
    y=(2.5*(r@x.T)).T+[3,-2,5]
    scale,found,t=similarity_align(x,y)
    assert scale==pytest.approx(2.5,rel=1e-12)
    assert np.allclose(found,r,atol=1e-12)
    assert np.allclose(t,[3,-2,5],atol=1e-12)
    assert np.linalg.det(found)==pytest.approx(1)


def test_similarity_reject_degenerate_trajectory():
    x=np.ones((4,3))
    with pytest.raises(DataError,match='degenerate'):
        similarity_align(x,x)


def test_ground_truth_is_posthoc_aligned_never_deployed():
    rng=np.random.default_rng(4)
    centers=rng.normal(size=(5,3))
    r=Rotation.from_euler('z',30,degrees=True).as_matrix()
    est=[pose(np.eye(3),x) for x in centers]
    gt=[pose(r,2*r@x+[1,2,3]) for x in centers]
    track={'units':'unknown','metric_scale_estimated':False,
           'frames':[{'name':f'frame_{i:05d}.png','camera_to_sfm_world':p} for i,p in enumerate(est)]}
    truth={f'frame_{i:05d}.png':p for i,p in enumerate(gt)}
    result=evaluate_sfm_poses(track,truth)
    assert result['matched_frames']==5
    assert result['ate_rmse_m_after_sim3']<1e-12
    assert result['rotation_median_degrees_after_sim3']<1e-5
    assert result['metric_scale_available_in_product'] is False
    assert result['evaluation_only_sim3_scale_m_per_sfm_unit']==pytest.approx(2)


def test_scale_unknown_enforced():
    with pytest.raises(DataError,match='unscaled'):
        evaluate_sfm_poses({'units':'meters','metric_scale_estimated':True,'frames':[1,2,3]}, {})


def test_colmap_world_to_camera_inverse(tmp_path):
    model=tmp_path
    (model/'images.txt').write_text('# header\n1 0.7071067811865476 0 0 0.7071067811865475 2 -1 -3 1 frame_00000.png\n\n'
                                    '2 1 0 0 0 -2 0 0 1 frame_00001.png\n\n'
                                    '3 1 0 0 0 -3 0 0 1 frame_00002.png\n\n')
    frames,count=_estimated_track(model,{f'frame_{i:05d}.png':float(i) for i in range(3)})
    assert count==3
    p=np.asarray(frames[0]['camera_to_sfm_world']).reshape(4,4,order='F')
    assert np.allclose(p[:3,3],[1,2,3])
    assert frames[0]['t_s']==0


def test_unknown_registered_file_rejected(tmp_path):
    (tmp_path/'images.txt').write_text('1 1 0 0 0 0 0 0 1 ../wrong.png\n\n')
    with pytest.raises(DataError):
        _estimated_track(tmp_path,{'frame_00000.png':0})


def test_timestamps_deny_path_traversal_and_duplicates(tmp_path):
    file=tmp_path/'stamps.csv'
    file.write_text('name,t\n../x.png,0\ny.png,1\nz.png,2\n')
    with pytest.raises(DataError,match='safe'):
        read_timestamps(file)
    file.write_text('name,t\na.png,0\nb.png,1\nb.png,2\n')
    with pytest.raises(DataError,match='duplicate'):
        read_timestamps(file)


def test_failing_external_binary_is_not_success(tmp_path):
    receipt={}
    with pytest.raises(DataError,match='failed'):
        run_step(['python','-c','import sys;sys.exit(3)'],receipt,'known_fail')
    assert receipt['commands'][0]['status']=='failed'
    assert receipt['commands'][0]['returncode']==3


def test_invalid_output_does_not_overwrite(tmp_path):
    dest=tmp_path/'already'
    dest.mkdir()
    (dest/'preserve').write_text('yes')
    args=Namespace(output=dest)
    with pytest.raises(DataError,match='already exists'):
        run_pipeline(args)
    assert (dest/'preserve').read_text()=='yes'