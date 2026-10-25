import argparse
import json
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation
from echo4d.model import Camera, DataError, load_recording, safe_path, rigid_matrix
from echo4d.output import output_transaction
from echo4d.tum import interpolate_pose, nearest_timestamp, import_tum


def minimal():
    return {'version':'echo-1','source':'recorded','units':'meters','duration':1,
            'intrinsics':{'width':2,'height':2,'fx':1,'fy':1,'cx':.5,'cy':.5},
            'frames':[{'t':0,'cameraToWorld':np.eye(4).flatten(order='F').tolist(),'depth':[1,1,1,1]}]}


def save(tmp_path,data):
    p=tmp_path/'recording.json';p.write_text(json.dumps(data),encoding='utf8');return p


def test_inline_round_trip(tmp_path):
    r=load_recording(save(tmp_path,minimal()));_,f,e=next(r.iter_frames())
    assert f.depth.shape==(2,2) and f.depth[0,0]==1 and e['index']==0
    assert len(r.manifest_sha256)==64


@pytest.mark.parametrize('change',[
    lambda d:d.update(units='millimeters'),lambda d:d.update(source='unknown'),lambda d:d.update(duration=0),
    lambda d:d['intrinsics'].update(fx=0),lambda d:d['intrinsics'].update(fx=1e-300),
    lambda d:d['intrinsics'].update(cx=1e300),lambda d:d['intrinsics'].update(width=True),
    lambda d:d['frames'][0].update(depth=[1,float('nan'),1,1]),lambda d:d['frames'][0].update(depth=[1,-1,1,1]),
    lambda d:d['frames'][0].update(t=-1),lambda d:d['frames'].append(d['frames'][0].copy()),
    lambda d:d['frames'][0].update(cameraToWorld=[1]*16),lambda d:d['frames'][0].update(excluded=[1,0,0,0]),
    lambda d:d['frames'][0].update(colors=[256]*12),lambda d:d['frames'][0].update(colors=[.5]*12),
    lambda d:d.update(entities=[{'id':'bad','jointNames':['a','b'],'bones':[[0,2]],'samples':[]}]),
    lambda d:d.update(entities=[{'id':'bad','jointNames':['a','b'],'bones':[[0,1]],'samples':[{'t':2,'joints':[[0,0,0],[0,1,0]]}]}]),
])
def test_invalid_inputs(tmp_path,change):
    d=minimal();change(d)
    with pytest.raises(DataError):load_recording(save(tmp_path,d))


def test_reflection_rejected():
    with pytest.raises(DataError):rigid_matrix(np.diag([1,-1,1,1]).flatten(order='F'))


def test_paths_cannot_escape(tmp_path):
    outer=tmp_path/'outside';outer.write_text('secret');root=tmp_path/'dataset';root.mkdir()
    with pytest.raises(DataError):safe_path(root,'../outside')
    try:(root/'link').symlink_to(outer)
    except OSError:pytest.skip('Host does not allow symlinks; relative escape checked')
    with pytest.raises(DataError):safe_path(root,'link')


def test_numpy_objects_rejected(tmp_path):
    np.save(tmp_path/'depth.npy',np.array([[{'x':1}]*2]*2,dtype=object))
    d=minimal();d['version']='echo-dataset-1';d['frames'][0]['depth']='depth.npy'
    r=load_recording(save(tmp_path,d))
    with pytest.raises(ValueError):list(r.iter_frames())


def test_png_scale_and_mask(tmp_path):
    Image.fromarray(np.array([[1000,2000],[0,3000]],dtype=np.uint16)).save(tmp_path/'depth.png')
    Image.fromarray(np.array([[255,0],[0,0]],dtype=np.uint8)).save(tmp_path/'mask.png')
    d=minimal();d.update(version='echo-dataset-1',depth_scale=1000);d['frames'][0].update(depth='depth.png',mask='mask.png')
    _,f,e=next(load_recording(save(tmp_path,d)).iter_frames())
    assert np.array_equal(f.depth,[[1,2],[0,3]]) and f.excluded[0,0]
    assert len(e['files']['depth']['sha256'])==64


def test_output_rollback_and_no_overwrite(tmp_path):
    out=tmp_path/'result'
    with pytest.raises(RuntimeError):
        with output_transaction(out) as temp:
            (temp/'file').write_text('partial');raise RuntimeError('fail')
    assert not out.exists() and list(tmp_path.iterdir())==[]
    with output_transaction(out) as temp:(temp/'file').write_text('done')
    with pytest.raises(DataError):
        with output_transaction(out):pass
    assert (out/'file').read_text()=='done'


def test_slerp_gaps_and_no_extrapolation():
    t=np.array([0.,1.]);p=np.array([[0,0,0],[2,0,0]]);r=Rotation.from_euler('z',[0,90],degrees=True)
    m=interpolate_pose(t,p,r,.5,1.1)
    assert np.allclose(m[:3,3],[1,0,0])
    assert np.allclose(m[:3,:3],Rotation.from_euler('z',45,degrees=True).as_matrix())
    assert interpolate_pose(t,p,r,-.01,1.1) is None
    assert interpolate_pose(t,p,r,1.01,1.1) is None
    assert interpolate_pose(t,p,r,.5,.1) is None
    assert nearest_timestamp(t,.5,.1) is None and nearest_timestamp(t,.5,.6)==0


def test_subsample_calibration():
    c=Camera(64,48,50.,50.,31.5,23.5);s=c.scaled(2)
    assert (10-s.cx)/s.fx==pytest.approx((20-c.cx)/c.fx) and s.width==32


def test_tum_adapter_end_to_end(tmp_path):
    root=tmp_path/'source';root.mkdir()
    for i in range(2):
        Image.fromarray(np.full((4,4),5000,dtype=np.uint16)).save(root/f'd{i}.png')
        Image.fromarray(np.full((4,4,3),100,dtype=np.uint8)).save(root/f'c{i}.png')
    (root/'depth.txt').write_text('1.025 d0.png\n1.075 d1.png\n')
    (root/'rgb.txt').write_text('1.025 c0.png\n1.075 c1.png\n')
    (root/'groundtruth.txt').write_text('1.0 0 0 0 0 0 0 1\n1.1 .1 0 0 0 0 0 1\n')
    args=argparse.Namespace(dataset=root,output=tmp_path/'converted',intrinsics=[3.,3.,1.5,1.5],size=[4,4],poses='groundtruth.txt',depth_scale=5000.,stride=1,limit=10,downsample=2,max_pose_gap=.11,max_color_gap=.03,world_up='y')
    assert import_tum(args)['frames']==2
    frames=list(load_recording(args.output/'recording.json').iter_frames())
    assert np.all(frames[0][1].depth==1)
    assert frames[0][1].camera_to_world[0,3]==pytest.approx(.025)
    assert frames[1][1].t==pytest.approx(.05)


def test_portrait_preview_preserves_intrinsics_and_valid_dimensions(tmp_path):
    from echo4d.output import preview_recording
    d=minimal();d['intrinsics'].update(width=2,height=120,fx=50,fy=100,cx=.5,cy=59.5)
    d['frames'][0]['depth']=[1]*240
    preview=preview_recording(load_recording(save(tmp_path,d)))
    assert preview['intrinsics']['width']==2
    assert preview['intrinsics']['height']==60
    assert preview['intrinsics']['fy']==50
    assert len(preview['frames'][0]['depth'])==120
