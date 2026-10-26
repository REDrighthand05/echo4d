"""Proof-by-example projection tests; synthetic fixtures are not furniture evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
# Trimesh is optional for main; it is pinned in this PR's dedicated real-object CI.
trimesh=pytest.importorskip('trimesh', reason='Dedicated object extraction CI installs optional trimesh')
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from real_furniture_extract import EvidenceError, extract

def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()

def fixture(tmp_path,second_mask='foreground'):
    root=tmp_path/'input'
    root.mkdir()
    mesh=trimesh.Trimesh(vertices=[[-.5,-.5,2],[.5,-.5,2],[.5,.5,2],[-.5,.5,2]],
          faces=[[0,1,2],[0,2,3]],process=False)
    scene=root/'scene.ply'
    mesh.export(scene)
    frames=[]
    for n in range(2):
        photo=root/f'photo{n}.png'
        # Distinct synthetic capture identity and camera view; these are
        # known-answer fixtures, not evidence of a physical chair.
        image = Image.new('RGB',(64,64),'white')
        if n == 1:
            image.putpixel((0,0),(24,24,24))
        image.save(photo)
        mask=np.zeros((64,64),np.uint8)
        if n==0 or second_mask=='foreground':
            mask[20:45,20:45]=255
        else:
            mask[:10,:10]=255
        Image.fromarray(mask).save(root/f'mask{n}.png')
        depth=np.zeros((64,64),np.float32)
        depth[20:45,20:45]=2
        np.save(root/f'depth{n}.npy',depth)
        camera_pose = np.eye(4)
        if n == 1:
            camera_pose[0,3] = .04
        frames.append({'name':f'view{n}','rgb':{'path':photo.name,'sha256':digest(photo)},
             'mask':{'path':f'mask{n}.png','sha256':digest(root/f'mask{n}.png')},
             'depth':{'path':f'depth{n}.npy','sha256':digest(root/f'depth{n}.npy')},
             'fx':20,'fy':20,'cx':32,'cy':32,
             'camera_to_world_column_major':camera_pose.flatten(order='F').tolist(),
             'mask_annotation_source':'two reviewed source RGB instance masks',
             'reviewed_same_object':True})
    spec={'version':'echo-furniture-evidence-1',
          'scene':{'path':'scene.ply','sha256':digest(scene)},
          'object_category':'chair','instance_id':'chair_test',
          'scale_basis':'externally_measured_sensor_depth_m',
          'reviewer':'Independent source-photo instance annotation',
          'frames':frames,'depth_tolerance_m':.03}
    manifest=root/'input.json'
    manifest.write_text(json.dumps(spec))
    return root,manifest,spec

def test_world_coordinates_and_exact_original_faces(tmp_path):
    root,manifest,_=fixture(tmp_path)
    result=extract(manifest,root,tmp_path/'out')
    assert result['kept_original_faces']==2
    assert result['world_transform']=='identity_no_rescaling_no_repositioning'
    assert result['verified_furniture'] is False
    original=trimesh.load_mesh(root/'scene.ply',process=False)
    extracted=trimesh.load_mesh(tmp_path/'out/observed-object-world.ply',process=False)
    assert np.allclose(np.sort(original.vertices,axis=0),np.sort(extracted.vertices,axis=0))
    assert np.array_equal(np.load(tmp_path/'out/selected-source-face-indices.npy'),[0,1])
    from echo4d.holdout import read_echo_ply
    v, faces = read_echo_ply(tmp_path/'out/observed-object-world.ply')
    assert v.shape == (4,3) and faces.shape == (2,3)
    assert np.array_equal(np.sort(v,axis=0),np.sort(original.vertices,axis=0))
    assert result['ply_scoring_compatibility_attributes']['not_an_original_tsdf_export'] is True
    assert result['ply_scoring_compatibility_attributes']['rgb'] == (
        'constant_grey_127_placeholder_not_measured_color')
    assert result['source_world_frame_id'].startswith('observed-scene-sha256:')

def test_second_image_disagrees(tmp_path):
    root,m,_=fixture(tmp_path,'background')
    with pytest.raises(EvidenceError,match='No original mesh triangle'):
        extract(m,root,tmp_path/'out')
    assert not (tmp_path/'out').exists()

def test_tampered_photo_digest(tmp_path):
    root,m,s=fixture(tmp_path)
    s['frames'][0]['rgb']['sha256']='0'*64
    m.write_text(json.dumps(s))
    with pytest.raises(EvidenceError,match='SHA256'):
        extract(m,root,tmp_path/'out')

def test_only_one_image_not_enough(tmp_path):
    root,m,s=fixture(tmp_path)
    s['frames']=s['frames'][:1]
    m.write_text(json.dumps(s))
    with pytest.raises(EvidenceError,match='2-30'):
        extract(m,root,tmp_path/'out')

def test_unknown_scale_blocked(tmp_path):
    root,m,s=fixture(tmp_path)
    s['scale_basis']='rgb_unknown_scale'
    m.write_text(json.dumps(s))
    with pytest.raises(EvidenceError,match='scale'):
        extract(m,root,tmp_path/'out')

def test_unreviewed_mask_blocked(tmp_path):
    root,m,s=fixture(tmp_path)
    s['frames'][0]['reviewed_same_object']=False
    m.write_text(json.dumps(s))
    with pytest.raises(EvidenceError,match='review'):
        extract(m,root,tmp_path/'out')

def test_wrong_depth_blocks_all_faces(tmp_path):
    root,m,s=fixture(tmp_path)
    path=root/'depth1.npy'
    a=np.load(path)
    a[a>0]=3
    np.save(path,a)
    s['frames'][1]['depth']['sha256']=digest(path)
    m.write_text(json.dumps(s))
    with pytest.raises(EvidenceError,match='No original'):
        extract(m,root,tmp_path/'out')

def test_pose_scale_rejected(tmp_path):
    root,m,s=fixture(tmp_path)
    bad=np.eye(4)
    bad[0,0]=2
    s['frames'][0]['camera_to_world_column_major']=bad.flatten(order='F').tolist()
    m.write_text(json.dumps(s))
    with pytest.raises(EvidenceError,match='nonrigid'):
        extract(m,root,tmp_path/'out')

def test_mirrored_pose_rejected(tmp_path):
    root,m,s=fixture(tmp_path)
    bad=np.eye(4)
    bad[0,0]=-1
    s['frames'][0]['camera_to_world_column_major']=bad.flatten(order='F').tolist()
    m.write_text(json.dumps(s))
    with pytest.raises(EvidenceError,match='mirrored'):
        extract(m,root,tmp_path/'out')

def test_camera_focal_length_required(tmp_path):
    root,m,s=fixture(tmp_path)
    s['frames'][0]['fx']=0
    m.write_text(json.dumps(s))
    with pytest.raises(EvidenceError,match='intrinsics'):
        extract(m,root,tmp_path/'out')

def test_output_cannot_be_overwritten(tmp_path):
    root,m,_=fixture(tmp_path)
    output=tmp_path/'out'
    output.mkdir()
    (output/'precious.txt').write_text('keep')
    with pytest.raises(EvidenceError,match='already exists'):
        extract(m,root,output)
    assert (output/'precious.txt').read_text()=='keep'

def test_path_traversal_rejected(tmp_path):
    root,m,s=fixture(tmp_path)
    s['frames'][0]['mask']['path']='../unsafe.png'
    m.write_text(json.dumps(s))
    with pytest.raises(EvidenceError,match='Traversal'):
        extract(m,root,tmp_path/'out')

def test_room_masquerading_as_object_rejected(tmp_path):
    root,m,s=fixture(tmp_path)
    s['object_category']='room'
    m.write_text(json.dumps(s))
    with pytest.raises(EvidenceError,match='furniture'):
        extract(m,root,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_source_frame_renamed_copy_cannot_supply_second_view(tmp_path):
    root, manifest, spec = fixture(tmp_path)
    from copy import deepcopy
    duplicate = deepcopy(spec['frames'][0])
    duplicate['name'] = 'renamed_capture'
    spec['frames'][1] = duplicate
    manifest.write_text(json.dumps(spec))
    with pytest.raises(EvidenceError, match='Duplicate source RGB'):
        extract(manifest, root, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_copied_photo_with_different_filename_cannot_count_as_new_view(tmp_path):
    root, manifest, spec = fixture(tmp_path)
    import shutil
    shutil.copyfile(root/'photo0.png', root/'copied-view.png')
    spec['frames'][1]['rgb'] = {
        'path':'copied-view.png', 'sha256':digest(root/'copied-view.png')}
    manifest.write_text(json.dumps(spec))
    with pytest.raises(EvidenceError, match='Duplicate source RGB'):
        extract(manifest, root, tmp_path/'out')


def test_distinct_pixel_bytes_at_identical_camera_pose_are_not_new_view(tmp_path):
    root, manifest, spec = fixture(tmp_path)
    spec['frames'][1]['camera_to_world_column_major'] = (
        spec['frames'][0]['camera_to_world_column_major'])
    manifest.write_text(json.dumps(spec))
    with pytest.raises(EvidenceError, match='no independent camera baseline'):
        extract(manifest, root, tmp_path/'out')
