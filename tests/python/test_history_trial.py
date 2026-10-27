"""Trial receipt tests; synthetic bytes/poses, not real scene acceptance."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import tum_history_real_scene as trial
from echo4d.model import DataError, file_sha256
from echo4d.fusion import Mesh
from echo4d.output import write_ply


def test_trial_metadata_raw_bytes_and_no_overwrite(tmp_path):
    source=tmp_path/'input.json';raw=b'{ "observed": true }\r\n';source.write_bytes(raw)
    dest=tmp_path/'archive';receipt=trial.archive_metadata({'input.json':source},dest)
    assert (dest/'input.json').read_bytes()==raw
    assert receipt['input.json']['sha256']==file_sha256(source)
    assert json.loads((dest/'archive.json').read_text())==receipt
    with pytest.raises(DataError):
        trial.archive_metadata({'input.json':source},dest)


def test_trial_metadata_change_rolls_back(tmp_path,monkeypatch):
    source=tmp_path/'input.json';source.write_text('{}')
    original=trial._json_snapshot
    def snapshot(path):
        result=original(path);path.write_text('{ }');return result
    monkeypatch.setattr(trial,'_json_snapshot',snapshot)
    with pytest.raises(DataError,match='changed'):
        trial.archive_metadata({'input.json':source},tmp_path/'archive')
    assert not (tmp_path/'archive').exists()


@pytest.mark.parametrize('name',['../escape.json','nested/file.json'])
def test_trial_metadata_names_cannot_escape(tmp_path,name):
    source=tmp_path/'input.json';source.write_text('{}')
    with pytest.raises(DataError):
        trial.archive_metadata({name:source},tmp_path/'archive')


def test_each_trial_view_has_actual_hash_intrinsics_and_pose(tmp_path):
    dataset=tmp_path/'dataset';dataset.mkdir();frames=[]
    for i in range(3):
        Image.fromarray(np.full((24,32,3),50+i,np.uint8)).save(dataset/f'{i}.png')
        np.save(dataset/f'{i}.npy',np.ones((24,32),np.float32))
        frames.append({'t':float(i),'color':f'{i}.png','depth':f'{i}.npy',
                       'cameraToWorld':np.eye(4).flatten(order='F').tolist()})
    (dataset/'recording.json').write_text(json.dumps({'version':'echo-dataset-1','source':'synthetic',
        'units':'meters','duration':2.,'depth_scale':1.,'intrinsics':{'width':32,'height':24,
        'fx':30.,'fy':30.,'cx':15.5,'cy':11.5},'frames':frames}))
    vertices=np.array([[-.2,-.2,1],[.2,-.2,1],[0,.2,1]],np.float32)
    mesh=Mesh(vertices=vertices,faces=np.array([[0,1,2]],np.int32),
              normals=np.tile([0.,0.,1.],(3,1)),colors=np.full((3,3),120,np.uint8),
              observations=np.ones(3,np.uint32),first_frame=np.zeros(3,np.int32),
              last_frame=np.zeros(3,np.int32))
    ply=tmp_path/'mesh.ply';write_ply(ply,mesh)
    out=tmp_path/'out';out.mkdir();receipt=trial.visualize(out,dataset,ply)
    assert receipt['mesh_sha256']==file_sha256(ply)
    assert len(receipt['views'])==5
    for view in receipt['views']:
        name=view.get('mesh_projection',view.get('file'))
        assert view['render_sha256']==file_sha256(out/'views'/name)
        assert view['render_intrinsics']['width']==8
        assert len(view['cameraToWorld'])==16
        if 'source_file' in view:
            assert view['source_image_sha256']==file_sha256(out/'views'/view['source_file'])
            assert view['observation_role']=='training_view_not_holdout'
