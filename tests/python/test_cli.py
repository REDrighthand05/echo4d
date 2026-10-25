import json
import numpy as np
from echo4d.cli import main
from echo4d.model import file_sha256


def test_end_to_end_reproducible_mesh_and_hashes(tmp_path,capsys):
    data=tmp_path/'data';out=tmp_path/'out';again=tmp_path/'again'
    assert main(['demo','--output',str(data),'--frames','6'])==0
    assert main(['validate',str(data/'recording.json')])==0
    args=['reconstruct',str(data/'recording.json'),'--bounds','-1.8','-.2','-1.9','1.8','2.2','1.8','--voxel','.08','--quiet']
    assert main(args+['--output',str(out)])==0
    assert main(args+['--output',str(again)])==0
    assert file_sha256(out/'mesh.ply')==file_sha256(again/'mesh.ply')
    report=json.loads((out/'report.json').read_text())
    assert report['frames_integrated']==6 and report['mesh']['triangles']>100
    for name,digest in report['outputs'].items():assert file_sha256(out/name)==digest
    scene=json.loads((out/'scene.echo.json').read_text())
    assert scene['version']=='echo-scene-1' and len(scene['mesh']['vertices'])==report['mesh']['vertices']
    assert np.load(out/'vertices.npy',allow_pickle=False).shape[1]==3
    assert main(args+['--output',str(out)])==2
    assert file_sha256(out/'mesh.ply')==file_sha256(again/'mesh.ply')


def test_cli_errors_are_actionable(tmp_path,capsys):
    assert main(['validate',str(tmp_path/'missing.json')])==2
    assert 'manifest' in json.loads(capsys.readouterr().err)['error'].lower()


def test_zero_truncation_and_extreme_voxel_fail_before_publication(tmp_path,capsys):
    data=tmp_path/'data'; assert main(['demo','--output',str(data),'--frames','2'])==0
    args=['reconstruct',str(data/'recording.json'),'--bounds','-1','-1','-1','1','1','1','--quiet']
    for flags in [['--truncation','0'],['--voxel','1e-300']]:
        assert main(args+flags+['--output',str(tmp_path/'failed')])==2
        assert not (tmp_path/'failed').exists()
