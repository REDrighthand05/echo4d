import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from echo4d.model import DataError
from echo4d.video_scene import build_parser, run, run_stage, verify_textured_glb


def glb(path, textured=True, mode=4):
    import numpy as np
    trimesh = pytest.importorskip('trimesh')
    from PIL import Image
    mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]], process=False)
    if textured:
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=np.array([[0,0],[1,0],[0,1]]), image=Image.new('RGB',(4,4),'red'))
    raw = mesh.export(file_type='glb')
    size = struct.unpack_from('<I', raw, 12)[0]
    doc = json.loads(raw[20:20+size])
    doc['meshes'][0]['primitives'][0]['mode'] = mode
    payload = json.dumps(doc).encode()
    payload += b' ' * (-len(payload)%4)
    binary = raw[20+size:]
    path.write_bytes(struct.pack('<III',0x46546c67,2,20+len(payload)+len(binary))+struct.pack('<II',len(payload),0x4e4f534a)+payload+binary)


def test_reject_untextured_or_point_geometry(tmp_path):
    path = tmp_path/'scene.glb'
    glb(path, textured=False)
    with pytest.raises(DataError, match='texture'):
        verify_textured_glb(path)
    glb(path, mode=0)
    with pytest.raises(DataError, match='triangle'):
        verify_textured_glb(path)
    glb(path)
    assert verify_textured_glb(path)['triangles'] == 1
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(DataError, match='header'):
        verify_textured_glb(path)


def test_stage_failure_retains_actual_log_and_receipt(tmp_path):
    receipt = {'stages':[]}
    with pytest.raises(subprocess.CalledProcessError):
        run_stage('backend', [sys.executable,'-c',"print('REAL FAILURE'); raise SystemExit(7)"], tmp_path, receipt)
    saved = json.loads((tmp_path/'scene-report.json').read_text())
    assert saved['stages'][0]['status'] == 'failed'
    assert 'REAL FAILURE' in (tmp_path/'backend.log').read_text()
    assert len(saved['stages'][0]['log_sha256']) == 64


def test_missing_backend_creates_no_success_output(tmp_path, monkeypatch):
    video = tmp_path/'input.mp4'
    video.write_bytes(b'not a real video')
    args = build_parser().parse_args(['--video',str(video),'--output',str(tmp_path/'out')])
    args.blender_script = Path(__file__)
    monkeypatch.setattr('echo4d.video_scene.shutil.which', lambda _: None)
    with pytest.raises(DataError, match='Missing dependency'):
        run(args)
    assert not args.output.exists()


def test_cli_exposes_same_pipeline():
    from echo4d.cli import build_parser as cli_parser
    args = cli_parser().parse_args(['video-scene','--video','clip.mp4','--output','out'])
    assert args.command == 'video-scene'
    assert args.min_registration == .8


def test_refuse_existing_output(tmp_path):
    args = build_parser().parse_args(['--video','input.mp4','--output',str(tmp_path)])
    with pytest.raises(DataError, match='fresh output'):
        run(args)


def test_registration_failure_stops_before_meshing(tmp_path, monkeypatch):
    video=tmp_path/'video.mp4'
    video.write_bytes(b'fixture')
    args=build_parser().parse_args(['--video',str(video),'--output',str(tmp_path/'out')])
    args.blender_script=Path(__file__)
    monkeypatch.setattr('echo4d.video_scene.shutil.which',lambda _:sys.executable)
    monkeypatch.setattr('echo4d.video_scene.run_sfm',lambda _: {'status':'passed_fixture','result':{'registration_fraction':.1}})
    with pytest.raises(DataError,match='Insufficient registered'):
        run(args)
    report=json.loads((args.output/'scene-report.json').read_text())
    assert report['status']=='failed'
    assert report['stages']==[]
    assert report['config']['min_registration']==.8
    assert not (args.output/'scene.blend').exists()


def test_orchestration_contract_with_explicit_fake_backends(tmp_path, monkeypatch):
    """E2 only: assert handoff filenames/calibration/order, not MVS quality."""
    video=tmp_path/'video.mp4'
    video.write_bytes(b'fixture')
    args=build_parser().parse_args(['--video',str(video),'--output',str(tmp_path/'out')])
    args.blender_script=Path(__file__)
    monkeypatch.setattr('echo4d.video_scene.shutil.which',lambda _:sys.executable)
    def sfm(namespace):
        namespace.output.mkdir()
        frames=[{'name':f'frame_{i:05d}.png','t_s':i,'camera_id':1,'camera_to_sfm_world':[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]} for i in range(3)]
        (namespace.output/'trajectory-scale-unknown.json').write_text(json.dumps({'frames':frames}))
        return {'status':'passed_fixture','result':{'registration_fraction':1,'sfm_model_dir':'text-model-0'}}
    monkeypatch.setattr('echo4d.video_scene.run_sfm',sfm)
    commands=[]
    def stage(name, command, output, receipt):
        commands.append((name,list(map(str,command))))
        receipt['stages'].append({'stage':name,'status':'FAKE_FOR_UNIT_TEST'})
        if name=='camera-model':
            model=output/'undistorted-model'
            (model/'cameras.txt').write_text('1 PINHOLE 640 480 525 525 319.5 239.5\n')
            (model/'images.txt').write_text(''.join(f'{i+1} 1 0 0 0 0 0 0 1 frame_{i:05d}.png\n\n' for i in range(3)))
        if name in ['interface','densify','mesh','refine','texture']:
            target=Path(command[command.index('-o')+1])
            target.write_bytes(b'FAKE_BACKEND_TEST_ONLY')
        if name=='texture':
            glb(output/'scene_textured.glb')
        if name=='blender':
            (output/'scene.blend').write_bytes(b'FAKE_BACKEND_TEST_ONLY')
            (output/'scene.json').write_text('{}')
    monkeypatch.setattr('echo4d.video_scene.run_stage',stage)
    monkeypatch.setattr('echo4d.video_scene.package_scene',lambda output,frames: (output/'scene-delivery.zip').write_bytes(b'FAKE_EXPORT') and {'fixture':'E2 only'})
    receipt=run(args)
    assert receipt['status']=='completed_not_quality_accepted'
    assert not receipt['visual_accepted']
    assert [n for n,_ in commands]==['undistort','camera-model','interface','densify','mesh','refine','texture','blender']
    for n,command in commands:
        if n in ['mesh','refine']:
            assert command[command.index('--close-holes')+1]=='0'
    timeline=json.loads((args.output/'camera-timeline.json').read_text())
    assert timeline['cameras']['1']==[640,480,525,525,319.5,239.5]
    assert [f['t_s'] for f in timeline['frames']]==[0,1,2]
