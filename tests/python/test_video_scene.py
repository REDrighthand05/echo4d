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
