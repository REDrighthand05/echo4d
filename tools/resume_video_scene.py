"""Resume the immutable #141 densify failure; never rerun/exchange its SfM."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from echo4d.model import file_sha256
from echo4d.video_scene import dense_and_export, write_json

output = Path('video-result').resolve()
original = json.loads((output/'scene-report.json').read_text())
stage=original['stages'][-1]['stage']
if original['status'] != 'failed' or stage not in ['densify','mesh']:
    raise ValueError('Expected the fixed densify or mesh-project export failure')
if original['registration']['frames_registered'] != 90 or original['registration']['frames_extracted'] != 90:
    raise ValueError('Not the fixed 90/90 COLMAP result')
if file_sha256(Path('trial-source/input.mp4')) != original['source_video_sha256']:
    raise ValueError('Source video changed')
if file_sha256(output/'sfm/trajectory-scale-unknown.json') != original['registration']['trajectory_sha256']:
    raise ValueError('Frozen SfM trajectory changed')
write_json(output/f'scene-report.original-{stage}-failed.json', original)
shutil.copyfile(output/(stage+'.log'), output/(stage+'.original-failed.log'))
programs = {name:str(Path('/tmp/mvs-bin')/name) for name in ['DensifyPointCloud','ReconstructMesh','RefineMesh','TextureMesh']}
programs['blender'] = shutil.which('blender')
receipt = dict(original)
receipt.update(status='running',stages=[],recovery_source_checkout=original.get('recovery_checkout','6818ce1bb0dd7a2e7d16ec467b69ce05473ba684'),
               recovery_checkout=os.environ.get('GITHUB_SHA'), recovery_source_run=40966804036,
               recovered_sfm=True, sfm_rerun=False,
               recovery_executable_sha256={n:file_sha256(Path(p)) for n,p in programs.items()},
               source_scene_mvs_sha256=file_sha256(output/'scene.mvs'),
               source_camera_timeline_sha256=file_sha256(output/'camera-timeline.json'))
receipt.pop('error',None)
frames=json.loads((output/'camera-timeline.json').read_text())['frames']
args=argparse.Namespace(threads=2,resolution_level=1,
    blender_script=Path('src/echo4d/blender_export.py').resolve())
started=time.monotonic()
try:
    if stage == 'mesh':
        receipt.update(densify_rerun=False, mesh_rerun=False,
                       source_coarse_mesh_sha256=file_sha256(output/'scene_mesh.ply'))
        dense_and_export(output,receipt,programs,args,frames,start='refine',mesh_file=output/'scene_mesh.ply')
    else:
        dense_and_export(output,receipt,programs,args,frames)
except Exception as error:
    receipt.update(status='failed',error=f'{type(error).__name__}: {error}')
    raise
finally:
    receipt['recovery_seconds']=time.monotonic()-started
    write_json(output/'scene-report.json',receipt)
