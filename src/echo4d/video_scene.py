"""Ordinary video -> existing SfM -> CPU OpenMVS -> textured Blender scene.

External programs do the reconstruction; no stock mesh, sensor depth or GT pose
is substituted when they fail. A successful execution is not quality acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from .model import DataError, file_sha256
from .video_sfm import run_pipeline as run_sfm
from .video_sfm import _estimated_track
from .colmap import _cameras


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_stage(name, command, output, receipt, timeout=7200):
    """Keep complete logs and persist failure before propagating it."""
    entry = {'stage': name, 'command': list(map(str, command)), 'status': 'running'}
    receipt['stages'].append(entry)
    write_json(output/'scene-report.json', receipt)
    started = time.monotonic()
    log = output/(name+'.log')
    try:
        print(f'[{name}]', file=sys.stderr, flush=True)
        with log.open('w') as stream:
            subprocess.run(entry['command'], check=True, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=timeout, cwd=output)
        entry['status'] = 'completed'
    except Exception as error:
        entry.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        entry['seconds'] = time.monotonic()-started
        if log.exists():
            entry['log_sha256'] = file_sha256(log)
        write_json(output/'scene-report.json', receipt)


def required_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise DataError(f'Backend did not produce {path.name}')
    return path


def package_scene(output, frames):
    from PIL import Image, ImageDraw
    review = output/'review'
    review.mkdir()
    for index in sorted({0,len(frames)//2,len(frames)-1}):
        with Image.open(output/'dense/images'/frames[index]['name']) as photo, Image.open(output/'views'/f'recorded-{index:05d}.png') as rendered:
            photo, rendered = photo.convert('RGB'), rendered.convert('RGB')
            if photo.size != rendered.size:
                raise DataError('Recorded camera render dimensions differ from undistorted RGB')
            canvas = Image.new('RGB',(photo.width*2,photo.height+28),(24,24,24))
            ImageDraw.Draw(canvas).text((8,8),'UNDISTORTED SOURCE RGB / ESTIMATED MESH',fill='white')
            canvas.paste(photo,(0,28))
            canvas.paste(rendered,(photo.width,28))
            canvas.save(review/f'recorded-{index:05d}.png')
    for p in (output/'views').glob('free-*.png'):
        shutil.copyfile(p,review/p.name)
    (output/'README.txt').write_text('Open scene.blend in Blender. Camera view follows the recorded source timeline; leave camera view to orbit the same mesh.\nScale is UNKNOWN. Geometry is estimated from RGB, with observed image texture.\nMissing surfaces and remaining reconstruction errors are not verified historical geometry.\nreview/ contains 3 source/render comparisons and 2 actual free-camera renders.\n')
    paths=[output/'scene_textured.glb',output/'scene.blend',output/'scene.json',output/'camera-timeline.json',output/'README.txt',*sorted(review.glob('*.png'))]
    with zipfile.ZipFile(output/'scene-delivery.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p,p.relative_to(output))
    return {str(p.relative_to(output)):file_sha256(p) for p in paths}


def verify_textured_glb(path):
    """Require actual triangles and embedded image/UV data, not colored points."""
    import struct
    data = required_file(path).read_bytes()
    if len(data) < 20:
        raise DataError('Truncated GLB')
    magic, version, length = struct.unpack_from('<III', data)
    if magic != 0x46546c67 or version != 2 or length != len(data):
        raise DataError('Invalid GLB header')
    size, kind = struct.unpack_from('<II', data, 12)
    if kind != 0x4e4f534a or 20+size > len(data):
        raise DataError('GLB has no valid JSON chunk')
    doc = json.loads(data[20:20+size])
    offset = 20+size
    if offset+8 > len(data):
        raise DataError('GLB has no embedded binary data')
    binary_size, binary_kind = struct.unpack_from('<II', data, offset)
    binary = data[offset+8:]
    if binary_kind != 0x004e4942 or binary_size != len(binary):
        raise DataError('Invalid GLB binary chunk')
    for view in doc.get('bufferViews', []):
        start, count = view.get('byteOffset', 0), view.get('byteLength', 0)
        if view.get('buffer', 0) != 0 or start < 0 or count <= 0 or start+count > len(binary):
            raise DataError('GLB buffer view is outside embedded data')
    primitives = [p for m in doc.get('meshes', []) for p in m.get('primitives', [])]
    accessors = doc.get('accessors', [])
    triangles = 0
    for p in primitives:
        if p.get('mode', 4) != 4 or 'POSITION' not in p.get('attributes', {}):
            raise DataError('GLB primitive is not a triangle mesh')
        count = accessors[p['indices']]['count'] if 'indices' in p else accessors[p['attributes']['POSITION']]['count']
        if count <= 0 or count % 3:
            raise DataError('GLB has invalid triangle count')
        triangles += count//3
        try:
            material = doc['materials'][p['material']]
            tex = material['pbrMetallicRoughness']['baseColorTexture']
            image = doc['images'][doc['textures'][tex['index']]['source']]
            uv = f"TEXCOORD_{tex.get('texCoord', 0)}"
            if uv not in p['attributes'] or 'bufferView' not in image:
                raise KeyError('missing embedded image or UV')
            view = doc['bufferViews'][image['bufferView']]
            start = view.get('byteOffset', 0)
            pixels = binary[start:start+view['byteLength']]
            from PIL import Image
            import io
            with Image.open(io.BytesIO(pixels)) as picture:
                picture.verify()
        except (KeyError, IndexError, TypeError) as error:
            raise DataError('Mesh lacks embedded source-image texture and UVs') from error
    if not triangles:
        raise DataError('GLB has no triangles')
    return {'triangles': triangles, 'primitives': len(primitives),
            'embedded_images': len(doc.get('images', [])), 'sha256': file_sha256(path)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=120)
    parser.add_argument('--camera-model', choices=['PINHOLE', 'SIMPLE_PINHOLE', 'SIMPLE_RADIAL'], default='SIMPLE_RADIAL')
    parser.add_argument('--camera-params')
    parser.add_argument('--openmvs-bin', type=Path, help='Directory containing OpenMVS executables; otherwise PATH')
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--blender-script', type=Path, default=Path(__file__).with_name('blender_export.py'))
    parser.add_argument('--resolution-level', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--min-registration', type=float, default=.8)
    parser.add_argument('--threads', type=int, default=2)
    return parser


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise DataError('Use a fresh output directory; previous runs are never overwritten')
    if not 3 <= args.max_frames <= 240 or not 1 <= args.threads <= 64:
        raise DataError('max-frames must be 3..240 and threads 1..64')
    if not math.isfinite(args.min_registration) or not 0 < args.min_registration <= 1:
        raise DataError('min-registration must be in (0,1]')
    if not args.video.is_file():
        raise DataError('Input video does not exist')
    if not args.blender_script.is_file():
        raise DataError('Missing Blender export script; use --blender-script')
    if args.camera_params:
        try:
            values = [float(x) for x in args.camera_params.split(',')]
        except ValueError as error:
            raise DataError('Invalid camera parameters') from error
        expected = {'PINHOLE':4, 'SIMPLE_PINHOLE':3, 'SIMPLE_RADIAL':4}[args.camera_model]
        if len(values) != expected or not all(math.isfinite(x) for x in values) or values[0] <= 0 or (args.camera_model == 'PINHOLE' and values[1] <= 0):
            raise DataError('Invalid camera calibration')
    programs = {}
    for name in ['ffmpeg', 'ffprobe', 'colmap', 'InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh', 'TextureMesh', 'blender']:
        value = args.blender if name == 'blender' else str(args.openmvs_bin/name) if args.openmvs_bin and name not in ['ffmpeg', 'ffprobe', 'colmap'] else name
        found = shutil.which(value)
        if not found:
            raise DataError(f'Missing dependency: {name} ({value})')
        programs[name] = str(Path(found).resolve())
    output.mkdir(parents=True)
    receipt = {'schema':'echo-video-scene-1', 'status':'running', 'stages':[],
               'source_video_sha256':file_sha256(args.video), 'scale':'unknown',
               'geometry':'estimated multi-view RGB geometry', 'appearance':'observed RGB projected onto estimated geometry',
               'generated_model_used':False, 'surface_support':'not independently classified; meshing is an estimate', 'visual_accepted':False,
               'config':{'max_frames':args.max_frames, 'resolution_level':args.resolution_level,
                         'min_registration':args.min_registration, 'threads':args.threads,
                         'camera_model':args.camera_model, 'camera_params':args.camera_params},
               'executable_sha256':{n:file_sha256(Path(p)) for n,p in programs.items()}}
    write_json(output/'scene-report.json', receipt)
    started = time.monotonic()
    try:
        sfm = run_sfm(argparse.Namespace(video=args.video.resolve(), images=None,
            timestamps=None, output=output/'sfm', fps=4., timestamp_mode='source-pts',
            max_frames=args.max_frames, camera_model=args.camera_model, camera_params=args.camera_params))
        if not sfm['status'].startswith('passed'):
            raise DataError('SfM failed; see sfm/report.json')
        result = sfm['result']
        receipt['registration'] = result
        if result['registration_fraction'] < args.min_registration:
            raise DataError('Insufficient registered video coverage; refusing to present one fragment as the whole recording')
        dense = output/'dense'
        dense.mkdir()
        run_stage('undistort', [programs['colmap'], 'image_undistorter', '--image_path', output/'sfm/images', '--input_path', output/'sfm'/result['sfm_model_dir'], '--output_path', dense, '--output_type', 'COLMAP'], output, receipt)
        camera_model = output/'undistorted-model'
        camera_model.mkdir()
        run_stage('camera-model', [programs['colmap'], 'model_converter', '--input_path', dense/'sparse', '--output_path', camera_model, '--output_type', 'TXT'], output, receipt)
        original = json.loads((output/'sfm/trajectory-scale-unknown.json').read_text())
        times = {f['name']:f['t_s'] for f in original['frames']}
        frames, _ = _estimated_track(camera_model, times)
        cameras = _cameras(camera_model/'cameras.txt')
        original['frames'] = frames
        original['cameras'] = {str(key):[c.width,c.height,c.fx,c.fy,c.cx,c.cy] for key,c in cameras.items()}
        original['images'] = 'dense/images: undistorted source RGB'
        write_json(output/'camera-timeline.json', original)
        mvs = output/'scene.mvs'
        run_stage('interface', [programs['InterfaceCOLMAP'], '-i', dense, '-o', mvs, '--image-folder', dense/'images'], output, receipt)
        required_file(mvs)
        stages = [('densify','DensifyPointCloud',mvs,output/'scene_dense.mvs'),
                  ('mesh','ReconstructMesh',output/'scene_dense.mvs',output/'scene_mesh.mvs'),
                  ('refine','RefineMesh',output/'scene_mesh.mvs',output/'scene_refined.mvs'),
                  ('texture','TextureMesh',output/'scene_refined.mvs',output/'scene_textured.mvs')]
        for label, program, source, target in stages:
            command = [programs[program], '-i', source, '-o', target, '--max-threads', args.threads]
            if label in ['densify','refine','texture']:
                command += ['--resolution-level', args.resolution_level]
            if label == 'texture':
                command += ['--export-type','glb']
            if label in ['mesh', 'refine']:
                command += ['--close-holes','0']
            run_stage(label, command, output, receipt)
            required_file(target)
        glb = output/'scene_textured.glb'
        receipt['mesh'] = verify_textured_glb(glb)
        run_stage('blender', [programs['blender'], '-b', '--python-exit-code', '1', '--python', args.blender_script.resolve(), '--', '--input', glb, '--output', output/'scene.blend', '--units', 'unknown', '--trajectory', output/'camera-timeline.json', '--render-views', '--provenance', 'RGB-only COLMAP/OpenMVS; scale unknown; estimated geometry'], output, receipt)
        required_file(output/'scene.blend')
        receipt['outputs'] = package_scene(output, frames)
        receipt['delivery_zip_sha256'] = file_sha256(output/'scene-delivery.zip')
        receipt['status'] = 'completed_not_quality_accepted'
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt['seconds'] = time.monotonic()-started
        write_json(output/'scene-report.json', receipt)
    return receipt


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
        print(json.dumps({'status':result['status'], 'output':str(args.output.resolve())}))
        return 0
    except (DataError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'status':'failed', 'error':str(error)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
