"""Real Stanford Lounge: estimate poses and Mesh from only 600 RGB-D frames.

Uses the complete upstream Open3D reconstruction_system; no full-3000-frame
poses, no existing Mesh, no pretrained depth model. This is a true reduction
of both pose-estimation and fusion input, unlike frozen-pose fusion ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

UPSTREAM_SHA = '1e7b17438687a0b0c1e5a7187321ac7044afe275'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_pairs(original):
    color_dir = Path(original.color_paths[0]).parent
    depth_dir = Path(original.depth_paths[0]).parent
    colors = sorted(p for p in color_dir.iterdir() if p.stem.isdecimal() and p.suffix.lower() in ('.jpg','.png'))
    depths = sorted(p for p in depth_dir.iterdir() if p.stem.isdecimal() and p.suffix.lower() == '.png')
    if len(colors) != 3000 or len(depths) != 3000 or [p.stem for p in colors] != [p.stem for p in depths]:
        raise RuntimeError('Expected 3000 matched native RGB and depth frames')
    return list(zip(colors, depths))


def compose(output):
    names = ['recorded-00000.png', 'recorded-00300.png', 'recorded-00599.png', 'free-1.png', 'free-2.png']
    canvas = Image.new('RGB', (640, 5*268 + 34), (20,20,20))
    draw = ImageDraw.Draw(canvas)
    draw.text((5,10), 'original real RGB / free view', fill='white')
    draw.text((330,10), 'independent 600 RGB-D estimated poses + Mesh', fill='white')
    for i,name in enumerate(names):
        with Image.open(output/'views'/name) as im:
            im = im.convert('RGB')
            canvas.paste(im.crop((0,28,320,268)) if i < 3 else Image.new('RGB',(320,240)), (0,34+i*268+28))
            canvas.paste(im.crop((im.width-320,28,im.width,268)), (320,34+i*268+28))
        draw.text((5,34+i*268+5),name[:-4],fill='white')
    path = output/'source-independent600-3plus2.png'
    canvas.save(path)
    return path


def run(upstream: Path, output: Path, sample_count: int=600):
    import open3d as o3d
    import trimesh
    from run_official_room import render_views
    upstream, output = upstream.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('Use fresh output directory')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip() != UPSTREAM_SHA:
        raise RuntimeError('Unexpected upstream Open3D source')
    output.mkdir(parents=True)
    report = {'status':'RUNNING','source':'Stanford Lounge actual RGB-D',
              'original_frames':3000,'selected_rgb_frames':sample_count,
              'selected_measured_depth_frames':sample_count,
              'pose_source':'estimated independently from selected RGB-D only',
              'full_3000_pose_prior':False,'existing_mesh_prior':False,
              'predicted_depth':False,'upstream_commit':UPSTREAM_SHA,
              'visual_acceptance':'NOT REVIEWED'}
    def save():
        (output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    original = o3d.data.LoungeRGBDImages()
    pairs = original_pairs(original)
    selected = np.linspace(0, len(pairs)-1, sample_count, dtype=int).tolist()
    if len(set(selected)) != sample_count:
        raise ValueError('Duplicate sampled frame')
    root = output/'sampled_dataset'
    (root/'color').mkdir(parents=True)
    (root/'depth').mkdir()
    ledger=[]
    for i,src_id in enumerate(selected):
        color,depth = pairs[src_id]
        dest_color = root/'color'/f'{i:06d}{color.suffix}'
        dest_depth = root/'depth'/f'{i:06d}.png'
        for source,dest in [(color,dest_color),(depth,dest_depth)]:
            try: os.link(source,dest)
            except OSError: shutil.copyfile(source,dest)
        ledger.append({'sample':i,'source_ordinal':src_id,'source_color':color.name,
                       'source_depth':depth.name,'rgb_sha256':file_sha(color),
                       'depth_sha256':file_sha(depth)})
    (output/'inputs.json').write_text(json.dumps(ledger,indent=2)+'\n')
    system=upstream/'examples/python/reconstruction_system'
    sys.path.insert(0,str(system))
    from data_loader import lounge_data_loader
    from initialize_config import initialize_config
    config=lounge_data_loader()
    config['path_dataset']=str(root)
    initialize_config(config)
    config_path=output/'config.json'
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    try:
        subprocess.run([sys.executable,str(system/'run_system.py'),'--config',str(config_path),
                        '--make','--register','--refine','--integrate'],check=True)
        scene=root/'scene'
        mesh=o3d.io.read_triangle_mesh(str(scene/'integrated.ply'))
        if not mesh.has_triangles() or not mesh.has_vertex_colors():
            raise RuntimeError('Upstream produced no colored Mesh')
        lines=(scene/'trajectory.log').read_text().splitlines()
        poses=[np.array([float(v) for line in lines[k+1:k+5] for v in line.split()]).reshape(4,4)
               for k in range(0,len(lines),5)]
        if len(poses)!=sample_count:
            raise RuntimeError('Not all sampled frames received estimated poses')
        xyz=np.asarray(mesh.vertices); faces=np.asarray(mesh.triangles)
        colors=np.clip(np.asarray(mesh.vertex_colors)*255,0,255).astype(np.uint8)
        trimesh.Trimesh(vertices=xyz,faces=faces,vertex_colors=colors,process=False).export(output/'room-independent600.glb')
        preview=mesh.simplify_quadric_decimation(300000) if len(faces)>300000 else mesh
        trimesh.Trimesh(vertices=np.asarray(preview.vertices),faces=np.asarray(preview.triangles),
            vertex_colors=np.clip(np.asarray(preview.vertex_colors)*255,0,255).astype(np.uint8),
            process=False).export(output/'room-independent600-preview.glb')
        shutil.copytree(scene,output/'scene')
        render_views(o3d,mesh,[root/'color'/f'{i:06d}{pairs[selected[i]][0].suffix}' for i in range(sample_count)],poses,output)
        image=compose(output)
        report.update(status='REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED',
                      glb_sha256=file_sha(output/'room-independent600.glb'),
                      image=image.name,visual_acceptance='FAIL until visual review')
    except Exception as exc:
        report.update(status='FAIL',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        # Keep geometry, pose graphs and source hashes; do not upload redundant public RGB-D images.
        shutil.rmtree(root / "color", ignore_errors=True)
        shutil.rmtree(root / "depth", ignore_errors=True)
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=600)
    a=p.parse_args()
    run(a.upstream,a.output,a.samples)
