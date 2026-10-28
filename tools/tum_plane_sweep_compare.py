"""Actual TUM old/new Mesh at identical frozen 3 heldout + 2 free poses.

Source photos remain 640x480 files with SHA; montage scales them to 320x240.
Rendered grayscale is actual triangle z-buffer, not a synthesized photograph.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from echo4d.model import Camera, file_sha256
from echo4d.holdout import read_echo_ply, rasterize_mesh_depth
from tum_rgb_stereo import rgb_pose_frames
from tum_public_roundtrip import ROOT


def run(old: Path, new: Path, source_root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    old_report = json.loads((old / 'report.json').read_text())
    frames = rgb_pose_frames(source_root / ROOT)
    k = Camera(width=320, height=240, fx=262.5, fy=262.5, cx=159.75, cy=119.75)
    models = {}
    for name, directory in [('old', old), ('new', new)]:
        path = directory / 'reconstruction' / 'mesh.ply'
        models[name] = read_echo_ply(path)
    rows, details = [], []
    for view in old_report['views']:
        name = view['name']
        pose = np.array(view['pose'], dtype=np.float64).reshape(4, 4, order='F')
        view_images = []
        source_meta = None
        if name.startswith('heldout-'):
            idx = int(name.split('-')[1])
            source_path = frames[idx]['rgb']
            source_img = Image.open(source_path).convert('RGB')
            if source_img.size != (640, 480):
                raise ValueError('Original TUM RGB is not 640x480')
            original_path = output / f'source-original-640x480-{name}.png'
            source_img.save(original_path)
            source_meta = {'source_index': idx,
                           'source_original_sha256': file_sha256(source_path),
                           'copied_png_sha256': file_sha256(original_path),
                           'resolution': [640, 480],
                           'original_filename': str(source_path.relative_to(source_root))}
            view_images.append(source_img.resize((320, 240), Image.Resampling.BILINEAR))
        else:
            blank = Image.new('RGB', (320, 240), (15, 15, 15))
            ImageDraw.Draw(blank).text((20, 100), 'FREE VIEW (no original photo)', fill='white')
            view_images.append(blank)
        coverage = {}
        for model_name in ('old', 'new'):
            vertices, faces = models[model_name]
            depth, stats = rasterize_mesh_depth(vertices, faces, k, pose, far=4.0)
            valid = (depth >= .15) & (depth <= 4)
            gray = np.zeros(depth.shape, dtype=np.uint8)
            gray[valid] = np.clip(np.rint(255 - (depth[valid] - .15) * 254 / 3.85),
                                  1, 255).astype(np.uint8)
            image = Image.fromarray(gray).convert('RGB')
            image.save(output / f'{model_name}-{name}.png')
            view_images.append(image)
            coverage[model_name] = int(np.count_nonzero(valid))
        details.append({'view': name, 'pose': view['pose'], 'source': source_meta,
                        'old_covered_pixels': coverage['old'],
                        'new_covered_pixels': coverage['new'],
                        'delta_covered_pixels': coverage['new'] - coverage['old']})
        rows.append(view_images)
    # Each row is one pose; source / old / new side by side.
    canvas = Image.new('RGB', (960, 5 * 266 + 40), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 6), 'Original RGB (640 source) | Old RGB MVS Mesh | New cross-view Mesh', fill='black')
    for i, row in enumerate(rows):
        y = 32 + 266 * i
        for j, img in enumerate(row):
            canvas.paste(img, (320 * j, y))
        draw.text((5, y + 242), details[i]['view'] + ' | grayscale=actual Mesh depth; black=unknown', fill='black')
    montage = output / 'source-old-new-five-frozen-poses.png'
    canvas.save(montage)
    # Export existing reconstructed colored triangles as rotatable GLB, no editing.
    import trimesh
    glbs = {}
    for name, directory in [('old', old), ('new', new)]:
        mesh = trimesh.load_mesh(directory / 'reconstruction' / 'mesh.ply',
                                 force='mesh', process=False)
        path = output / f'{name}-colored-mesh.glb'
        mesh.export(path)
        glbs[name] = {'sha256': file_sha256(path), 'bytes': path.stat().st_size}
    report = {'input': 'TUM fr1_xyz observed RGB + measured poses, no measured depth pixels or prior Mesh',
              'source_attribution': 'TUM RGB-D Dataset, J. Sturm et al., IROS 2012, CC BY 4.0',
              'old_mesh_sha256': file_sha256(old / 'reconstruction' / 'mesh.ply'),
              'new_mesh_sha256': file_sha256(new / 'reconstruction' / 'mesh.ply'),
              'montage_sha256': file_sha256(montage),
              'glb': glbs, 'views': details,
              'assessment': 'Mesh generated, visual recognition not independently accepted'}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.old, args.new, args.source_root, args.output)
    print(json.dumps({'montage_sha256': result['montage_sha256'],
                      'coverage': [(x['view'], x['old_covered_pixels'],
                                    x['new_covered_pixels']) for x in result['views']]}))
