"""Reduce *measured depth* 150 -> 75 -> 30, holding 600 RGB-only COLMAP poses fixed.

All three Stanford Lounge meshes are fused independently from original measured RGB-D.
Camera poses are estimated by COLMAP from 600 RGB photos (run 40917183649),
NOT from 600 depth frames or an existing mesh. All three meshes use the same
metric scale inferred from the same 30 measured depth maps. No model weights,
predicted depth, ground-truth trajectory or prior mesh are used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stanford_colmap_metric300 import original_pairs, read_model, depth_scale
from stanford_colmap600rgb_150depth import K


def nested_depth_indices():
    """Deterministic, evenly distributed, strictly nested 150/75/30 samples."""
    selected150 = np.linspace(0, 599, 150, dtype=int)
    selected75 = selected150[::2]
    selected30 = selected75[np.linspace(0, 74, 30, dtype=int)]
    sets = tuple(set(map(int, x)) for x in (selected150, selected75, selected30))
    assert tuple(map(len, sets)) == (150, 75, 30)
    assert sets[2] < sets[1] < sets[0]
    return sets


def render_board(o3d, meshes, photos, poses, output):
    """Same 3 recorded + 2 free rays for all three real meshes."""
    scenes = []
    for mesh in meshes:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        scenes.append(scene)
    center = (meshes[0].get_min_bound() + meshes[0].get_max_bound()) / 2
    span = meshes[0].get_max_bound() - meshes[0].get_min_bound()
    half = K.copy()
    half[:2] /= 2
    half[0, 2] = (K[0, 2] + .5) / 2 - .5
    half[1, 2] = (K[1, 2] + .5) / 2 - .5
    keys = sorted(poses)
    rows = []
    for k in (keys[0], keys[len(keys)//2], keys[-1]):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(half.astype(np.float32)),
            o3d.core.Tensor(poses[k].astype(np.float32)), 320, 240)
        rows.append((f'recorded-{k:03d}', rays, photos[k]))
    for j, offset in enumerate(([-.5, -.9, -.7], [.8, -1.2, .05]), 1):
        eye = center + span * np.asarray(offset)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            78., o3d.core.Tensor(center.astype(np.float32)),
            o3d.core.Tensor(eye.astype(np.float32)),
            o3d.core.Tensor(np.array([0., -1., 0.], np.float32)), 320, 240)
        rows.append((f'free-{j}', rays, None))
    image = Image.new('RGB', (1280, 5*268+34), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    for j, label in enumerate(('real RGB / free view', '600 RGB + 150 depths',
                                '600 RGB + 75 depths', '600 RGB + 30 depths')):
        draw.text((8 + j*320, 10), label, fill='white')
    coverage = {}
    for i, (name, rays, photo) in enumerate(rows):
        y = 34 + i*268
        draw.text((8, y+6), name, fill='white')
        if photo is not None:
            with Image.open(photo) as im:
                image.paste(im.convert('RGB').resize((320, 240)), (0, y+28))
        for j, (mesh, scene) in enumerate(zip(meshes, scenes)):
            hits = scene.cast_rays(rays)
            ids = hits['primitive_ids'].numpy().reshape(-1)
            uv = hits['primitive_uvs'].numpy().reshape(-1, 2)
            distances = hits['t_hit'].numpy().reshape(-1)
            faces = np.asarray(mesh.triangles)
            colors = np.asarray(mesh.vertex_colors)
            valid = (np.isfinite(distances) & (distances > .15) &
                     (distances < 16.) & (ids < len(faces)))
            pixels = np.zeros((320*240, 3), np.uint8)
            if np.any(valid):
                weights = np.column_stack((1 - uv[valid].sum(axis=1), uv[valid]))
                rgb = (colors[faces[ids[valid].astype(int)]] *
                       weights[:, :, None]).sum(axis=1)
                pixels[valid] = np.clip(rgb*255, 0, 255).astype(np.uint8)
            coverage[f'{name}_depth{(150, 75, 30)[j]}'] = round(float(valid.mean()), 4)
            image.paste(Image.fromarray(pixels.reshape(240, 320, 3)),
                        ((j+1)*320, y+28))
    image.save(output/'source-depth150-75-30-3plus2.png')
    return coverage


def run(source, output):
    import open3d as o3d
    import trimesh

    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True)
    groups = nested_depth_indices()
    receipt = dict(status='RUNNING',
        scene='real Stanford Lounge original RGB-D',
        pose_source='COLMAP SfM from 600 original RGB photos, run 40917183649',
        source_artifact='12917124559', selected_rgb_frames=600,
        measured_depth_counts=[150, 75, 30],
        metric_scale_source='same nested 30 measured depth maps for all three',
        old_mesh_prior=False, ground_truth_pose_prior=False,
        predicted_depth=False, visual_acceptance='NOT REVIEWED')
    def save():
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    save()
    try:
        ledger = json.loads((source/'inputs.json').read_text())
        if len(ledger) != 600:
            raise ValueError('Expected 600-image COLMAP manifest')
        source_hashes = {}
        for filename in ('images.txt', 'points3D.txt'):
            source_hashes[filename] = hashlib.sha256(
                (source/'model-txt-0'/filename).read_bytes()).hexdigest()
        receipt['source_colmap_sha256'] = source_hashes
        original = original_pairs(o3d.data.LoungeRGBDImages())
        chosen600 = np.linspace(0, len(original)-1, 600, dtype=int)
        photos, depths = {}, {}
        for k, (item, idx) in enumerate(zip(ledger, chosen600)):
            if item['sample'] != k or item['original_index'] != int(idx):
                raise ValueError('600 RGB frame manifest does not match original sequence')
            rgb, depth = original[idx]
            if hashlib.sha256(rgb.read_bytes()).hexdigest() != item['rgb_sha256']:
                raise ValueError(f'RGB source changed at sample {k}')
            photos[k] = rgb
            if k in groups[0]:
                if hashlib.sha256(depth.read_bytes()).hexdigest() != item['depth_sha256']:
                    raise ValueError(f'Measured depth source changed at sample {k}')
                depths[k] = depth
        views = read_model(source/'model-txt-0')
        names = [int(Path(name).stem) for name, _, _ in views]
        if len(views) != 600 or set(names) != set(range(600)):
            raise ValueError('Expected 600 independently registered COLMAP RGB poses')
        anchor_views = [v for v in views if int(Path(v[0]).stem) in groups[2]]
        scale, witnesses = depth_scale(
            anchor_views, {f'{k:06d}.png':depths[k] for k in groups[2]})
        poses = {}
        for name, ext, _ in views:
            k = int(Path(name).stem)
            metric = ext.copy()
            metric[:3, 3] *= scale
            poses[k] = metric
        receipt.update(metric_scale_m_per_colmap_unit=scale,
                       metric_scale_witnesses=witnesses,
                       selected_depth_samples={str(n): sorted(g)
                           for n, g in zip((150, 75, 30), groups)})
        save()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 525., 525., 319.5, 239.5)
        def volume():
            return o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=.025, sdf_trunc=.10,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        volumes = [volume(), volume(), volume()]
        for k in sorted(groups[0]):
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.io.read_image(str(photos[k])),
                o3d.io.read_image(str(depths[k])),
                depth_scale=1000., depth_trunc=3.,
                convert_rgb_to_intensity=False)
            for group, vol in zip(groups, volumes):
                if k in group:
                    vol.integrate(rgbd, intrinsic, poses[k])
        meshes = [v.extract_triangle_mesh() for v in volumes]
        if any(not mesh.has_triangles() or not mesh.has_vertex_colors()
               for mesh in meshes):
            raise RuntimeError('One or more measured-depth meshes are empty/uncolored')
        for n, mesh in zip((150, 75, 30), meshes):
            trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                vertex_colors=np.clip(
                    np.asarray(mesh.vertex_colors)*255, 0, 255).astype(np.uint8),
                process=False).export(output/f'measured{n}.glb')
        coverage = render_board(o3d, meshes, photos, poses, output)
        receipt.update(status='REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED',
                       triangle_counts={str(n): len(mesh.triangles)
                           for n, mesh in zip((150, 75, 30), meshes)},
                       fixed_camera_coverage=coverage,
                       image='source-depth150-75-30-3plus2.png',
                       visual_acceptance='FAIL pending direct five-view inspection')
    except Exception as exc:
        receipt.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.source, args.output)
