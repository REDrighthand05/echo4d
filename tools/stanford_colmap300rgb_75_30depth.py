"""Stanford Lounge: 300 RGB-only COLMAP poses, registered / 75 / 30 real depth fusion.

Use the frozen 300-RGB COLMAP artifact from run 40906999521, not 600-RGB poses.
Only independently registered RGB poses can be fused. All three Meshes share
one metric calibration from the SAME nested 30 measured depth frames. No
old Mesh, full-3000 poses, predicted depth, or GPU. Full-depth reference
is for same-camera visual comparison, never a prior for sparse fusion.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stanford_colmap_metric300 import original_pairs, read_model, depth_scale
from stanford_colmap600rgb_150depth import K


def nested_depth_indices(registered_keys):
    """Evenly spaced, strictly nested subsets of independently registered frames."""
    all_keys = np.asarray(sorted(set(registered_keys)), dtype=int)
    if len(all_keys) < 75:
        raise ValueError('At least 75 registered camera poses are required')
    selected75 = all_keys[np.linspace(0, len(all_keys)-1, 75, dtype=int)]
    selected30 = selected75[np.linspace(0, 74, 30, dtype=int)]
    sets = tuple(set(map(int, x)) for x in (all_keys, selected75, selected30))
    assert tuple(map(len, sets)) == (len(all_keys), 75, 30)
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
    for j, label in enumerate(('real RGB / free view', f'300 RGB + {len(poses)} depths',
                                '300 RGB + 75 depths', '300 RGB + 30 depths')):
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
            coverage[f'{name}_depth{(len(poses), 75, 30)[j]}'] = round(float(valid.mean()), 4)
            image.paste(Image.fromarray(pixels.reshape(240, 320, 3)),
                        ((j+1)*320, y+28))
    image.save(output/'source-300rgb-depth-registered-75-30-3plus2.png')
    return coverage


def run(source, output):
    import open3d as o3d
    import trimesh

    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True)
    receipt = dict(status='RUNNING',
        scene='real Stanford Lounge original RGB-D',
        pose_source='COLMAP SfM from 300 original RGB photos, run 40906999521',
        source_artifact='12911194244', selected_rgb_frames=300,
        measured_depth_counts='registered/75/30 pending source-model read',
        metric_scale_source='same nested 30 measured depth maps for all three',
        old_mesh_prior=False, ground_truth_pose_prior=False,
        predicted_depth=False, visual_acceptance='NOT REVIEWED')
    def save():
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    save()
    try:
        ledger = json.loads((source/'inputs.json').read_text())
        if len(ledger) != 300:
            raise ValueError('Expected 300-image COLMAP manifest')
        views = read_model(source/'model-txt-0')
        names = [int(Path(name).stem) for name, _, _ in views]
        if len(names) != len(set(names)) or any(k < 0 or k >= 300 for k in names):
            raise ValueError('Invalid independently registered 300-RGB camera model')
        groups = nested_depth_indices(names)
        receipt['registered_rgb_frames'] = len(names)
        receipt['measured_depth_counts'] = [len(groups[0]), 75, 30]
        source_hashes = {}
        for filename in ('images.txt', 'points3D.txt'):
            source_hashes[filename] = hashlib.sha256(
                (source/'model-txt-0'/filename).read_bytes()).hexdigest()
        receipt['source_colmap_sha256'] = source_hashes
        original = original_pairs(o3d.data.LoungeRGBDImages())
        chosen300 = np.linspace(0, len(original)-1, 300, dtype=int)
        photos, depths = {}, {}
        for k, (item, idx) in enumerate(zip(ledger, chosen300)):
            if item['sample'] != k or item['original_index'] != int(idx):
                raise ValueError('300 RGB frame manifest does not match original sequence')
            rgb, depth = original[idx]
            if hashlib.sha256(rgb.read_bytes()).hexdigest() != item['rgb_sha256']:
                raise ValueError(f'RGB source changed at sample {k}')
            photos[k] = rgb
            if k in groups[0]:
                if hashlib.sha256(depth.read_bytes()).hexdigest() != item['depth_sha256']:
                    raise ValueError(f'Measured depth source changed at sample {k}')
                depths[k] = depth
        anchor_views = [v for v in views if int(Path(v[0]).stem) in groups[2]]
        scale, witnesses = depth_scale(
            anchor_views, {name: depths[int(Path(name).stem)]
                           for name, _, _ in anchor_views})
        poses = {}
        for name, ext, _ in views:
            k = int(Path(name).stem)
            metric = ext.copy()
            metric[:3, 3] *= scale
            poses[k] = metric
        receipt.update(metric_scale_m_per_colmap_unit=scale,
                       metric_scale_witnesses=witnesses,
                       selected_depth_samples={str(n): sorted(g)
                           for n, g in zip((len(groups[0]), 75, 30), groups)})
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
        for n, mesh in zip((len(groups[0]), 75, 30), meshes):
            trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                vertex_colors=np.clip(
                    np.asarray(mesh.vertex_colors)*255, 0, 255).astype(np.uint8),
                process=False).export(output/f'measured{n}.glb')
        coverage = render_board(o3d, meshes, photos, poses, output)
        receipt.update(status='REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED',
                       triangle_counts={str(n): len(mesh.triangles)
                           for n, mesh in zip((len(groups[0]), 75, 30), meshes)},
                       fixed_camera_coverage=coverage,
                       image='source-300rgb-depth-registered-75-30-3plus2.png',
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
