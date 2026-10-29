"""Independent COLMAP RGB poses from 600 photos, only 150 measured depths for scale + Mesh.

The 600-frame RGB-only COLMAP camera model was previously solved in run
40917183649; depth from its other 450 frames is never read in the candidate.
The full 600-depth Mesh is reconstructed separately as an A/B reference.
No old Mesh, logged GT trajectory, predicted depth, or full-depth pose prior.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from stanford_colmap_metric300 import original_pairs, read_model, depth_scale

K = np.array([[525., 0, 319.5], [0, 525., 239.5], [0, 0, 1.]])


def board(o3d, old, new, photos, w2c, output):
    """Three identical recorded camera rays and two identical free camera rays."""
    def scene(mesh):
        s = o3d.t.geometry.RaycastingScene()
        s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        return s
    scenes = [scene(old), scene(new)]
    center = (old.get_min_bound() + old.get_max_bound()) / 2
    span = old.get_max_bound() - old.get_min_bound()
    small = K.copy()
    small[:2] /= 2
    small[0, 2] = (K[0, 2] + .5) / 2 - .5
    small[1, 2] = (K[1, 2] + .5) / 2 - .5
    keys = sorted(w2c)
    rows = []
    for k in (keys[0], keys[len(keys)//2], keys[-1]):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(small.astype(np.float32)),
            o3d.core.Tensor(w2c[k].astype(np.float32)), 320, 240)
        rows.append((f'recorded-{k:03d}', rays, photos[k]))
    for j, offset in enumerate(([-.5, -.9, -.7], [.8, -1.2, .05]), 1):
        eye = center + span * np.asarray(offset)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            78., o3d.core.Tensor(center.astype(np.float32)),
            o3d.core.Tensor(eye.astype(np.float32)),
            o3d.core.Tensor(np.array([0., -1., 0.], np.float32)), 320, 240)
        rows.append((f'free-{j}', rays, None))
    image = Image.new('RGB', (960, 5*268+34), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    draw.text((8, 10), 'real RGB (recorded)', fill='white')
    draw.text((330, 10), '600 RGB + 600 measured depths', fill='white')
    draw.text((650, 10), '600 RGB + 150 measured depths', fill='white')
    coverage = {}
    for i, (name, rays, photo) in enumerate(rows):
        y = 34 + i*268
        draw.text((8, y+6), name, fill='white')
        if photo:
            with Image.open(photo) as im:
                image.paste(im.convert('RGB').resize((320, 240)), (0, y+28))
        for j, (mesh, s) in enumerate(zip((old, new), scenes)):
            hit = s.cast_rays(rays)
            ids = hit['primitive_ids'].numpy().reshape(-1)
            uv = hit['primitive_uvs'].numpy().reshape(-1, 2)
            dist = hit['t_hit'].numpy().reshape(-1)
            faces = np.asarray(mesh.triangles)
            colors = np.asarray(mesh.vertex_colors)
            mask = np.isfinite(dist) & (dist > .15) & (dist < 16.) & (ids < len(faces))
            pixels = np.zeros((320*240, 3), np.uint8)
            if np.any(mask):
                weights = np.column_stack((1 - uv[mask].sum(axis=1), uv[mask]))
                rgb = (colors[faces[ids[mask].astype(int)]] *
                       weights[:, :, None]).sum(axis=1)
                pixels[mask] = np.clip(rgb*255, 0, 255).astype(np.uint8)
            coverage[f'{name}_{j}'] = round(float(mask.mean()), 4)
            image.paste(Image.fromarray(pixels.reshape(240, 320, 3)),
                        ((j+1)*320, y+28))
    image.save(output/'source-full600-depth150-3plus2.png')
    return coverage


def run(source, output):
    import open3d as o3d
    import trimesh
    if output.exists():
        raise FileExistsError('Fresh output directory required')
    output.mkdir(parents=True)
    receipt = {'status': 'RUNNING',
               'source_artifact': 'GitHub Actions 12917124559 / run 40917183649',
               'candidate_rgb_frames': 600, 'candidate_measured_depth_frames': 150,
               'candidate_pose_source': 'COLMAP SIFT RGB-only 600 images; independent RGB SfM',
               'candidate_scale_source': 'only the same 150 measured depth maps',
               'existing_mesh_prior': False, 'full_depth_trajectory_prior': False,
               'predicted_depth': False, 'visual_acceptance': 'NOT REVIEWED'}
    def save():
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    save()
    try:
        ledger = json.loads((source/'inputs.json').read_text())
        assert len(ledger) == 600
        pairs = original_pairs(o3d.data.LoungeRGBDImages())
        selected600 = np.linspace(0, len(pairs)-1, 600, dtype=int)
        depth150 = set(map(int, np.linspace(0, 599, 150, dtype=int)))
        assert len(depth150) == 150
        photos, depth_paths = {}, {}
        for item, index in zip(ledger, selected600):
            assert item['original_index'] == int(index)
            rgb, dep = pairs[index]
            k = item['sample']
            assert 0 <= k < 600
            photos[k] = rgb
            depth_paths[k] = dep
            if k in depth150:
                assert hashlib.sha256(rgb.read_bytes()).hexdigest() == item['rgb_sha256']
                assert hashlib.sha256(dep.read_bytes()).hexdigest() == item['depth_sha256']
        views = read_model(source/'model-txt-0')
        assert len(views) == 600, 'COLMAP 600 RGB poses not all registered'
        sparse_views = [v for v in views if int(Path(v[0]).stem) in depth150]
        assert len(sparse_views) == 150
        metric_scale, witnesses = depth_scale(
            sparse_views, {f'{k:06d}.png':depth_paths[k] for k in depth150})
        poses = {}
        for name, ext, _ in views:
            k = int(Path(name).stem)
            metric = ext.copy()
            metric[:3, 3] *= metric_scale
            poses[k] = metric
        receipt.update(metric_scale_m_per_colmap_unit=metric_scale,
                       scale_depth_witnesses=witnesses,
                       selected_depth_sample_indices=sorted(depth150))
        save()
        intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 525., 525., 319.5, 239.5)
        def volume():
            return o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=.025, sdf_trunc=.10,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        full, sparse = volume(), volume()
        for k in sorted(poses):
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.io.read_image(str(photos[k])),
                o3d.io.read_image(str(depth_paths[k])),
                depth_scale=1000., depth_trunc=3.,
                convert_rgb_to_intensity=False)
            full.integrate(rgbd, intr, poses[k])
            if k in depth150:
                sparse.integrate(rgbd, intr, poses[k])
        a, b = full.extract_triangle_mesh(), sparse.extract_triangle_mesh()
        if not a.has_triangles() or not b.has_triangles():
            raise RuntimeError('No real colored Mesh')
        def export(mesh, filename):
            trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                vertex_colors=np.clip(np.asarray(mesh.vertex_colors)*255, 0, 255).astype(np.uint8),
                process=False).export(output/filename)
        export(a, 'reference600depth.glb')
        export(b, 'candidate150depth.glb')
        coverage = board(o3d, a, b, photos, poses, output)
        receipt.update(status='REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED',
                       reference_faces=len(a.triangles), candidate_faces=len(b.triangles),
                       fixed_camera_coverage=coverage,
                       visual_acceptance='FAIL until actual 3+2 visual inspection',
                       image='source-full600-depth150-3plus2.png')
    except Exception as exc:
        receipt.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.output)
