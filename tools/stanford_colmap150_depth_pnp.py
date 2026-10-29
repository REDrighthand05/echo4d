"""Recover unregistered Stanford frames by measured-depth-verified local PnP.

Only the original selected 150 RGB-D frames, independent COLMAP camera poses,
and their own measured depths are used. No full trajectory or existing Mesh.
Reject uncorroborated geometry rather than forcing missing poses into the scene.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from stanford_colmap_metric300 import original_pairs, read_model

K = np.array([[525., 0, 319.5], [0, 525., 239.5], [0, 0, 1.]], np.float64)


def candidate(anchor, target, features, depths, w2c_anchor):
    ka, da = features[anchor]
    kt, dt = features[target]
    if da is None or dt is None or len(da) < 40 or len(dt) < 40:
        return None
    good = [a for a, b in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, dt, k=2)
            if a.distance < .72 * b.distance]
    if len(good) < 40:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kt[m.trainIdx].pt for m in good])
    x, y = np.rint(src[:, 0]).astype(int), np.rint(src[:, 1]).astype(int)
    z = depths[anchor][np.clip(y, 0, 479), np.clip(x, 0, 639)].astype(float) / 1000.
    valid = (x >= 0) & (x < 640) & (y >= 0) & (y < 480) & (z > .3) & (z < 3.)
    if np.count_nonzero(valid) < 35:
        return None
    src, dst, z = src[valid], dst[valid], z[valid]
    xyz = np.column_stack(((src[:, 0] - K[0, 2]) * z / K[0, 0],
                           (src[:, 1] - K[1, 2]) * z / K[1, 1], z))
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        xyz, dst, K, None, iterationsCount=400, reprojectionError=3.,
        confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 25:
        return None
    ix = inliers[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(xyz[ix], dst[ix], K, None, rvec, tvec)
    xyz_t = (cv2.Rodrigues(rvec)[0] @ xyz[ix].T + tvec).T
    if np.any(xyz_t[:, 2] <= 0):
        return None
    projected = (K @ xyz_t.T).T
    projected = projected[:, :2] / projected[:, 2:]
    reproj = np.linalg.norm(projected - dst[ix], axis=1)
    if float(np.median(reproj)) > 2.5:
        return None
    u, v = np.rint(dst[ix]).astype(int).T
    valid_uv = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    if np.count_nonzero(valid_uv) < 25:
        return None
    measured = depths[target][v[valid_uv], u[valid_uv]].astype(float) / 1000.
    predicted = xyz_t[valid_uv, 2]
    valid_z = (measured > .3) & (measured < 3.)
    if np.count_nonzero(valid_z) < 20:
        return None
    residual = np.abs(predicted[valid_z] - measured[valid_z])
    med = float(np.median(residual))
    fraction = float(np.mean(residual < .15))
    if med > .09 or fraction < .65:
        return None
    rel = np.eye(4)
    rel[:3, :3] = cv2.Rodrigues(rvec)[0]
    rel[:3, 3] = tvec[:, 0]
    if (np.linalg.norm(rel[:3, 3]) > 1.2 or
            Rotation.from_matrix(rel[:3, :3]).magnitude() > math.radians(80)):
        return None
    return dict(pose=rel @ w2c_anchor, anchor=anchor, inliers=len(ix),
                depth_witnesses=int(np.count_nonzero(valid_z)),
                depth_median_m=med, depth_agreement=fraction,
                reprojection_median_px=float(np.median(reproj)))


def pose_separation(a, b):
    ca = -a[:3, :3].T @ a[:3, 3]
    cb = -b[:3, :3].T @ b[:3, 3]
    distance = float(np.linalg.norm(ca - cb))
    angle = float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())
    return distance, angle


def choose(candidates, target):
    candidates.sort(key=lambda c: (-c['inliers'] * c['depth_agreement'],
                                   c['depth_median_m']))
    for c in candidates:
        if any(x['anchor'] != c['anchor'] and
               pose_separation(c['pose'], x['pose'])[0] < .16 and
               pose_separation(c['pose'], x['pose'])[1] < math.radians(12)
               for x in candidates):
            return c, 'two-independent-depth-verified-anchors'
        if (abs(c['anchor'] - target) <= 2 and c['inliers'] >= 40 and
                c['depth_witnesses'] >= 30 and c['depth_median_m'] < .055 and
                c['depth_agreement'] >= .8):
            return c, 'strict-single-nearby-anchor'
    return None, 'no-pose-met-depth-and-consistency-gates'


def render_board(o3d, old, new, photos, poses, out):
    """Render both meshes from exactly the same three recorded and two free cameras."""
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
    keys = sorted(poses)
    rows = []
    for k in (keys[0], keys[len(keys)//2], keys[-1]):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(small.astype(np.float32)),
            o3d.core.Tensor(poses[k].astype(np.float32)), 320, 240)
        rows.append((f'recorded-{k:03d}', rays, photos[k]))
    for i, offset in enumerate(([-.5, -.9, -.7], [.8, -1.2, .05]), 1):
        eye = center + span * np.array(offset)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            78., o3d.core.Tensor(center.astype(np.float32)),
            o3d.core.Tensor(eye.astype(np.float32)),
            o3d.core.Tensor(np.array([0., -1., 0.], np.float32)), 320, 240)
        rows.append((f'free-{i}', rays, None))
    board = Image.new('RGB', (960, 5 * 268 + 34), (18, 18, 18))
    draw = ImageDraw.Draw(board)
    draw.text((8, 10), 'original RGB (recorded only)', fill='white')
    draw.text((330, 10), 'COLMAP 150 exhaustive baseline', fill='white')
    draw.text((650, 10), 'depth-verified PnP recovery', fill='white')
    for i, (name, rays, photo) in enumerate(rows):
        y = 34 + i * 268
        draw.text((8, y + 6), name, fill='white')
        if photo is not None:
            with Image.open(photo) as im:
                board.paste(im.convert('RGB').resize((320, 240)), (0, y + 28))
        for j, (mesh, s) in enumerate(zip((old, new), scenes)):
            hit = s.cast_rays(rays)
            ids = hit['primitive_ids'].numpy().reshape(-1)
            uv = hit['primitive_uvs'].numpy().reshape(-1, 2)
            dist = hit['t_hit'].numpy().reshape(-1)
            faces = np.asarray(mesh.triangles)
            colors = np.asarray(mesh.vertex_colors)
            mask = (np.isfinite(dist) & (dist > .15) &
                    (dist < 16.) & (ids < len(faces)))
            pixels = np.zeros((320 * 240, 3), np.uint8)
            if np.any(mask):
                weights = np.column_stack((1 - uv[mask].sum(axis=1), uv[mask]))
                rgb = (colors[faces[ids[mask].astype(int)]] *
                       weights[:, :, None]).sum(axis=1)
                pixels[mask] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            board.paste(Image.fromarray(pixels.reshape(240, 320, 3)),
                        ((j + 1) * 320, y + 28))
    board.save(out / 'source-baseline-pnp-3plus2.png')


def run(source, output):
    import open3d as o3d
    import trimesh
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True)
    receipt = {'status': 'RUNNING',
               'source_artifact': 'GitHub Actions 12923839862 / run 40936161910',
               'input': 'same selected 150 real RGB + 150 measured depth frames',
               'existing_mesh_prior': False, 'full_trajectory_prior': False,
               'pose_method': 'independent COLMAP exhaustive SIFT + depth-verified RGB PnP',
               'visual_acceptance': 'NOT REVIEWED'}
    def save():
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    try:
        ledger = json.loads((source / 'inputs.json').read_text())
        assert len(ledger) == 150
        pairs = original_pairs(o3d.data.LoungeRGBDImages())
        selected = np.linspace(0, len(pairs) - 1, 150, dtype=int)
        photos, depth_paths = {}, {}
        for item, index in zip(ledger, selected):
            assert item['original_index'] == int(index)
            rgb, dep = pairs[index]
            assert hashlib.sha256(rgb.read_bytes()).hexdigest() == item['rgb_sha256']
            assert hashlib.sha256(dep.read_bytes()).hexdigest() == item['depth_sha256']
            k = item['sample']
            photos[k], depth_paths[k] = rgb, dep
        registered = read_model(source / 'model-txt-0')
        scale = json.loads((source / 'receipt.json').read_text())['metric_scale_m_per_colmap_unit']
        original_poses = {}
        for name, ext, _ in registered:
            k = int(Path(name).stem)
            metric = ext.copy()
            metric[:3, 3] *= scale
            original_poses[k] = metric
        assert len(original_poses) == 94
        sift = cv2.SIFT_create(nfeatures=8192)
        features, depths = {}, {}
        for k in range(150):
            gray = cv2.imread(str(photos[k]), cv2.IMREAD_GRAYSCALE)
            assert gray is not None and gray.shape == (480, 640)
            features[k] = sift.detectAndCompute(gray, None)
            depths[k] = np.asarray(Image.open(depth_paths[k])).copy()
            assert depths[k].shape == (480, 640)
        poses, recovered, rejected = dict(original_poses), [], {}
        for k in range(150):
            if k in poses:
                continue
            nearby = sorted(original_poses, key=lambda j: abs(j - k))
            nearby = [j for j in nearby if abs(j - k) <= 8][:8]
            candidates = []
            for anchor in nearby:
                c = candidate(anchor, k, features, depths, original_poses[anchor])
                if c is not None:
                    candidates.append(c)
            accepted, reason = choose(candidates, k)
            if accepted is None:
                rejected[str(k)] = {'reason': reason,
                                    'candidate_count': len(candidates)}
                continue
            poses[k] = accepted['pose']
            recovered.append({'sample': k, 'decision': reason,
                              **{x: y for x, y in accepted.items() if x != 'pose'}})
        receipt.update(original_registered=len(original_poses),
                       recovered=len(recovered), final_registered=len(poses),
                       recovered_details=recovered, rejected=rejected)
        save()
        intr = o3d.camera.PinholeCameraIntrinsic(640, 480, 525., 525., 319.5, 239.5)
        def volume():
            return o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=.025, sdf_trunc=.10,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        baseline, extended = volume(), volume()
        for k in sorted(poses):
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.io.read_image(str(photos[k])),
                o3d.io.read_image(str(depth_paths[k])),
                depth_scale=1000., depth_trunc=3.,
                convert_rgb_to_intensity=False)
            extended.integrate(rgbd, intr, poses[k])
            if k in original_poses:
                baseline.integrate(rgbd, intr, poses[k])
        old, new = baseline.extract_triangle_mesh(), extended.extract_triangle_mesh()
        if not old.has_triangles() or not new.has_triangles():
            raise RuntimeError('Missing real colored Mesh')
        def export(mesh, filename):
            trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                vertex_colors=np.clip(np.asarray(mesh.vertex_colors) * 255, 0, 255).astype(np.uint8),
                process=False).export(output / filename)
        export(old, 'baseline150.glb')
        export(new, 'pnp150.glb')
        render_board(o3d, old, new, photos, original_poses, output)
        receipt.update(status='REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED',
                       visual_acceptance='FAIL until image inspection',
                       comparison='source-baseline-pnp-3plus2.png',
                       original_faces=len(old.triangles),
                       recovered_faces=len(new.triangles))
    except Exception as exc:
        receipt.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
