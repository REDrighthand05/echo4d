"""Recover missing Stanford COLMAP camera poses using RGB-only 2D-3D PnP.

All 300 images come from the frozen independent 300-RGB dataset. The existing
COLMAP point cloud supplies only sparse feature correspondences, never a Mesh
or camera pose for an unregistered image. No depth enters localization. The
same 30-measured-depth budget is redistributed over registered+rescued views;
the original 30-depth GLB is an A/B reference, never a geometry prior.

A failed PnP is not replaced by an invented interpolated camera trajectory.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from stanford_colmap_metric300 import original_pairs, read_model, depth_scale
from stanford_colmap300rgb_75_30depth import run as fuse_300, nested_depth_indices, K

CAMERA_K = np.array([[525., 0., 319.5], [0., 525., 239.5], [0., 0., 1.]], np.float64)
MIN_INLIERS = 25
MAX_MEDIAN_ERROR_PX = 3.0


def make_references(sift, image_path, observations):
    """Attach COLMAP's world points to OpenCV SIFT pixels by image position."""
    keypoints, descriptors = sift.detectAndCompute(cv2.imread(str(image_path), 0), None)
    if descriptors is None or len(observations) < 30:
        return None
    coords = np.asarray([[x, y] for x, y, _ in observations], np.float64)
    xyz = np.asarray([p for _, _, p in observations], np.float64)
    distance, index = cKDTree(coords).query(np.asarray([kp.pt for kp in keypoints]), k=1)
    attached = {i: xyz[int(j)] for i, (d, j) in enumerate(zip(distance, index)) if d < 2.0}
    if len(attached) < 30:
        return None
    return keypoints, descriptors, attached


def solve_missing(target, reference_keys, refs, photos, sift, matcher, feature_cache):
    if target not in feature_cache:
        feature_cache[target] = sift.detectAndCompute(cv2.imread(str(photos[target]), 0), None)
    keypoints, desc = feature_cache[target]
    if desc is None or len(keypoints) < 40:
        return None
    candidates = {}
    for ref_key in sorted(reference_keys, key=lambda k: abs(k - target))[:12]:
        ref = refs.get(ref_key)
        if ref is None:
            continue
        _, ref_desc, known_xyz = ref
        for pair in matcher.knnMatch(desc, ref_desc, k=2):
            if len(pair) != 2 or pair[0].distance >= .72 * pair[1].distance:
                continue
            m = pair[0]
            if m.trainIdx not in known_xyz:
                continue
            previous = candidates.get(m.queryIdx)
            if previous is None or m.distance < previous[0]:
                candidates[m.queryIdx] = (m.distance, known_xyz[m.trainIdx])
    if len(candidates) < 40:
        return None
    qidx = sorted(candidates)
    xy = np.asarray([keypoints[i].pt for i in qidx], np.float64)
    xyz = np.asarray([candidates[i][1] for i in qidx], np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        xyz, xy, CAMERA_K, None, flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=4., confidence=.999, iterationsCount=2000)
    if not ok or inliers is None or len(inliers) < MIN_INLIERS:
        return None
    inds = inliers.ravel()
    # Refit only on robust inliers; reject clustered and high-error solutions.
    ok, rvec, tvec = cv2.solvePnP(xyz[inds], xy[inds], CAMERA_K, None,
                                  rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(xyz[inds], rvec, tvec, CAMERA_K, None)
    error = np.linalg.norm(projected.reshape(-1, 2) - xy[inds], axis=1)
    median = float(np.median(error))
    spread = np.ptp(xy[inds], axis=0)
    area = float(np.prod(spread) / (640 * 480))
    rot, _ = cv2.Rodrigues(rvec)
    camera_z = (rot @ xyz[inds].T + tvec).T[:, 2]
    if median > MAX_MEDIAN_ERROR_PX or area < .02 or np.mean(camera_z > .1) < .95:
        return None
    # The inlier query descriptors become a new RGB-only bridge toward older
    # frames; every attached 3D point still comes from the original COLMAP map.
    associated = {qidx[int(j)]: xyz[int(j)] for j in inds}
    return dict(rot=rot, t=tvec.ravel(), inliers=len(inds),
                median_error_px=round(median, 3),
                inlier_image_area=round(area, 4)), (keypoints, desc, associated)


def rescue(source, output):
    import open3d as o3d

    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='RUNNING', source_run=40906999521,
                   selected_rgb=300, localization='OpenCV SIFT + original COLMAP 3D tracks + RGB-only PnP',
                   measured_depth_for_localization=0, old_mesh_prior=False,
                   interpolated_camera_poses=False, visual_acceptance='NOT REVIEWED')
    def save():
        (output / 'rescue-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    ledger = json.loads((source / 'inputs.json').read_text())
    if len(ledger) != 300:
        raise ValueError('Expected exactly 300 source RGB images')
    pairs = original_pairs(o3d.data.LoungeRGBDImages())
    photos = {}
    for k, item in enumerate(ledger):
        if item['sample'] != k:
            raise ValueError('Out-of-order RGB ledger')
        rgb = pairs[item['original_index']][0]
        if hashlib.sha256(rgb.read_bytes()).hexdigest() != item['rgb_sha256']:
            raise ValueError('Original RGB SHA mismatch')
        photos[k] = rgb
    views = read_model(source / 'model-txt-0')
    registered = {int(Path(name).stem): (name, ext, obs) for name, ext, obs in views}
    missing = set(range(300)) - set(registered)
    receipt.update(original_registered=len(registered), missing_before=sorted(missing))
    save()

    sift = cv2.SIFT_create(nfeatures=6000)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    feature_cache = {}
    refs = {}
    relevant = {r for r in registered if any(abs(r - m) <= 45 for m in missing)}
    for k, (_, _, obs) in registered.items():
        if k not in relevant:
            continue
        r = make_references(sift, photos[k], obs)
        if r is not None:
            refs[k] = r
    receipt['initial_reference_views'] = len(refs)
    recovered = {}
    attempts = {}
    # Nearest registered camera first; successful PnP frames become bridges.
    for iteration in range(4):
        progress = 0
        pending = sorted(missing - set(recovered),
                         key=lambda k: min(abs(k - r) for r in refs))
        for k in pending:
            nearby = [r for r in refs if abs(r - k) <= 45]
            if not nearby:
                attempts[k] = 'no reference within 45 frames'
                continue
            found = solve_missing(k, nearby, refs, photos, sift, matcher, feature_cache)
            if found is None:
                attempts[k] = 'insufficient reliable 2D-3D RGB correspondences'
                continue
            pose, bridge = found
            recovered[k] = pose
            refs[k] = bridge
            progress += 1
        if progress == 0:
            break
    receipt.update(recovered_count=len(recovered), recovered={
        str(k): {x: v for x, v in p.items() if x not in ('rot', 't')}
        for k, p in sorted(recovered.items())},
        still_missing=sorted(missing - set(recovered)),
        failed_attempts={str(k): attempts.get(k, 'not attempted')
                         for k in sorted(missing - set(recovered))})
    save()
    if not recovered:
        receipt.update(status='NO_POSE_RECOVERY_NO_NEW_MESH',
                       visual_acceptance='FAIL: no RGB-only pose rescue')
        save()
        return receipt

    modified = output / 'rescued-source'
    modified.mkdir()
    shutil.copy2(source / 'inputs.json', modified / 'inputs.json')
    shutil.copytree(source / 'model-txt-0', modified / 'model-txt-0')
    images_file = modified / 'model-txt-0' / 'images.txt'
    original_lines = images_file.read_text()
    image_rows = [line.split() for line in original_lines.splitlines()
                  if line and not line.startswith('#') and len(line.split()) == 10]
    camera_id = image_rows[0][8]
    next_id = max(int(v[0]) for v in image_rows) + 1
    with images_file.open('a') as file:
        if not original_lines.endswith('\n'):
            file.write('\n')
        for k, pose in sorted(recovered.items()):
            qx, qy, qz, qw = Rotation.from_matrix(pose['rot']).as_quat()
            tx, ty, tz = pose['t']
            name = f'{k:06d}{photos[k].suffix.lower()}'
            file.write(f'{next_id} {qw:.16g} {qx:.16g} {qy:.16g} {qz:.16g} '
                       f'{tx:.16g} {ty:.16g} {tz:.16g} {camera_id} {name}\n')
            # Non-empty dummy observation keeps the legacy COLMAP text reader
            # synchronized. -1 denotes no original triangulated 3D point.
            file.write('0 0 -1\n')
            next_id += 1
    new_views = read_model(modified / 'model-txt-0')
    assert len(new_views) == len(views) + len(recovered)
    candidate = output / 'candidate'
    fuse_300(modified, candidate)
    receipt.update(status='REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED',
                   candidate_glb='candidate/measured30.glb',
                   candidate_five_views='candidate/source-300rgb-depth-registered-75-30-3plus2.png',
                   candidate_depth30=json.loads((candidate / 'receipt.json').read_text())
                                    ['selected_depth_samples']['30'],
                   visual_acceptance='FAIL until same-ray visual comparison')
    save()
    return receipt


def comparison(output, baseline, source):
    """Actual same-camera old/new 30-depth GLB renders, not stock images."""
    import open3d as o3d
    import trimesh
    old_receipt = json.loads((baseline / 'receipt.json').read_text())
    new_receipt = json.loads((output / 'candidate' / 'receipt.json').read_text())
    ratio = (old_receipt['metric_scale_m_per_colmap_unit'] /
             new_receipt['metric_scale_m_per_colmap_unit'])
    def load(path, rescale=1.):
        tri = trimesh.load(path, force='mesh', process=False)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tri.vertices) * rescale)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tri.faces))
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(tri.visual.vertex_colors)[:, :3] / 255.)
        return mesh
    meshes = [load(baseline / 'measured30.glb'),
              load(output / 'candidate' / 'measured30.glb', ratio)]
    scenes = []
    for mesh in meshes:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        scenes.append(scene)
    views = read_model(source / 'model-txt-0')
    anchors = nested_depth_indices([int(Path(name).stem) for name, _, _ in views])[2]
    depths = {}
    import open3d as o3d_data
    pairs = original_pairs(o3d_data.data.LoungeRGBDImages())
    ledger = json.loads((source / 'inputs.json').read_text())
    for name, _, _ in views:
        k = int(Path(name).stem)
        if k in anchors:
            depths[name] = pairs[ledger[k]['original_index']][1]
    scale, _ = depth_scale([v for v in views if int(Path(v[0]).stem) in anchors], depths)
    poses = {int(Path(name).stem): np.column_stack((ext[:, :3], ext[:, 3]))
             for name, ext, _ in views}
    for ext in poses.values():
        ext[:3, 3] *= scale
    keys = sorted(poses)
    center = (meshes[0].get_min_bound() + meshes[0].get_max_bound()) / 2
    span = meshes[0].get_max_bound() - meshes[0].get_min_bound()
    half = K.copy()
    half[:2] /= 2
    half[0, 2] = (K[0, 2] + .5) / 2 - .5
    half[1, 2] = (K[1, 2] + .5) / 2 - .5
    rows = []
    for k in (keys[0], keys[len(keys)//2], keys[-1]):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(half.astype(np.float32)),
            o3d.core.Tensor(poses[k].astype(np.float32)), 320, 240)
        rows.append((f'recorded-{k:03d}', rays,
                     pairs[ledger[k]['original_index']][0]))
    for j, offset in enumerate(([-.5, -.9, -.7], [.8, -1.2, .05]), 1):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            78., o3d.core.Tensor(center.astype(np.float32)),
            o3d.core.Tensor((center + span * offset).astype(np.float32)),
            o3d.core.Tensor(np.array([0., -1., 0.], np.float32)), 320, 240)
        rows.append((f'free-{j}', rays, None))
    image = Image.new('RGB', (960, 5 * 268 + 34), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    for j, label in enumerate(('real RGB / free', 'old 300RGB+30depth', 'PnP rescued 300RGB+30depth')):
        draw.text((j * 320 + 8, 10), label, fill='white')
    coverage = {}
    for i, (name, rays, photo) in enumerate(rows):
        y = 34 + i * 268
        draw.text((8, y + 5), name, fill='white')
        if photo is not None:
            with Image.open(photo) as im:
                image.paste(im.convert('RGB').resize((320, 240)), (0, y + 28))
        for j, (mesh, scene) in enumerate(zip(meshes, scenes)):
            hits = scene.cast_rays(rays)
            ids = hits['primitive_ids'].numpy().reshape(-1)
            uv = hits['primitive_uvs'].numpy().reshape(-1, 2)
            distances = hits['t_hit'].numpy().reshape(-1)
            faces = np.asarray(mesh.triangles)
            colors = np.asarray(mesh.vertex_colors)
            valid = np.isfinite(distances) & (distances > .15) & (distances < 16.) & (ids < len(faces))
            pixels = np.zeros((320 * 240, 3), np.uint8)
            if valid.any():
                weights = np.column_stack((1 - uv[valid].sum(axis=1), uv[valid]))
                pixels[valid] = np.clip((colors[faces[ids[valid].astype(int)]] *
                                         weights[:, :, None]).sum(axis=1) * 255, 0, 255).astype(np.uint8)
            image.paste(Image.fromarray(pixels.reshape(240, 320, 3)), ((j + 1) * 320, y + 28))
            coverage[f'{name}_{"old" if j == 0 else "rescued"}'] = round(float(valid.mean()), 4)
    image.save(output / 'source-old-vs-pnp-rescue-3plus2.png')
    return coverage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = rescue(args.source, args.output)
    if receipt['status'] == 'REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED':
        try:
            receipt['same_ray_coverage'] = comparison(args.output, args.baseline, args.source)
            receipt['same_ray_image'] = 'source-old-vs-pnp-rescue-3plus2.png'
        except Exception as exc:
            receipt['comparison_error'] = f'{type(exc).__name__}: {exc}'
        (args.output / 'rescue-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
