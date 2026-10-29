"""Known-pose RGB-only OpenCV StereoSGBM trial; no sensor Z or prior Mesh.

Source RGBs and poses must be in an echo-dataset-1 recording directory.
The input poses are known metric camera-to-world matrices, flattened column-major.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


def camera_to_world(frame):
    return np.asarray(frame['cameraToWorld'], dtype=np.float64).reshape(4, 4).T


def candidates(poses, index, max_neighbors=8):
    ref = poses[index]
    forward = ref[:3, 2]
    center = ref[:3, 3]
    scored = []
    for j, pose in enumerate(poses):
        if j == index:
            continue
        offset = pose[:3, 3] - center
        baseline = np.linalg.norm(offset)
        facing = float(forward @ pose[:3, 2])
        transverse = np.linalg.norm(np.cross(forward, offset)) / max(baseline, 1e-9)
        if 0.08 <= baseline <= 0.55 and facing >= 0.75 and transverse >= 0.5:
            scored.append((baseline * facing * transverse / (1 + 0.015 * abs(j-index)), j))
    return [j for _, j in sorted(scored, reverse=True)[:max_neighbors]]


def stereo_pair(left, right, left_pose, right_pose, K, max_depth=4.0):
    """Rectify known metric cameras; infer Z via SGBM and L/R disparity agreement."""
    h, w = left.shape[:2]
    relative = np.linalg.inv(right_pose) @ left_pose
    R, T = relative[:3, :3], relative[:3, 3]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, None, K, None, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    # This first trial handles horizontal rectification only; vertical pairs are reported missing.
    if abs(P2[1, 3]) > abs(P2[0, 3]):
        return None, 'vertical_baseline'
    maps = [cv2.initUndistortRectifyMap(K, None, rot, proj, (w, h), cv2.CV_32FC1)
            for rot, proj in ((R1, P1), (R2, P2))]
    a = cv2.remap(left, *maps[0], cv2.INTER_LINEAR)
    b = cv2.remap(right, *maps[1], cv2.INTER_LINEAR)
    a_gray, b_gray = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (a, b))
    direction = 1 if P2[0, 3] < 0 else -1

    def matcher(sign):
        minimum = 0 if sign > 0 else -128
        return cv2.StereoSGBM_create(
            minDisparity=minimum, numDisparities=128, blockSize=5,
            P1=8*25, P2=32*25, disp12MaxDiff=1, uniquenessRatio=12,
            speckleWindowSize=60, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    d = matcher(direction).compute(a_gray, b_gray).astype(np.float32) / 16
    dr = matcher(-direction).compute(b_gray, a_gray).astype(np.float32) / 16
    xx = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w))
    xr = np.rint(xx - d).astype(np.int32)
    inside = (xr >= 0) & (xr < w)
    reverse = np.take_along_axis(dr, np.clip(xr, 0, w-1), axis=1)
    valid = inside & (np.abs(d + reverse) <= 1.5)
    valid &= (d * direction > 0.5) & (d * direction < 127)
    xyz = cv2.reprojectImageTo3D(d, Q)
    valid &= np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > 0.35) & (xyz[:, :, 2] < max_depth)
    x, y, rw, rh = roi1
    if rw <= 0 or rh <= 0:
        return None, 'empty_rectified_roi'
    roi = np.zeros((h, w), dtype=bool)
    roi[y:y+rh, x:x+rw] = True
    valid &= roi
    xyz[~valid] = 0
    return {'xyz': xyz, 'valid': valid, 'color': a, 'R1': R1, 'disparity': d}, None


def mesh_from_pair(pair, camera_pose, stride=3, max_edge_m=0.12):
    """Triangulate only neighboring valid pixels, never bridge missing surfaces."""
    xyz = pair['xyz'][::stride, ::stride]
    mask = pair['valid'][::stride, ::stride]
    rgb = cv2.cvtColor(pair['color'], cv2.COLOR_BGR2RGB)[::stride, ::stride]
    h, w = mask.shape
    camera_xyz = xyz.reshape(-1, 3) @ pair['R1']
    world = camera_xyz @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    ids = np.arange(h*w, dtype=np.int32).reshape(h, w)
    quads = np.stack([ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]], axis=-1).reshape(-1, 4)
    good = mask.reshape(-1)[quads].all(axis=1)
    quads = quads[good]
    if not len(quads):
        return None
    # Require both world-space proximity and relative Z continuity.
    corners = world[quads]
    longest = np.linalg.norm(corners[:, :, None] - corners[:, None, :], axis=-1).max(axis=(1, 2))
    depths = xyz[:, :, 2].reshape(-1)[quads]
    keep = (longest <= max_edge_m) & ((depths.max(axis=1)-depths.min(axis=1)) < .08)
    q = quads[keep]
    if not len(q):
        return None
    faces = np.concatenate([q[:, [0, 2, 1]], q[:, [1, 2, 3]]])
    used, inverse = np.unique(faces, return_inverse=True)
    return trimesh.Trimesh(vertices=world[used], faces=inverse.reshape(-1, 3),
                           vertex_colors=rgb.reshape(-1, 3)[used], process=False)


def run(recording, output, limit=48, stride=3):
    source, dest = Path(recording), Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'recording.json').read_text())
    frames = manifest['frames']
    poses = [camera_to_world(f) for f in frames]
    k = manifest['intrinsics']
    K = np.array([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1]], dtype=np.float64)
    images = [cv2.imread(str(source/f['color'])) for f in frames]
    if any(im is None for im in images):
        raise ValueError('Missing source RGB image')
    # Evenly spread reference cameras across the entire real room trajectory.
    refs = np.unique(np.linspace(0, len(frames)-1, min(limit, len(frames)), dtype=int))
    meshes, trials, failures = [], [], {}
    for i in refs:
        best = None
        for j in candidates(poses, i, 6):
            pair, failure = stereo_pair(images[i], images[j], poses[i], poses[j], K)
            if failure:
                failures[failure] = failures.get(failure, 0) + 1
                continue
            pixels = int(pair['valid'].sum())
            if best is None or pixels > best[0]:
                best = (pixels, j, pair)
        if best is None or best[0] < 500:
            trials.append({'reference': int(i), 'status': 'no_supported_stereo', 'best_pixels': 0 if best is None else best[0]})
            continue
        pixels, j, pair = best
        mesh = mesh_from_pair(pair, poses[i], stride=stride)
        if mesh is None:
            trials.append({'reference': int(i), 'status': 'no_continuous_faces', 'best_pixels': pixels})
            continue
        meshes.append(mesh)
        trials.append({'reference': int(i), 'partner': int(j), 'valid_stereo_pixels': pixels,
                       'faces': int(len(mesh.faces)), 'status': 'mesh_generated'})
    if not meshes:
        raise RuntimeError('No real RGB stereo geometry could be reconstructed')
    merged = trimesh.util.concatenate(meshes)
    merged.export(dest/'opencv-sgbm-known-pose-rgb.glb')
    merged.export(dest/'opencv-sgbm-known-pose-rgb.ply')
    report = {'method': 'OpenCV StereoSGBM 4.13, known-pose metric rectification, left-right consistency',
              'source': 'TUM fr1_room 96 real RGB 320x240, 96 real metric camera poses',
              'removed': ['measured sensor depth pixels', 'prior Mesh'],
              'not_removed': ['real camera poses', 'camera intrinsics'],
              'unknowns': ['vertical stereo pairs skipped', 'holes and false stereo matches remain'],
              'frames': len(frames), 'attempted_references': len(refs),
              'successful_references': len(meshes), 'faces': int(len(merged.faces)),
              'vertices': int(len(merged.vertices)), 'pair_failures': failures, 'trials': trials,
              'visual_acceptance': 'UNREVIEWED; never infer quality from face count'}
    (dest/'report.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=48)
    parser.add_argument('--stride', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.recording, args.output, args.limit, args.stride), indent=2))
