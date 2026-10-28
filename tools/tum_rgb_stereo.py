"""Real TUM fr1_xyz ablation: measured depth removed, known metric poses retained.

RGB-only depth estimation uses CPU OpenCV calibrated stereo. The source
depth.txt timestamp index is used solely to reproduce the frozen frame
selection, never depth PNG pixels. No holdout RGB or depth trains the Mesh.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
import shutil
import re
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from echo4d.model import Camera, DataError, file_sha256, safe_path
from echo4d.tum import read_rows, interpolate_pose, nearest_timestamp
from echo4d.holdout import read_echo_ply, rasterize_mesh_depth
from echo4d.known_pose_stereo import infer_stereo, rectify_geometry
from echo4d.stereo_crossview import reject_crossview_foreground
from echo4d.stereo_surface_extension import extend_contiguous_stereo_depth
from echo4d.output import output_transaction, write_json
from tum_public_roundtrip import URL, ROOT, PINNED_ARCHIVE_SHA256, PINNED_ARCHIVE_BYTES, BOUNDS, download, extract

HOLDOUT = (4, 5, 9, 10, 14, 19, 24, 25, 29, 34, 35, 55)
TOTAL = 60


def archive_stereo_inputs(manifest: Path, frames: list[dict], source_root: Path,
                          destination: Path) -> dict:
    """Retain actual inferred Z and raw metadata; source RGB stays in pinned TUM.

    This is a replay bundle, not a self-contained video/photo distribution or a
    claim that inferred stereo Z is sensor-measured depth.
    """
    manifest_raw = manifest.read_bytes()
    if len(manifest_raw) > 8 * 1024 * 1024:
        raise DataError('Stereo manifest exceeds archive budget')
    data = json.loads(manifest_raw)
    files = {'recording.json': manifest}
    rgb_hashes = {}
    for frame in data['frames']:
        depth_name = frame['depth']
        if not isinstance(depth_name, str) or not re.fullmatch(r"depth/[0-9]{5}\.npy", depth_name):
            raise DataError("Unsafe stereo archive depth path")
        files[depth_name] = safe_path(manifest.parent, depth_name)
        color = safe_path(manifest.parent, frame['color'])
        rgb_hashes[frame['color']] = file_sha256(color)
    for name in ('rgb.txt', 'depth.txt', 'groundtruth.txt'):
        files['source-index/' + name] = safe_path(source_root, name)
    if sum(p.stat().st_size for p in files.values()) > 96 * 1024 * 1024:
        raise DataError('Stereo replay archive exceeds 96 MiB budget')
    pins = {name: file_sha256(p) for name, p in files.items()}
    source_frames = []
    for frame in frames:
        if file_sha256(frame['rgb']) != frame['rgb_sha256']:
            raise DataError('Source RGB changed before archival')
        source_frames.append({'index': frame['index'], 't_s': frame['t_s'],
            'rgb': str(frame['rgb'].relative_to(source_root.resolve())),
            'rgb_sha256': frame['rgb_sha256'],
            'cameraToWorld': frame['pose'].flatten(order='F').tolist()})
    with output_transaction(destination) as folder:
        receipt = {}
        for name, source in files.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_sha256(target) != pins[name]:
                raise DataError('Stereo archive input changed during copy')
            receipt[name] = {'sha256': pins[name], 'bytes': target.stat().st_size}
        # Ensure the manifest snapshot is exactly the bytes parsed above.
        if (folder / 'recording.json').read_bytes() != manifest_raw:
            raise DataError('Stereo manifest changed during archive preparation')
        write_json(folder / 'source-frames.json', {'frames': source_frames,
            'holdout_indices': list(HOLDOUT), 'archive_sha256': PINNED_ARCHIVE_SHA256})
        receipt['source-frames.json'] = {'sha256': file_sha256(folder / 'source-frames.json'),
                                        'bytes': (folder / 'source-frames.json').stat().st_size}
        write_json(folder / 'archive.json', {'files': receipt,
            'depth_kind': 'stereo_inferred_camera_z_not_measured',
            'omitted_training_rgb_sha256': rgb_hashes,
            'rgb_note': 'RGB not redistributed; reproduce from pinned official TUM and source frame identities'})
        if any(file_sha256(source) != pins[name] for name, source in files.items()):
            raise DataError('Stereo input changed before archive publication')
    return receipt






def rgb_pose_frames(root: Path) -> list[dict]:
    """Match the prior 60-frame protocol without opening any depth image."""
    # depth.txt timestamps are a selection index, NOT measured depth values.
    index = read_rows(root / 'depth.txt', 2)
    rgb_rows = read_rows(root / 'rgb.txt', 2)
    pose_rows = np.asarray(read_rows(root / 'groundtruth.txt', 8), dtype=np.float64)
    if not np.isfinite(pose_rows).all() or np.any(np.abs(np.linalg.norm(pose_rows[:, 4:8], axis=1) - 1) > .01):
        raise DataError('Invalid known pose quaternion')
    rotations = Rotation.from_quat(pose_rows[:, 4:8])
    rgb_times = np.array([float(row[0]) for row in rgb_rows])
    world = np.eye(4)
    world[:3, :3] = [[1,0,0], [0,0,1], [0,-1,0]]
    frames = []
    for timestamp, _unused_depth_filename in index[::12]:
        if len(frames) >= TOTAL:
            break
        t = float(timestamp)
        pose = interpolate_pose(pose_rows[:, 0], pose_rows[:, 1:4], rotations, t, .1)
        rgb_idx = nearest_timestamp(rgb_times, t, .03)
        if pose is None or rgb_idx is None:
            continue
        rgb_path = safe_path(root, rgb_rows[rgb_idx][1])
        with Image.open(rgb_path) as im:
            if im.size != (640, 480):
                raise DataError('Expected original TUM 640x480 RGB')
        frames.append({'index': len(frames), 't_s': t, 'rgb': rgb_path,
                       'rgb_sha256': file_sha256(rgb_path), 'pose': world @ pose})
    if len(frames) != TOTAL:
        raise DataError(f'Frozen RGB+pose selection yielded {len(frames)} instead of 60')
    return frames


def select_pairs(frames: list[dict], camera: Camera, max_pairs: int = 20,
                 backups_per_left: int = 3) -> list[tuple[int, int]]:
    """Bounded train-only stereo candidates with alternative baselines per left view.

    Geometry-only ranking cannot tell whether a pair has texture/overlap. Keep
    several fixed, pre-ranked candidates per selected training view, so a bad
    first match does not discard that view. No holdout pixels or scores used.
    """
    if not 1 <= max_pairs <= 48 or not 1 <= backups_per_left <= 4:
        raise DataError('Stereo candidate budget outside fixed bounds')
    training = [i for i in range(TOTAL) if i not in HOLDOUT]
    candidates: dict[int, list[tuple[float, int]]] = {}
    for pos, i in enumerate(training):
        for j in training[pos + 1:]:
            if j - i > 14:
                break
            for left, right in ((i, j), (j, i)):
                try:
                    geom = rectify_geometry(camera, frames[left]['pose'], frames[right]['pose'])
                except DataError:
                    continue
                baseline, angle = geom[-2:]
                score = abs(baseline - .11) + .06 * angle + .002 * (j - i)
                candidates.setdefault(left, []).append((score, right))
    anchors = sorted(candidates)
    if len(anchors) > max_pairs:
        selected = np.linspace(0, len(anchors) - 1, max_pairs, dtype=int)
        anchors = [anchors[int(k)] for k in selected]
    return [(left, right) for left in anchors
            for _score, right in sorted(candidates[left])[:backups_per_left]]


def stereo_resolution(width: int) -> tuple[Camera, int, int]:
    """TUM fr1 calibration at native 640 or exact 2x decimation.

    Width is fixed before any holdout inspection. The metric baseline is
    supplied by the known camera poses, not by stereo alone.
    """
    if type(width) is not int or width not in (320, 640):
        raise DataError('Stereo width must be 320 or native 640 pixels')
    scale = width / 640
    camera = Camera(width, int(480 * scale), 525.0 * scale, 525.0 * scale,
                    319.5 * scale, 239.5 * scale)
    return camera, 640 // width, (128 if width == 640 else 64)


def load_stereo_rgb(path: Path, width: int) -> np.ndarray:
    """Read original source pixels; decimate only for the 320 control."""
    _camera, stride, _disparities = stereo_resolution(width)
    with Image.open(path) as im:
        if im.size != (640, 480):
            raise DataError('Expected original TUM 640x480 RGB')
        pixels = np.asarray(im.convert('RGB')).copy()
    return pixels[::stride, ::stride].copy()


def reduce_native_depth_for_fusion(depth: np.ndarray) -> np.ndarray:
    """Conservative 2x2 consolidation before TSDF, with no invented surfaces."""
    if depth.shape != (480, 640) or not np.isfinite(depth).all() or (depth < 0).any():
        raise DataError('Expected finite nonnegative native 640x480 depth')
    blocks = depth.reshape(240, 2, 320, 2).transpose(0, 2, 1, 3).reshape(240, 320, 4)
    valid = blocks > 0
    count = valid.sum(axis=-1)
    ordered = np.sort(np.where(valid, blocks, np.inf), axis=-1)
    mid = np.take_along_axis(ordered, np.maximum((count - 1) // 2, 0)[..., None], axis=-1)[..., 0]
    high = np.take_along_axis(ordered, np.maximum(count - 1, 0)[..., None], axis=-1)[..., 0]
    mid = np.where(count > 0, mid, 0)
    high = np.where(count > 0, high, 0)
    agree = (count == 1) | ((high - ordered[..., 0]) <= np.maximum(.08, .05 * mid))
    return np.where((count > 0) & agree, mid, 0).astype(np.float32)



def fuse_stereo_candidate_depths(depths: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Fuse up to three independent RGB stereo baselines in one *original* camera.

    Zero is unobserved. Keep isolated supported pixels, take agreeing pairs,
    and discard conflicting overlaps; do not fill unseen surfaces or use Z GT.
    """
    if not 1 <= len(depths) <= 3:
        raise DataError('Stereo consensus requires one to three depth candidates')
    shape = depths[0].shape
    if len(shape) != 2 or any(
        not isinstance(d, np.ndarray) or d.shape != shape
        or not np.issubdtype(d.dtype, np.number)
        or not np.isfinite(d).all() or (d < 0).any() for d in depths
    ):
        raise DataError('Stereo consensus requires same-size finite nonnegative depth')
    stack = np.stack(depths).astype(np.float32)
    valid = stack > 0
    count = valid.sum(axis=0)
    ordered = np.sort(np.where(valid, stack, np.inf), axis=0)
    a = ordered[0]
    b = ordered[1] if len(depths) > 1 else np.full(shape, np.inf, np.float32)
    c = ordered[2] if len(depths) > 2 else np.full(shape, np.inf, np.float32)
    agree_ab = (b - a) <= np.maximum(.08, .05 * a)
    agree_bc = (c - b) <= np.maximum(.08, .05 * b)
    merged = np.zeros(shape, np.float32)
    merged[count == 1] = a[count == 1]
    two = (count == 2) & agree_ab
    merged[two] = (a[two] + b[two]) * .5
    three = count == 3
    both = three & agree_ab & agree_bc
    merged[both] = b[both]
    lower = three & agree_ab & ~agree_bc
    merged[lower] = (a[lower] + b[lower]) * .5
    upper = three & agree_bc & ~agree_ab
    merged[upper] = (b[upper] + c[upper]) * .5
    # With three mutually inconsistent depths, the pixel is unknown.
    rejected = (count >= 2) & (merged == 0)
    return merged, {
        'candidate_count': len(depths),
        'single_candidate_pixels': int(np.count_nonzero(count == 1)),
        'agreed_multi_candidate_pixels': int(np.count_nonzero((count >= 2) & (merged > 0))),
        'conflict_rejected_pixels': int(np.count_nonzero(rejected)),
        'fused_valid_pixels': int(np.count_nonzero(merged)),
    }



def choose_contiguous_stereo_depth(depths: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Choose one train-only RGB match with the largest contiguous observed patch.

    A single baseline is selected, rather than unioning contradictory surfaces.
    Largest 8-connected valid region wins; valid count breaks ties; earlier
    candidate wins exact ties. No sensor depth, holdout, or geometry is filled.
    """
    from scipy.ndimage import label
    if not 1 <= len(depths) <= 3:
        raise DataError('Best-backup selection requires one to three candidates')
    shape = depths[0].shape
    if len(shape) != 2 or any(
        not isinstance(d, np.ndarray) or d.shape != shape
        or not np.issubdtype(d.dtype, np.number)
        or not np.isfinite(d).all() or (d < 0).any() for d in depths
    ):
        raise DataError('Best-backup depths must be finite, nonnegative, same-size arrays')
    scores = []
    for depth in depths:
        valid = depth > 0
        components, count = label(valid, structure=np.ones((3, 3), dtype=bool))
        sizes = np.bincount(components[valid], minlength=count + 1)
        scores.append([int(sizes.max()) if count else 0, int(valid.sum())])
    best = max(range(len(depths)), key=lambda i: (scores[i][0], scores[i][1], -i))
    return depths[best], {
        'candidate_count': len(depths),
        'selected_candidate_index': best,
        'candidate_support_scores_largest_component_then_pixels': scores,
        'selected_valid_pixels': scores[best][1],
    }


def build_stereo_recording(frames: list[dict], target: Path, report: dict,
                           *, stereo_width: int = 640, fusion_width: int = 320,
                           max_anchors: int = 48, merge_backups: bool = False,
                           select_best_backup: bool = False,
                           extend_best_backup: bool = False,
                           cross_view_consistency: bool = False) -> Path:
    if sum((merge_backups, select_best_backup, extend_best_backup)) > 1:
        raise DataError('Choose exactly one backup strategy')
    camera, _stride, num_disparities = stereo_resolution(stereo_width)
    fusion_camera, _fusion_stride, _ = stereo_resolution(fusion_width)
    if fusion_width > stereo_width:
        raise DataError('Fusion resolution cannot exceed stereo matching resolution')
    report['tsdf_fusion_resolution'] = [fusion_camera.width, fusion_camera.height]
    report['stereo_algorithm_resolution'] = [camera.width, camera.height]
    report['stereo_num_disparities'] = num_disparities
    report['merge_backups'] = merge_backups
    report['select_best_backup'] = select_best_backup
    report['extend_best_backup'] = extend_best_backup
    report['cross_view_consistency'] = cross_view_consistency
    pairs = select_pairs(frames, camera, max_pairs=max_anchors)
    report['anchor_budget'] = max_anchors
    report['eligible_anchor_count'] = len({left for left, _ in pairs})
    report['selected_pairs'] = pairs
    report['candidate_pairs_attemptable'] = len(pairs)
    if len({left for left, _ in pairs}) < 6:
        raise DataError(f'Only {len(pairs)} eligible stereo pairs; require >=6')
    target.mkdir(parents=True, exist_ok=False)
    (target / 'color').mkdir()
    (target / 'depth').mkdir()
    entries, pair_evidence = [], []
    accepted_left: set[int] = set()
    pending_depths: dict[int, list[np.ndarray]] = {}
    saved_left_indices: list[int] = []
    for left_idx, right_idx in pairs:
        if left_idx in accepted_left and not (merge_backups or select_best_backup or extend_best_backup):
            continue  # Legacy control: first usable pair wins.
        left, right = frames[left_idx], frames[right_idx]
        a = load_stereo_rgb(left['rgb'], stereo_width)
        b = load_stereo_rgb(right['rgb'], stereo_width)
        rgb, depth, rect_camera, rect_pose, diagnostics = infer_stereo(
            a, b, camera, left['pose'], right['pose'],
            num_disparities=num_disparities)
        item = {'left_index': left_idx, 'right_index': right_idx,
                'left_rgb_sha256': left['rgb_sha256'], 'right_rgb_sha256': right['rgb_sha256'],
                **diagnostics}
        pair_evidence.append(item)
        if diagnostics['valid_depth_pixels'] < 100:
            continue
        # The existing ECHO TSDF expects one fixed camera K. Reproject inferred
        # rectified depth to native left-camera pixels, never mislabel rectified K.
        z = depth
        yy, xx = np.indices(z.shape)
        points_rect = np.stack(((xx-rect_camera.cx)*z/rect_camera.fx,
                                (yy-rect_camera.cy)*z/rect_camera.fy, z), axis=-1)
        rotation = left['pose'][:3,:3].T @ rect_pose[:3,:3]
        points_original = points_rect @ rotation.T
        zz = points_original[...,2]
        valid = (z > 0) & (zz > 0.3)
        uu = np.rint(camera.fx*points_original[...,0]/np.maximum(zz,1e-6)+camera.cx).astype(int)
        vv = np.rint(camera.fy*points_original[...,1]/np.maximum(zz,1e-6)+camera.cy).astype(int)
        valid &= (uu>=0)&(uu<camera.width)&(vv>=0)&(vv<camera.height)
        projected = np.full((camera.height,camera.width),np.inf,dtype=np.float32)
        np.minimum.at(projected,(vv[valid],uu[valid]),zz[valid].astype(np.float32))
        projected[~np.isfinite(projected)] = 0
        if stereo_width == 640 and fusion_width == 320:
            projected = reduce_native_depth_for_fusion(projected)
        if np.count_nonzero(projected) < 100:
            continue
        if merge_backups or select_best_backup or extend_best_backup:
            pending_depths.setdefault(left_idx, []).append(projected)
            item['candidate_for_consensus'] = merge_backups
            item['candidate_for_selection'] = select_best_backup
            item['candidate_for_extension'] = extend_best_backup
            continue
        n=len(entries)
        Image.fromarray(load_stereo_rgb(left['rgb'], fusion_width)).save(
            target / 'color' / f'{n:05d}.png')
        np.save(target / 'depth' / f'{n:05d}.npy',projected,allow_pickle=False)
        entries.append({'t':float(left['t_s']-frames[0]['t_s']),
                        'cameraToWorld':left['pose'].flatten(order='F').tolist(),
                        'color':f'color/{n:05d}.png','depth':f'depth/{n:05d}.npy'})
        saved_left_indices.append(left_idx)
        item['saved_depth_sha256']=file_sha256(target / 'depth' / f'{n:05d}.npy')
        item['reprojected_valid_pixels']=int(np.count_nonzero(projected))
        accepted_left.add(left_idx)
    if merge_backups or select_best_backup or extend_best_backup:
        strategy_key = ('anchor_consensus' if merge_backups else
                        'anchor_extension' if extend_best_backup else 'anchor_selection')
        report[strategy_key] = []
        for left_idx, candidates in sorted(pending_depths.items()):
            if merge_backups:
                fused, stats = fuse_stereo_candidate_depths(candidates)
            elif extend_best_backup and len(candidates) >= 2:
                fused, stats = extend_contiguous_stereo_depth(
                    candidates, load_stereo_rgb(frames[left_idx]['rgb'], fusion_width))
            else:
                fused, stats = choose_contiguous_stereo_depth(candidates)
            report[strategy_key].append({'left_index': left_idx, **stats})
            if np.count_nonzero(fused) < 100:
                continue
            left = frames[left_idx]
            n = len(entries)
            Image.fromarray(load_stereo_rgb(left['rgb'], fusion_width)).save(
                target / 'color' / f'{n:05d}.png')
            np.save(target / 'depth' / f'{n:05d}.npy', fused, allow_pickle=False)
            entries.append({'t': float(left['t_s'] - frames[0]['t_s']),
                            'cameraToWorld': left['pose'].flatten(order='F').tolist(),
                            'color': f'color/{n:05d}.png', 'depth': f'depth/{n:05d}.npy'})
            saved_left_indices.append(left_idx)
            accepted_left.add(left_idx)
    report['pairs']=pair_evidence
    if len(entries)<6:
        raise DataError(f'Only {len(entries)} RGB-derived depth frames have >=100 valid pixels')
    if cross_view_consistency:
        # Only selected training anchors; no holdout RGB/depth or sensor Z.
        # Sort by actual frame index so adjacent observations are compared.
        order = sorted(range(len(entries)), key=lambda k: saved_left_indices[k])
        ordered_depths = [np.load(target / entries[k]['depth'], allow_pickle=False)
                          for k in order]
        ordered_poses = [frames[saved_left_indices[k]]['pose'] for k in order]
        filtered, diagnostics = reject_crossview_foreground(
            ordered_depths, ordered_poses, fusion_camera)
        receipts = []
        for pos, k in enumerate(order):
            depth_path = target / entries[k]['depth']
            old_sha = file_sha256(depth_path)
            np.save(depth_path, filtered[pos], allow_pickle=False)
            receipts.append({'left_index': saved_left_indices[k],
                             'before_sha256': old_sha,
                             'after_sha256': file_sha256(depth_path),
                             **diagnostics[pos]})
        report['cross_view_filter'] = {
            'rule': 'reject uncorroborated nearer-than-neighbor RGB-inferred Z; preserve occluded and unknown',
            'neighbor_radius': 2, 'tolerance_m': 0.12,
            'rejected_pixels': sum(x['rejected_pixels'] for x in receipts),
            'frames_with_100_pixels': sum(np.count_nonzero(d) >= 100 for d in filtered),
            'per_training_frame': receipts}
        if report['cross_view_filter']['frames_with_100_pixels'] < 6:
            raise DataError('Cross-view filtering left fewer than six usable training frames')
    entries.sort(key=lambda x:x['t'])
    data={'version':'echo-dataset-1','source':'recorded','units':'meters',
          'name':'TUM RGB stereo, known GT poses, inferred depths',
          'duration':float(max(entries[-1]['t'],.001)), 'depth_scale':1,
          'intrinsics':fusion_camera.to_dict(),'frames':entries,
          'notes':'Depth is RGB stereo-inferred, not sensor-measured. Metric scale comes exclusively from supplied TUM GT poses; depth index used only for frozen frame selection.'}
    manifest=target/'recording.json'
    manifest.write_text(json.dumps(data,allow_nan=False,indent=2)+'\n',encoding='utf8')
    report['rgb_stereo_manifest_sha256']=file_sha256(manifest)
    report['saved_frames']=len(entries)
    return manifest


def render_mesh(mesh: Path, frames: list[dict], folder: Path, report: dict,
                camera: Camera) -> None:
    from echo4d.model import rigid_matrix
    verts, faces = read_echo_ply(mesh)
    folder.mkdir(exist_ok=True)
    views=[]
    poses=[(f'heldout-{i}',frames[i]['pose']) for i in (4,24,55)]
    center=np.median(verts,axis=0)
    def look_at(eye,target):
        forward=(target-eye)/np.linalg.norm(target-eye)
        up=np.array([0.,0.,1.])
        if abs(forward@up)>.98:up=np.array([0.,1.,0.])
        right=np.cross(forward,up);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        out=np.eye(4);out[:3,:3]=np.column_stack((right,down,forward));out[:3,3]=eye
        return rigid_matrix(out.flatten(order='F'))
    poses += [('free-a',look_at(center+np.array([1.3,-1.2,.7]),center)),
              ('free-b',look_at(center+np.array([-1.1,1.4,.9]),center))]
    for name,pose in poses:
        depth, stats=rasterize_mesh_depth(verts,faces,camera,pose,far=4.0)
        valid=(depth>=.15)&(depth<=4)
        gray=np.zeros(depth.shape,dtype=np.uint8)
        gray[valid]=np.clip(np.rint(255-(depth[valid]-.15)*254/3.85),1,255).astype(np.uint8)
        path=folder/f'{name}.png';Image.fromarray(gray).save(path)
        views.append({'name':name,'mesh_render_sha256':file_sha256(path),
                      'covered_pixels':int(valid.sum()),'pose':pose.flatten(order='F').tolist(),
                      'render_stats':stats})
    report['views']=views


def execute(args: argparse.Namespace, report: dict) -> None:
    work=args.work.resolve();work.mkdir(parents=True,exist_ok=False)
    archive=work/'source.tgz'
    if args.archive:
        import shutil
        shutil.copyfile(args.archive,archive)
        report['download']={'bytes':archive.stat().st_size,'sha256':file_sha256(archive),'method':'offline pinned'}
    else:download(archive,report)
    if report['download']['sha256']!=PINNED_ARCHIVE_SHA256 or report['download']['bytes']!=PINNED_ARCHIVE_BYTES:
        raise DataError('Pinned official TUM source mismatch')
    report['extraction']=extract(archive,work/'dataset')
    frames=rgb_pose_frames(work/'dataset'/ROOT)
    camera, _stride, _num_disparities = stereo_resolution(args.stereo_width)
    report['rgb_input']={'original_resolution':[640,480],
                         'algorithm_resolution':[camera.width,camera.height],
                         'tsdf_fusion_resolution':[args.fusion_width, int(args.fusion_width*3/4)],
                         'frame_count':len(frames),'train_count':48,'holdout_indices':list(HOLDOUT),
                         'rgb_source_sha256':[f['rgb_sha256'] for f in frames],
                         'depth_png_files_opened':0,'depth_index_used_for_frame_selection_only':True,
                         'metric_pose_source':'TUM supplied groundtruth.txt; not RGB-only pose estimation'}
    manifest=build_stereo_recording(frames,work/'stereo',report,
                                    stereo_width=args.stereo_width,
                                    fusion_width=args.fusion_width,
                                    max_anchors=args.max_anchors,
                                    merge_backups=args.merge_backups,
                                    select_best_backup=args.select_best_backup,
                                    extend_best_backup=args.extend_best_backup,
                                    cross_view_consistency=args.cross_view_consistency)
    report['archived_inputs'] = archive_stereo_inputs(
        manifest, frames, work/'dataset'/ROOT, args.report_dir/'inputs')
    target=args.report_dir/'reconstruction'
    cmd=[sys.executable,'-m','echo4d','reconstruct',str(manifest),'--output',str(target),
         '--bounds',*BOUNDS,'--voxel','0.05','--max-depth','4',
         '--max-voxels','4000000','--min-observations','1','--quiet','--no-scene']
    report['reconstruct_command']=cmd
    run=subprocess.run(cmd,capture_output=True,text=True,timeout=900)
    report['reconstruct_returncode']=run.returncode
    report['reconstruct_stdout_tail']=run.stdout[-2000:]
    report['reconstruct_stderr_tail']=run.stderr[-2000:]
    if run.returncode or not (target/'mesh.ply').is_file():
        raise DataError('RGB stereo TSDF failed; see reconstruct stderr and pair evidence')
    report['mesh_sha256']=file_sha256(target/'mesh.ply')
    report['mesh_output']=json.loads(run.stdout.strip().splitlines()[-1])
    render_mesh(target/'mesh.ply',frames,args.report_dir/'visual',report,camera)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--archive',type=Path)
    p.add_argument('--stereo-width',type=int,choices=(320,640),default=640,
                   help='Algorithm RGB width: native 640 (default), or 320 control')
    p.add_argument('--fusion-width',type=int,choices=(320,640),default=320,
                   help='TSDF depth width: conservative 320 (default), or native 640')
    p.add_argument('--max-anchors',type=int,default=48,
                   help='Max fixed training anchors, never includes 12 frozen holdouts')
    p.add_argument('--merge-backups',action='store_true',
                   help='Use all valid train-only stereo backup baselines per anchor; reject conflicting overlap')
    p.add_argument('--select-best-backup', action='store_true',
                   help='Choose the most spatially contiguous train-only RGB match per anchor')
    p.add_argument('--extend-best-backup', action='store_true',
                   help='Extend best RGB stereo surface with consistent backup pixels, no inpainting')
    p.add_argument('--cross-view-consistency', action='store_true',
                   help='Reject inferred foreground contradicted by nearby train-only views')
    args=p.parse_args(argv)
    args.report_dir.mkdir(parents=True,exist_ok=True)
    report={'experiment':'tum-xyz-known-pose-RGB-stereo-no-measured-depth-v1',
            'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
            'checkout_sha':os.environ.get('GITHUB_SHA'), 'source_url':URL,
            'input_removed':'measured depth maps from reconstruction',
            'cross_view_consistency_enabled':args.cross_view_consistency,
            'stereo_candidate_fusion':('train-only multi-baseline agreement' if args.merge_backups
                                       else 'train-only largest-contiguous-backup' if args.select_best_backup
                                       else 'train-only bounded backup surface extension' if args.extend_best_backup
                                       else 'first-valid control'),
            'inputs_retained':['640x480 source RGB','TUM known metric poses','camera calibration',
                               'depth.txt timestamps for matching the frozen prior frame selection'],
            'quality':'not independent surface accuracy; no #74 acceptance'}
    start=time.perf_counter()
    try:
        execute(args,report)
        report['status']='mesh_generated_unreviewed'
    except Exception as exc:
        report['status']='failed_or_blocked'
        report['error']={'type':type(exc).__name__,'message':str(exc)}
    finally:
        report['elapsed_seconds']=round(time.perf_counter()-start,3)
        (args.report_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
        print(json.dumps({'status':report['status'],'report':str(args.report_dir/'report.json'),
                          'mesh_sha256':report.get('mesh_sha256'), 'error':report.get('error')}))
    return 0 if report['status']=='mesh_generated_unreviewed' else 1

if __name__=='__main__':raise SystemExit(main())
