"""Metric gauge from registered *measured* Z-depth and real COLMAP 2D/3D tracks.

This does not infer scene depth from RGB. It never reads GT camera poses,
never accepts a post-hoc Sim(3) evaluation scale, and never produces a TSDF.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .colmap import _cameras
from .model import DataError, file_sha256, rigid_matrix, safe_path

MIN_CORRESPONDENCES = 10
MAX_ANCHOR_DISAGREEMENT = 0.20


def read_points3d(path: Path) -> dict[int, np.ndarray]:
    if not path.is_file() or path.stat().st_size > 300 * 1024**2:
        raise DataError('Missing or oversized COLMAP points3D.txt')
    points: dict[int,np.ndarray] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        if len(fields) < 8:
            raise DataError('Invalid points3D.txt record')
        try:
            identifier = int(fields[0])
            point = np.array([float(x) for x in fields[1:4]], dtype=np.float64)
        except ValueError as e:
            raise DataError('Invalid 3D point ID/coordinate') from e
        if identifier <= 0 or identifier in points or not np.isfinite(point).all():
            raise DataError('Duplicate/invalid 3D point ID/coordinate')
        points[identifier] = point
        if len(points) > 2_000_000:
            raise DataError('COLMAP sparse cloud exceeds safety budget')
    if not points:
        raise DataError('COLMAP cloud has no 3D points')
    return points


def read_colmap_observations(path: Path, wanted: set[str]) -> dict[str,list[tuple[float,float,int]]]:
    """Read *actual observed* COLMAP image keypoints linked to 3D POINT3D_IDs."""
    if not path.is_file() or path.stat().st_size > 300 * 1024**2:
        raise DataError('Missing or oversized COLMAP images.txt')
    lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
             if not line.lstrip().startswith('#')]
    while lines and not lines[-1]:
        lines.pop()
    if len(lines) % 2:
        raise DataError('COLMAP images.txt must have pose/observation line pairs')
    found: dict[str,list[tuple[float,float,int]]] = {}
    for at in range(0,len(lines),2):
        pose = lines[at].split(maxsplit=9)
        if len(pose) != 10:
            raise DataError('Invalid COLMAP image pose line')
        name = pose[-1]
        if name not in wanted:
            continue
        if name in found:
            raise DataError('Duplicate registered anchor/holdout image')
        cells = lines[at+1].split()
        if len(cells) % 3:
            raise DataError('Invalid COLMAP 2D keypoint triples')
        observations = []
        for k in range(0,len(cells),3):
            try:
                u,v = float(cells[k]),float(cells[k+1])
                point_id = int(cells[k+2])
            except ValueError as e:
                raise DataError('Invalid COLMAP image feature coordinate') from e
            if not math.isfinite(u) or not math.isfinite(v):
                raise DataError('COLMAP image feature nonfinite')
            if point_id >= 0:
                observations.append((u,v,point_id))
        found[name] = observations
    if set(found) != wanted:
        raise DataError('Some anchor/holdout images were not registered by COLMAP: '
                        + ', '.join(sorted(wanted - set(found))))
    return found


def read_measured_depth(path: Path, scale: float, shape: tuple[int,int]) -> np.ndarray:
    if not math.isfinite(scale) or scale <= 0 or scale > 100000:
        raise DataError('Depth numeric units per meter must be explicitly valid')
    if path.suffix.lower() == '.npy':
        depth = np.load(path, allow_pickle=False)
        if depth.dtype.kind != 'f':
            raise DataError('NPY measured depth must be floating-point')
    elif path.suffix.lower() == '.png':
        with Image.open(path) as img:
            depth = np.asarray(img)
        if depth.dtype.kind not in 'ui':
            raise DataError('PNG measured depth must be integer single-channel')
    else:
        raise DataError('Only PNG depth or float NPY measured depth is supported')
    if depth.shape != shape or not np.isfinite(depth).all() or np.any(depth < 0):
        raise DataError('Measured depth dimensions/values invalid')
    depth = depth.astype(np.float64) / scale
    if np.any(depth > 100):
        raise DataError('Measured depth exceeds 100 m')
    depth.flags.writeable = False
    return depth


def sample_observed_point_depths(
    frame_pose: np.ndarray,
    points: dict[int,np.ndarray],
    observations: list[tuple[float,float,int]],
    depth: np.ndarray,
    intrinsics: Any,
    max_local_depth_range_m: float = .10,
) -> dict[str,np.ndarray | int]:
    """Return metric depth observations and COLMAP camera-Z for registered keypoints.

    Occlusion/discontinuity control is fixed *a priori*, not adjusted from GT.
    """
    world_to_camera = np.linalg.inv(frame_pose)
    z_observed,z_sfm = [],[]
    counts = defaultdict(int)
    height,width = depth.shape
    for u,v,pid in observations:
        counts['colmap_tracked_2d_features'] += 1
        point = points.get(pid)
        if point is None:
            counts['missing_sparse_points'] += 1
            continue
        position = world_to_camera[:3,:3]@point + world_to_camera[:3,3]
        if position[2] <= .001 or not np.isfinite(position).all():
            counts['behind_camera'] += 1
            continue
        expected_u = intrinsics.fx*position[0]/position[2]+intrinsics.cx
        expected_v = intrinsics.fy*position[1]/position[2]+intrinsics.cy
        if math.hypot(expected_u-u,expected_v-v) > 3.0:
            counts['failed_reprojection_check'] += 1
            continue
        col,row = int(np.rint(u)),int(np.rint(v))
        if col < 1 or row < 1 or col >= width-1 or row >= height-1:
            counts['image_border'] += 1
            continue
        patch = depth[row-1:row+2,col-1:col+2]
        # Depth discontinuity pixels are excluded BEFORE scale estimation;
        # all exclusions and their counts are reported for audit.
        if (np.any((patch < .15)|(patch > 6.0)) or
                float(patch.max()-patch.min()) > max_local_depth_range_m):
            counts['invalid_or_discontinuous_depth'] += 1
            continue
        z_observed.append(float(depth[row,col]))
        z_sfm.append(float(position[2]))
    counts['used_correspondences'] = len(z_observed)
    return {'depth_m':np.asarray(z_observed,dtype=np.float64),
            'camera_z_sfm_units':np.asarray(z_sfm,dtype=np.float64),
            'counts':dict(counts)}


def estimate_scale_from_three_anchors(observations: list[dict]) -> tuple[float,list[dict]]:
    if len(observations) != 3:
        raise DataError('Exactly three independently measured anchor views required')
    medians,details = [],[]
    for i,obs in enumerate(observations):
        observed = np.asarray(obs['depth_m'],dtype=np.float64)
        estimated = np.asarray(obs['camera_z_sfm_units'],dtype=np.float64)
        if (len(observed) < MIN_CORRESPONDENCES or observed.shape != estimated.shape or
            np.any(~np.isfinite(observed)) or np.any(~np.isfinite(estimated)) or
            np.any(observed <= 0) or np.any(estimated <= 0)):
            raise DataError(f'Anchor {i} lacks {MIN_CORRESPONDENCES} valid measured depth/SfM correspondences')
        ratios = observed/estimated
        median=float(np.median(ratios))
        mad=float(np.median(np.abs(ratios-median)))
        if median <= 0 or mad/median > .20:
            raise DataError(f'Anchor {i} has excessively scattered scale ratios')
        medians.append(median)
        details.append({'anchor_slot':i,'used_correspondences':len(ratios),
                        'median_meters_per_sfm_unit':median,
                        'relative_mad':mad/median,
                        'ratio_p10':float(np.percentile(ratios,10)),
                        'ratio_p90':float(np.percentile(ratios,90))})
    metric_scale=float(np.median(medians))
    if any(abs(s/metric_scale-1) > MAX_ANCHOR_DISAGREEMENT for s in medians):
        raise DataError('Three independent sensor-depth anchors disagree by more than 20%')
    return metric_scale,details


def describe_holdout(obs: dict, scale: float) -> dict:
    measured=np.asarray(obs['depth_m'],dtype=np.float64)
    sfm=np.asarray(obs['camera_z_sfm_units'],dtype=np.float64)
    if measured.shape != sfm.shape:
        raise DataError('Bad heldout correspondences')
    result={'measured_valid_correspondences':len(measured),
            'counts':obs.get('counts',{})}
    if not len(measured):
        return {**result,'status':'not_measured'}
    residual=sfm*scale-measured
    absolute=np.abs(residual)
    return {**result,'status':'measured',
            'mae_m':float(absolute.mean()),
            'rmse_m':float(np.sqrt(np.mean(residual**2))),
            'median_abs_m':float(np.median(absolute)),
            'p95_abs_m':float(np.percentile(absolute,95)),
            'signed_bias_m':float(residual.mean()),
            'fraction_error_at_most_0_05m':float(np.mean(absolute<=.05))}


def calibrate(trajectory_path: Path, colmap_text_dir: Path,
              depth_root: Path, depth_spec_path: Path) -> dict:
    """No ground-truth pose arguments are accepted: measured depth is sole scale source."""
    track=json.loads(trajectory_path.read_text(encoding='utf-8'))
    spec=json.loads(depth_spec_path.read_text(encoding='utf-8'))
    if (track.get('version')!='echo-sfm-trajectory-1' or
            track.get('metric_scale_estimated') is not False or
            track.get('units')!='unknown'):
        raise DataError('Cannot calibrate an already-metric or invalid SfM track')
    if (spec.get('source_kind')!='measured_registered_z_depth' or
            not isinstance(spec.get('source_description'),str) or
            len(spec['source_description'].strip())<12 or
            len(spec.get('anchors',[]))!=3 or
            len(spec.get('holdouts',[]))!=7):
        raise DataError('Need three measured-depth anchors, seven holdouts and physical provenance')
    for key in ('images','cameras'):
        p=colmap_text_dir/(key+'.txt')
        if file_sha256(p)!=track.get('model_'+key+'_txt_sha256'):
            raise DataError(f'COLMAP {key}.txt hash mismatch')
    cameras=_cameras(colmap_text_dir/'cameras.txt')
    point_file=colmap_text_dir/'points3D.txt'
    points=read_points3d(point_file)
    frames=track.get('frames')
    if not isinstance(frames,list) or len(frames)<3:
        raise DataError('Invalid/unregistered sparse track')
    frame_map={frame['name']:frame for frame in frames}
    if len(frame_map)!=len(frames):
        raise DataError('Duplicate SfM frame')
    entries=spec['anchors']+spec['holdouts']
    wanted=[item.get('name') for item in entries]
    if not all(isinstance(n,str) for n in wanted) or len(set(wanted))!=10:
        raise DataError('Anchor/holdout frame names must be unique')
    if any(name not in frame_map for name in wanted):
        raise DataError('Some metric anchor/holdout frames are not registered')
    observations=read_colmap_observations(colmap_text_dir/'images.txt',set(wanted))
    depths=[]
    for item in entries:
        if not isinstance(item,dict) or not isinstance(item.get('depth'),str):
            raise DataError('Invalid measured depth manifest entry')
        name=item['name']
        frame=frame_map[name]
        camera_id=frame['camera_id']
        if camera_id not in cameras:
            raise DataError('SfM image references unknown camera calibration')
        camera=cameras[camera_id]
        gap=item.get('timestamp_delta_seconds')
        if not isinstance(gap,(int,float)) or not math.isfinite(gap) or abs(gap)>.03:
            raise DataError('Unacceptably offset/unreported depth timestamp (>30ms)')
        path=safe_path(depth_root,item['depth'])
        depth=read_measured_depth(path,float(spec['depth_numeric_units_per_meter']),
                                  (camera.height,camera.width))
        pose=rigid_matrix(frame['camera_to_sfm_world'])
        sampled=sample_observed_point_depths(pose,points,observations[name],depth,camera)
        depths.append(sampled)
        item['depth_sha256']=file_sha256(path)
    scale,diagnostics=estimate_scale_from_three_anchors(depths[:3])
    scaled_frames=[]
    for frame in frames:
        pose=rigid_matrix(frame['camera_to_sfm_world']).copy()
        pose[:3,3]*=scale
        scaled_frames.append({'name':frame['name'],'t_s':frame['t_s'],
            'camera_id':frame['camera_id'],'camera_to_metric_sfm_world':pose.flatten(order='F').tolist()})
    return {'version':'echo-sfm-metric-scale-1','units':'meters',
            'metric_scale_estimated_from_measured_depth':True,
            'scale_m_per_sfm_unit':scale,
            'scale_is_from_external_gt_poses':False,
            'scale_evidence':'External sensor Z-depth, not RGB, not GT pose alignment',
            'sensor_claim_as_supplied':spec['source_description'],
            'source_trajectory_sha256':file_sha256(trajectory_path),
            'source_points3d_sha256':file_sha256(point_file),
            'depth_spec_sha256':file_sha256(depth_spec_path),
            'anchor_diagnostics':[{**d,'name':entries[i]['name'],
                'sensor_depth_sha256':entries[i]['depth_sha256'],
                'rgb_depth_timestamp_delta_s':entries[i]['timestamp_delta_seconds'],
                'observation_filter_counts':depths[i]['counts']}
                for i,d in enumerate(diagnostics)],
            'heldout_depth_diagnostics':[
                {'name':item['name'],'sensor_depth_sha256':item['depth_sha256'],
                 'rgb_depth_timestamp_delta_s':item['timestamp_delta_seconds'],
                 **describe_holdout(depths[i+3],scale)}
                 for i,item in enumerate(entries[3:])],
            'frames':scaled_frames,
            'limitations':['Requires independent registered metric Z depth; RGB-only scale remains unknown.',
                           'Sparse keypoints only; no dense depth map or full surface reconstructed.',
                           'TUM RGB and depth are near-time, not exactly simultaneous.',
                           'Sensor claims are declared input provenance, not externally certified.']}


def main(argv: list[str]|None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory',type=Path,required=True)
    parser.add_argument('--colmap-model',type=Path,required=True)
    parser.add_argument('--depth-root',type=Path,required=True)
    parser.add_argument('--depth-spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output exists; refusing to overwrite')
    try:
        result=calibrate(args.trajectory,args.colmap_model,args.depth_root,args.depth_spec)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps({'result':'metric_scale_from_external_sensor',
                          'scale_m_per_sfm_unit':result['scale_m_per_sfm_unit'],
                          'anchors':3,'registered_frames':len(result['frames'])}))
        return 0
    except (DataError,OSError,ValueError,KeyError,TypeError) as e:
        import sys
        print(json.dumps({'error':str(e),'type':type(e).__name__}),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
