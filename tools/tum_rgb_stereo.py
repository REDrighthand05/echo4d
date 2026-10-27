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
from tum_public_roundtrip import URL, ROOT, PINNED_ARCHIVE_SHA256, PINNED_ARCHIVE_BYTES, BOUNDS, download, extract

HOLDOUT = (4, 5, 9, 10, 14, 19, 24, 25, 29, 34, 35, 55)
TOTAL = 60


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


def select_pairs(frames: list[dict], camera: Camera, max_pairs: int = 20) -> list[tuple[int, int]]:
    """Select fixed-order positive-disparity horizontal pairs, train-only."""
    training = [i for i in range(TOTAL) if i not in HOLDOUT]
    candidates = {}
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
                if left not in candidates or score < candidates[left][0]:
                    candidates[left] = (score, right)
                break
    chosen = sorted(candidates)
    if len(chosen) > max_pairs:
        selected = np.linspace(0, len(chosen)-1, max_pairs, dtype=int)
        chosen = [chosen[int(k)] for k in selected]
    return [(i, candidates[i][1]) for i in chosen]


def build_stereo_recording(frames: list[dict], target: Path, report: dict) -> Path:
    camera = Camera(320, 240, 262.5, 262.5, 159.75, 119.75)
    pairs = select_pairs(frames, camera)
    report['selected_pairs'] = pairs
    if len(pairs) < 6:
        raise DataError(f'Only {len(pairs)} eligible stereo pairs; require >=6')
    target.mkdir(parents=True, exist_ok=False)
    (target / 'color').mkdir()
    (target / 'depth').mkdir()
    entries, pair_evidence = [], []
    for left_idx, right_idx in pairs:
        left, right = frames[left_idx], frames[right_idx]
        with Image.open(left['rgb']) as im:
            a = np.asarray(im.convert('RGB'))[::2, ::2].copy()
        with Image.open(right['rgb']) as im:
            b = np.asarray(im.convert('RGB'))[::2, ::2].copy()
        rgb, depth, rect_camera, rect_pose, diagnostics = infer_stereo(
            a, b, camera, left['pose'], right['pose'])
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
        if np.count_nonzero(projected) < 100:
            continue
        n=len(entries)
        Image.fromarray(a).save(target / 'color' / f'{n:05d}.png')
        np.save(target / 'depth' / f'{n:05d}.npy',projected,allow_pickle=False)
        entries.append({'t':float(left['t_s']-frames[0]['t_s']),
                        'cameraToWorld':left['pose'].flatten(order='F').tolist(),
                        'color':f'color/{n:05d}.png','depth':f'depth/{n:05d}.npy'})
        item['saved_depth_sha256']=file_sha256(target / 'depth' / f'{n:05d}.npy')
        item['reprojected_valid_pixels']=int(np.count_nonzero(projected))
    report['pairs']=pair_evidence
    if len(entries)<6:
        raise DataError(f'Only {len(entries)} RGB-derived depth frames have >=100 valid pixels')
    entries.sort(key=lambda x:x['t'])
    data={'version':'echo-dataset-1','source':'recorded','units':'meters',
          'name':'TUM RGB stereo, known GT poses, inferred depths',
          'duration':float(max(entries[-1]['t'],.001)), 'depth_scale':1,
          'intrinsics':camera.to_dict(),'frames':entries,
          'notes':'Depth is RGB stereo-inferred, not sensor-measured. Metric scale comes exclusively from supplied TUM GT poses; depth index used only for frozen frame selection.'}
    manifest=target/'recording.json'
    manifest.write_text(json.dumps(data,allow_nan=False,indent=2)+'\n',encoding='utf8')
    report['rgb_stereo_manifest_sha256']=file_sha256(manifest)
    report['saved_frames']=len(entries)
    return manifest


def render_mesh(mesh: Path, frames: list[dict], folder: Path, report: dict) -> None:
    from echo4d.model import rigid_matrix
    verts, faces = read_echo_ply(mesh)
    camera=Camera(320,240,262.5,262.5,159.75,119.75)
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
    report['rgb_input']={'original_resolution':[640,480],'algorithm_resolution':[320,240],
                         'frame_count':len(frames),'train_count':48,'holdout_indices':list(HOLDOUT),
                         'rgb_source_sha256':[f['rgb_sha256'] for f in frames],
                         'depth_png_files_opened':0,'depth_index_used_for_frame_selection_only':True,
                         'metric_pose_source':'TUM supplied groundtruth.txt; not RGB-only pose estimation'}
    manifest=build_stereo_recording(frames,work/'stereo',report)
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
    render_mesh(target/'mesh.ply',frames,args.report_dir/'visual',report)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--archive',type=Path)
    args=p.parse_args(argv)
    args.report_dir.mkdir(parents=True,exist_ok=True)
    report={'experiment':'tum-xyz-known-pose-RGB-stereo-no-measured-depth-v1',
            'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
            'checkout_sha':os.environ.get('GITHUB_SHA'), 'source_url':URL,
            'input_removed':'measured depth maps from reconstruction',
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
