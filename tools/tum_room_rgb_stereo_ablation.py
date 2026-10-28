"""Real TUM fr1_room: known-pose RGB stereo vs the frozen measured-depth room Mesh.

The existing TUM importer associates RGB, sensor timestamps and known poses,
but sensor depth arrays NEVER enter stereo or candidate TSDF. The high-info
Mesh is read only after the candidate Mesh exists, for frozen 3+2 comparison.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

from echo4d.holdout import read_echo_ply
from echo4d.model import Camera, DataError, file_sha256, load_recording, safe_path
from sparse_open3d_fusion import fuse_sparse
from tum_highres_tsdf import face_components, invoke
from tum_graph_visual import rasterize_vertex_rgb, read_vertex_rgb
from tum_room_sparse_quality import ROOT, get_archive, extract_rgbd, select_training, MAX_DEPTH_M
from tum_sparse_training_density import sheet

MAX_ANCHORS = 48
MAX_DISPARITY = 128


def choose_stereo_partner(poses: list[np.ndarray], i: int,
                          *, min_baseline: float = .045, max_baseline: float = .35) -> int | None:
    """Select a distinct right-hand camera with real horizontal parallax."""
    left = poses[i]
    candidates = []
    for j, right in enumerate(poses):
        if i == j:
            continue
        relative = np.linalg.inv(right) @ left
        baseline = float(np.linalg.norm(relative[:3, 3]))
        if not min_baseline <= baseline <= max_baseline:
            continue
        t = relative[:3, 3]
        if t[0] >= -.025 or abs(t[0]) < 1.3 * abs(t[1]):
            continue
        forward_cos = float(left[:3, 2] @ right[:3, 2])
        if forward_cos < math.cos(math.radians(16)):
            continue
        score = abs(baseline - .12) + .2 * (1 - forward_cos) + .001 * abs(j-i)
        candidates.append((score, j))
    return min(candidates)[1] if candidates else None


def infer_rectified_stereo_depth(left_rgb: np.ndarray, right_rgb: np.ndarray,
                                 camera: Camera, left_pose: np.ndarray,
                                 right_pose: np.ndarray) -> tuple[np.ndarray, dict]:
    """Calibrated SGBM -> 3D -> original-left camera Z; missing stays unknown."""
    h, w = left_rgb.shape[:2]
    if (left_rgb.shape != right_rgb.shape or left_rgb.shape != (camera.height, camera.width, 3)
            or left_rgb.dtype != np.uint8 or right_rgb.dtype != np.uint8
            or w < 160 or h < 96 or w > 1280 or h > 960):
        raise DataError("Stereo requires same calibrated uint8 RGB size")
    p0, p1 = np.asarray(left_pose, dtype=np.float64), np.asarray(right_pose, dtype=np.float64)
    if (p0.shape != (4,4) or p1.shape != (4,4) or not np.isfinite(p0).all()
            or not np.isfinite(p1).all()):
        raise DataError("Stereo requires finite camera-to-world poses")
    relative = np.linalg.inv(p1) @ p0
    R, T = relative[:3, :3], relative[:3, 3]
    if not (.045 <= np.linalg.norm(T) <= .35 and T[0] < -.025):
        raise DataError("No bounded positive-disparity stereo baseline")
    K = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy],
                  [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5)
    R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(
        K, D, K, D, (w,h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if P1[0,3] >= -1e-6 or abs(P1[0,3]) < abs(P1[1,3]):
        raise DataError("Stereo rectified vertically or reversed")
    x0, y0 = cv2.initUndistortRectifyMap(K, D, R0, P0, (w,h), cv2.CV_32FC1)
    x1, y1 = cv2.initUndistortRectifyMap(K, D, R1, P1, (w,h), cv2.CV_32FC1)
    a = cv2.remap(cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY), x0, y0, cv2.INTER_LINEAR)
    b = cv2.remap(cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY), x1, y1, cv2.INTER_LINEAR)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=MAX_DISPARITY, blockSize=5,
        P1=8*25, P2=32*25, disp12MaxDiff=1, uniquenessRatio=12,
        speckleWindowSize=80, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = stereo.compute(a,b).astype(np.float32) / 16.
    valid = np.isfinite(disparity) & (disparity > 1.)
    xyz = cv2.reprojectImageTo3D(disparity, Q)
    z = (xyz @ R0)[:,:,2]
    valid &= np.isfinite(z) & (z >= .2) & (z <= MAX_DEPTH_M)
    u = np.rint(x0[valid]).astype(np.int32)
    v = np.rint(y0[valid]).astype(np.int32)
    zz = z[valid]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    depth = np.full((h,w), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v[inside],u[inside]), zz[inside])
    depth[~np.isfinite(depth)] = 0
    return depth, {
        "valid_rectified_pixels": int(valid.sum()),
        "valid_original_pixels": int(np.count_nonzero(depth)),
        "baseline_m": float(np.linalg.norm(T)),
        "depth_source": "RGB SGBM disparity, not sensor Z",
        "metric_scale_source": "known TUM metric camera baseline"}


def build_rgb_stereo_manifest(imported: Path, target: Path, report: dict) -> Path:
    """Re-use exactly the 96 real high-info training identities, not their Z."""
    data = json.loads((imported/'recording.json').read_text())
    _, selected, receipt = select_training(data)
    report['selection'] = receipt
    camera = Camera.from_dict(data['intrinsics']).scaled(2)
    if (camera.width, camera.height) != (320,240):
        raise DataError("Expected calibrated 640x480 RGB downsampled to 320x240")
    frames = selected['frames']
    poses = [np.asarray(f['cameraToWorld'], dtype=np.float64).reshape(4,4,order='F')
             for f in frames]
    eligible = [(i,j) for i in range(len(frames))
                if (j := choose_stereo_partner(poses,i)) is not None]
    report['stereo_pair_selection'] = {
        'eligible_anchors':len(eligible), 'candidate_training_frames':len(frames),
        'missing_horizontal_baseline':len(frames)-len(eligible)}
    if len(eligible) < 8:
        raise DataError("Too few real horizontal stereo pairs; no fabricated depth")
    indices = np.linspace(0,len(eligible)-1,min(MAX_ANCHORS,len(eligible)))
    pairs = [eligible[int(round(k))] for k in indices]
    target.mkdir(parents=True,exist_ok=False)
    (target/'color').mkdir()
    (target/'depth').mkdir()
    cache = {}
    def rgb(i):
        if i not in cache:
            path = safe_path(imported,frames[i]['color'])
            with Image.open(path) as im:
                if im.size != (640,480):
                    raise DataError("Expected 640x480 recorded RGB")
                cache[i] = np.asarray(im.convert('RGB').resize((320,240),Image.Resampling.BILINEAR))
        return cache[i]
    output, receipts = [], []
    for i,j in pairs:
        try:
            z, detail = infer_rectified_stereo_depth(rgb(i),rgb(j),camera,poses[i],poses[j])
        except (DataError,cv2.error) as exc:
            receipts.append({'anchor':i,'partner':j,'status':'failed','reason':str(exc)})
            continue
        receipts.append({'anchor':i,'partner':j,'status':'inferred',**detail})
        if detail['valid_original_pixels'] < 150:
            continue
        k = len(output)
        Image.fromarray(rgb(i)).save(target/'color'/f'{k:03d}.png')
        np.save(target/'depth'/f'{k:03d}.npy',z,allow_pickle=False)
        output.append({'t':float(frames[i]['t']-frames[0]['t']),
                       'cameraToWorld':frames[i]['cameraToWorld'],
                       'color':f'color/{k:03d}.png','depth':f'depth/{k:03d}.npy'})
    report['stereo_pair_results'] = receipts
    report['inferred_depth_frames'] = len(output)
    if len(output) < 8:
        raise DataError("Stereo produced fewer than 8 usable real depth frames")
    recording = {'version':'echo-dataset-1','source':'recorded','units':'meters',
                 'name':'TUM room RGB known-pose stereo inferred Z',
                 'duration':max(.001,float(output[-1]['t'])),
                 'depth_scale':1,'intrinsics':camera.to_dict(),'frames':output,
                 'notes':'Z inferred from RGB SGBM and known metric poses; sensor Z absent from TSDF; missing stays unknown'}
    manifest = target/'recording.json'
    manifest.write_text(json.dumps(recording,indent=2,allow_nan=False)+'\n')
    return manifest


def correct_inferred_mesh_provenance(folder: Path) -> dict:
    """Correct measured-only provider's output label, without changing geometry."""
    mesh = folder/'mesh.ply'
    original = b'comment Open3D sparse TSDF; measured RGB-D; known poses; no inferred depth'
    replacement = b'comment Open3D sparse TSDF; RGB-stereo inferred depth; known metric poses; no sensor Z'
    with mesh.open('rb') as stream:
        header = stream.read(4096)
    if header.count(original) != 1 or b'end_header\n' not in header:
        raise DataError("Cannot identify measured-only PLY provenance comment")
    data = mesh.read_bytes()
    if data.count(original) != 1:
        raise DataError("Unexpected binary PLY provenance occurrence")
    mesh.write_bytes(data.replace(original,replacement,1))
    path = folder/'report.json'
    receipt = json.loads(path.read_text())
    receipt['geometry_source'] = 'RGB stereo disparity Z; known TUM metric poses; no measured Z'
    receipt['mesh_sha256'] = file_sha256(mesh)
    receipt['depth_provenance_correction'] = 'Header and report only; geometry unchanged'
    path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt


def compare_five(imported: Path, baseline: Path, candidate: Path,
                 high: dict, target: Path) -> list[dict]:
    # Sensor depth may be opened only after the candidate Mesh is completed.
    original = load_recording(imported/'recording.json')
    assoc = json.loads((imported/'association.json').read_text())
    meshes = []
    for path in (baseline,candidate):
        v,f = read_echo_ply(path)
        meshes.append((v,f,read_vertex_rgb(path,len(v))))
    cam = original.camera.scaled(2)
    target.mkdir(parents=True,exist_ok=False)
    results = []
    for item in high['visual']:
        if item['kind'] == 'heldout_camera':
            idx = item['index']
            pose = np.asarray(original.entries[idx]['cameraToWorld']).reshape(4,4,order='F')
            source = safe_path(imported.parent/'source'/ROOT,assoc['frames'][idx]['rgb_path'])
            if file_sha256(source) != assoc['frames'][idx]['rgb_sha256']:
                raise DataError("Original room RGB checksum changed")
            with Image.open(source) as im:
                raw = im.convert('RGB').copy()
        else:
            pose = np.asarray(item['camera_to_world']).reshape(4,4,order='F')
            raw = None
        renders = [rasterize_vertex_rgb(*mesh,cam,pose,near=.15,far=10.) for mesh in meshes]
        filename = ('heldout-'+str(idx) if raw is not None else 'free-'+str(len(results)))+'.png'
        sheet(raw,renders[0][0],renders[1][0],target/filename,
              'TUM real room RGB-D vs RGB stereo inferred-Z',
              ('measured depth / no prior Mesh','RGB-inferred depth / no prior Mesh'))
        results.append({'kind':item['kind'],'filename':filename,
                        'sha256':file_sha256(target/filename),
                        'baseline_covered_pixels':renders[0][1],
                        'candidate_covered_pixels':renders[1][1],
                        'camera_to_world':pose.flatten(order='F').tolist()})
    return results


def run(args, report):
    work = args.work.resolve()
    work.mkdir(parents=True,exist_ok=False)
    archive = work/'room.tgz'
    get_archive(archive,report,args.archive.resolve() if args.archive else None)
    report['source_archive_sha256'] = file_sha256(archive)
    report['extraction'] = extract_rgbd(archive,work/'source')
    imported = work/'room-import'
    invoke('import-source-associations', [
        sys.executable,'-m','echo4d','import-tum',str(work/'source'/ROOT),
        '--output',str(imported),'--intrinsics','525','525','319.5','239.5',
        '--stride','8','--limit','180','--downsample','1','--world-up','z'],report,timeout=900)
    high = json.loads((args.baseline/'room-sparse-quality-report.json').read_text())
    if (high.get('archive_observed_sha256') != report['source_archive_sha256']
            or high.get('status','').startswith('failed')
            or 'room-96-fine-wide' not in high.get('results',{})):
        raise DataError("Frozen measured baseline is not from same real source")
    baseline_mesh = args.baseline/'room-96-fine-wide'/'mesh.ply'
    if file_sha256(baseline_mesh) != high['results']['room-96-fine-wide']['mesh_sha256']:
        raise DataError("Frozen high-info Mesh SHA mismatch")
    manifest = build_rgb_stereo_manifest(imported,work/'inferred',report)
    if report['selection']['expanded_indices'] != high['training_selection']['expanded_indices']:
        raise DataError("RGB-only training identities differ from measured baseline")
    report['inferred_manifest_sha256'] = file_sha256(manifest)
    fuse_sparse(manifest,args.report_dir/'rgb-stereo-mesh',
                voxel_m=.04,truncation_m=.12,max_depth_m=4.,pixel_stride=2)
    report['fusion'] = correct_inferred_mesh_provenance(args.report_dir/'rgb-stereo-mesh')
    candidate = args.report_dir/'rgb-stereo-mesh'/'mesh.ply'
    v,f = read_echo_ply(candidate)
    report['new_mesh']={'sha256':file_sha256(candidate),'vertices':len(v),
                        'faces':len(f),'connectivity':face_components(len(v),f)}
    report['baseline_mesh']={'sha256':file_sha256(baseline_mesh),
                             'source':'GitHub Actions high-info room artifact, comparison only'}
    report['visual'] = compare_five(imported,baseline_mesh,candidate,high,args.report_dir/'visual')
    import trimesh
    for name,path in (('highinfo-measured',baseline_mesh),('rgb-stereo-inferred',candidate)):
        vv,ff = read_echo_ply(path)
        cc = read_vertex_rgb(path,len(vv))
        tm = trimesh.Trimesh(vertices=vv,faces=ff,vertex_colors=cc,process=False)
        tm.export(args.report_dir/(name+'.glb'))
    report['status']='real_room_mesh_and_five_views_generated_visual_FAIL_pending_review'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--archive',type=Path)
    args=p.parse_args()
    args.report_dir.mkdir(parents=True,exist_ok=True)
    report={'experiment':'tum-fr1-room-known-pose-rgb-stereo-v1',
            'started_utc':datetime.now(timezone.utc).isoformat(),
            'checkout_sha':os.environ.get('GITHUB_SHA'),
            'retained':['real 640x480 RGB (320x240 stereo)','known metric poses',
                        'intrinsics','96 high-info matched training frame identities'],
            'removed_from_reconstruction':['sensor depth pixels','prior Mesh'],
            'source_archive_contains_sensor_depth':True,
            'importer_uses_sensor_depth_for_timestamp_association':True,
            'measured_depth_used_for_stereo_or_tsdf':False,
            'baseline_used_for_inference':False,
            'metric_scale':'known TUM motion-capture translation in metres',
            'status':'started','issue_74':'FAIL / NOT ACCEPTED'}
    t=time.perf_counter()
    try:
        run(args,report)
        code=0
    except Exception as exc:
        report['status']='failed'
        report['error']={'type':type(exc).__name__,'message':str(exc)}
        code=1
    finally:
        report['elapsed_seconds']=round(time.perf_counter()-t,2)
        (args.report_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':report['status'],'error':report.get('error'),
                          'new_mesh':report.get('new_mesh')},ensure_ascii=False))
    return code


if __name__=='__main__':
    raise SystemExit(main())
