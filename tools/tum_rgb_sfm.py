"""Real RGB-only COLMAP SfM from TUM sequence encoded as a public test MP4.

GT is loaded ONLY after the external SfM mapper completes. No measured depth is
passed to the solver. Sim(3) fit reports *evaluation-only* ATE, not metric scale.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from echo4d.model import DataError, file_sha256, safe_path
from echo4d.sfm_eval import evaluate_sfm_poses
from echo4d.tum import read_rows, interpolate_pose
from tum_public_roundtrip import download, extract, invoke, URL, ROOT, PINNED_ARCHIVE_SHA256

FRAME_COUNT = 90
SOURCE_STRIDE = 8
VIDEO_FPS = 4


def prepare_video(tum_root: Path, work: Path, report: dict) -> list[dict]:
    index=read_rows(tum_root/'rgb.txt',2)
    selected=index[::SOURCE_STRIDE][:FRAME_COUNT]
    if len(selected)!=FRAME_COUNT:
        raise DataError('TUM sequence does not contain expected selected RGB frames')
    images=work/'rgb-input'
    images.mkdir()
    source=[]
    for i,(ts,path) in enumerate(selected):
        source_path=safe_path(tum_root,path)
        dest=images/f'frame_{i:05d}.png'
        shutil.copyfile(source_path,dest)
        source.append({'name':dest.name,'t_tum_rgb_s':float(ts),
                       'original_tum_path':path,'source_rgb_sha256':file_sha256(dest)})
    mp4=work/'test-video-from-real-tum-rgb.mp4'
    invoke('encode-real-rgb-images-into-test-mp4',
           ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y',
            '-framerate',str(VIDEO_FPS),'-start_number','0',
            '-i',str(images/'frame_%05d.png'),'-frames:v',str(FRAME_COUNT),
            '-c:v','libx264','-preset','ultrafast','-crf','15','-pix_fmt','yuv420p',str(mp4)],report,timeout=180)
    report['test_video']={'synthetic_timebase':True,
                          'real_tum_images':True,
                          'physical_phone_video':False,
                          'frames':FRAME_COUNT,
                          'source_stride':SOURCE_STRIDE,
                          'encoded_fps':VIDEO_FPS,
                          'file_sha256':file_sha256(mp4),
                          'bytes':mp4.stat().st_size,
                          'source_rgb_manifest':source}
    return source


def gt_after_sfm(tum_root: Path, source: list[dict]) -> tuple[dict,list]:
    """DO NOT CALL until actual COLMAP feature/matching/mapper has completed."""
    rows=read_rows(tum_root/'groundtruth.txt',8)
    gt=np.array(rows,dtype=np.float64)
    poses=Rotation.from_quat(gt[:,4:8])
    truths={}
    excluded=[]
    for entry in source:
        pose=interpolate_pose(gt[:,0],gt[:,1:4],poses,entry['t_tum_rgb_s'],max_gap=.1)
        if pose is not None:
            truths[entry['name']]=pose.flatten(order='F').tolist()
        else:
            excluded.append(entry['name'])
    if len(truths)<30:
        raise DataError('Insufficient independently timestamp-matched TUM GT poses')
    return truths,excluded


def experiment(args: argparse.Namespace, receipt: dict) -> None:
    work,report=args.work.resolve(),args.report_dir.resolve()
    work.mkdir(parents=True,exist_ok=False)
    archive=work/'source.tgz'
    download(archive,receipt)
    receipt['dataset']={'sha256':file_sha256(archive),'official_url':URL}
    receipt['extraction']=extract(archive,work/'dataset')
    tum=work/'dataset'/ROOT
    frames=prepare_video(tum,work,receipt)
    sfm=work/'sfm-out'
    cmd=[sys.executable,'-m','echo4d.video_sfm',
         '--video',str(work/'test-video-from-real-tum-rgb.mp4'),
         '--output',str(sfm),'--fps',str(VIDEO_FPS),
         '--max-frames',str(FRAME_COUNT),
         '--camera-model','PINHOLE',
         '--camera-params','525,525,319.5,239.5']
    # Entire image/video reconstruction runs before the TUM pose file is accessed.
    invoke('run-real-colmap-rgb-only-SfM',cmd,receipt,timeout=1800)
    sfm_report=json.loads((sfm/'report.json').read_text())
    if not sfm_report['status'].startswith('passed'):
        raise DataError('COLMAP workflow did not return a real registered camera track')
    receipt['sfm'] = sfm_report
    sparse=json.loads((sfm/'trajectory-scale-unknown.json').read_text())
    if sparse['metric_scale_estimated'] is not False or sparse['units']!='unknown':
        raise DataError('Uncalibrated RGB-only SfM cannot produce metric coordinate claims')
    if receipt['sfm']['result']['frames_extracted']!=FRAME_COUNT:
        raise DataError('Output frame selection differs from pre-registered count')
    gt,excluded=gt_after_sfm(tum,frames)
    receipt['groundtruth_for_evaluation_only']={
        'groundtruth_txt_sha256':file_sha256(tum/'groundtruth.txt'),
        'matched_with_TUM_timestamps':len(gt),'not_covered_by_GT':excluded,
        'scope':'Read after COLMAP solver completed; Sim3 fit is evaluation-only.'}
    receipt['pose_evaluation']=evaluate_sfm_poses(sparse,gt)
    n=receipt['sfm']['result']['frames_registered']
    receipt['registration_gate']={'min_registered_images':10,
                                  'min_fraction_of_extracted':0.35,
                                  'observed_images':n,
                                  'observed_fraction':n/FRAME_COUNT,
                                  'passes':n>=10 and n/FRAME_COUNT>=.35}
    if not receipt['registration_gate']['passes']:
        raise DataError('Measured real SfM registration rate too low for baseline')
    for name in ('trajectory-scale-unknown.json', 'registered_timestamps.csv'):
        shutil.copyfile(sfm/name,report/name)
    textpath=sfm/receipt['sfm']['result']['sfm_model_dir']
    for name in ('cameras.txt','images.txt','points3D.txt'):
        p=textpath/name
        if p.is_file() and p.stat().st_size < 8*1024**2:
            shutil.copyfile(p,report/name)
    receipt['artifact_sha256']={p.name:file_sha256(p) for p in report.iterdir() if p.is_file()}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    args=p.parse_args()
    args.report_dir.mkdir(parents=True,exist_ok=True)
    r={'experiment':'TUM-fr1_xyz-real-RGB-video-to-COLMAP-SfM-v1',
       'started_utc':datetime.now(timezone.utc).isoformat(),
       'git_sha':os.environ.get('ECHO_HEAD_SHA',os.environ.get('GITHUB_SHA')),
       'actions_url':f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}" if os.environ.get('GITHUB_RUN_ID') else None,
       'status':'running', 'commands':[],
       'quality_boundaries':['Real RGB images encoded as a NEW synthetic-timing MP4; not a native phone MP4.',
           'SfM gets RGB only and known camera intrinsics, NOT depth or TUM GT poses.',
           'COLMAP coordinate system has unknown metric scale even if Sim3 evaluator yields a value.',
           'GT is read after SfM; fitted/validated on same poses: optimistic in-sample ATE.',
           'No dense depth, metric mesh, scene reconstruction or person tracking claimed.']}
    start=time.perf_counter()
    try:
        experiment(args,r)
        r['status']='passed_rgb_only_sfm_with_posthoc_evaluation'
        code=0
    except Exception as e:
        r['status']='failed'
        r['error']={'type':type(e).__name__,'message':str(e)}
        print(f'RGB_SFM_EXPERIMENT_FAILED: {type(e).__name__}: {e}',file=sys.stderr)
        code=1
    finally:
        r['elapsed_seconds']=round(time.perf_counter()-start,3)
        (args.report_dir/'report.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        sfm=r.get('sfm',{}).get('result',{})
        ev=r.get('pose_evaluation',{})
        summary=['# RGB-only video to real COLMAP SfM', '',
                 f"- **Status:** {r['status']}",
                 f"- [GitHub run]({r['actions_url']})",
                 f"- Commit: {r['git_sha']}",
                 f"- TUM archive SHA: {r.get('dataset',{}).get('sha256','missing')}",
                 f"- Extracted/registered RGB frames: {sfm.get('frames_extracted','?')}/{sfm.get('frames_registered','?')}",
                 f"- Sparse 3D points: {sfm.get('sparse_points','?')}",
                 f"- ATE RMSE (evaluation-only Sim3; m): {ev.get('ate_rmse_m_after_sim3','not measured')}",
                 f"- Median rotation error (eval-only; degrees): {ev.get('rotation_median_degrees_after_sim3','not measured')}",
                 f"- **Metric scale recovered from RGB:** NO",
                 f"- Failure: {r.get('error',{}).get('message','none')}", '',
                 'See report.json and camera trajectory artifact for detailed commands and evidence.']
        (args.report_dir/'SUMMARY.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    return code


if __name__=='__main__':
    raise SystemExit(main())