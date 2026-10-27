"""Actual TUM RGB-only COLMAP SfM + *three sensor-depth* scale anchors.

TUM groundtruth.txt is forbidden until the scale and heldout depth residuals
have been computed and saved. This is a sensor-assisted measurement, NOT
monocular RGB metrification or automatically predicted dense depth.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from echo4d.metric_anchor import calibrate
from echo4d.model import DataError, file_sha256, rigid_matrix
from echo4d.sfm_eval import evaluate_sfm_poses
from echo4d.tum import read_rows
from tum_public_roundtrip import download, extract, URL, ROOT, PINNED_ARCHIVE_SHA256
from tum_rgb_sfm import FRAME_COUNT, VIDEO_FPS, prepare_video, gt_after_sfm, invoke

ANCHORS=(15,45,75)
HOLDOUTS=(5,10,25,35,55,65,85)
MAX_GAP_S=.03


def measured_spec(tum_root:Path, source:list[dict], report:dict) -> dict:
    """This may be called only after the SFm run and its verified track exist."""
    depth_index=tum_root/'depth.txt'
    rows=read_rows(depth_index,2)
    times=np.array([float(row[0]) for row in rows])
    output={'source_kind':'measured_registered_z_depth',
            'source_description':'TUM Freiburg1 XYZ real registered RGB-D depth sensor (1/5000m PNG); externally measured, not RGB inferred',
            'source_dataset_url':URL,
            'depth_index_sha256':file_sha256(depth_index),
            'depth_numeric_units_per_meter':5000,
            'sensor_timestamps_are_rgb-nearest_not_simultaneous':True,
            'anchors':[],'holdouts':[]}
    for role,indices in (('anchors',ANCHORS),('holdouts',HOLDOUTS)):
        for index in indices:
            entry=source[index]
            rgb_time=entry['t_tum_rgb_s']
            at=int(np.searchsorted(times,rgb_time))
            candidates=[i for i in (at-1,at) if 0<=i<len(times)]
            near=min(candidates,key=lambda i:abs(times[i]-rgb_time))
            gap=float(times[near]-rgb_time)
            if abs(gap)>MAX_GAP_S:
                raise DataError(f'TUM measured depth unavailable within 30ms of RGB {index}: {gap:.3f}s')
            output[role].append({'name':entry['name'],
                                 'depth':rows[near][1],
                                 'timestamp_delta_seconds':gap,
                                 'source_rgb_time_s':rgb_time,
                                 'source_depth_time_s':float(times[near]),
                                 'description':'Observed by registered TUM depth sensor; not used by RGB SfM'})
    all_paths=[r['depth'] for r in output['anchors']+output['holdouts']]
    if len(set(all_paths))!=len(all_paths):
        raise DataError('Sensor depth samples for anchor/holdout are not independent files')
    report['sensor_depth_manifest']={
        'calibration_indices':ANCHORS,'heldout_indices':HOLDOUTS,
        'total_measurement_frames':len(all_paths),'max_rgb_depth_offset_s':max(
            abs(r['timestamp_delta_seconds']) for r in output['anchors']+output['holdouts']),
        'measured_depth_source':output['source_description'],
        'source_depth_index_sha256':output['depth_index_sha256']}
    return output


def metric_pose_check_using_one_gt_for_world_gauge(metric:dict,gt:dict[str,list[float]],
                                                    origin_anchor:str) -> dict:
    """Metric scale is locked: GT is used only for a *rigid* evaluation frame gauge.

    Anchoring one estimated camera pose in GT fixes arbitrary world origin/axes.
    No Sim(3) scaling of the predicted camera centers is permitted here.
    """
    frames={entry['name']:entry for entry in metric['frames']}
    if origin_anchor not in frames or origin_anchor not in gt:
        raise DataError('World-frame gauge anchor must have externally held-out GT pose')
    reference_anchor=rigid_matrix(gt[origin_anchor])
    prediction_anchor=rigid_matrix(frames[origin_anchor]['camera_to_metric_sfm_world'])
    transform=reference_anchor@np.linalg.inv(prediction_anchor)
    if not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-6):
        raise DataError('Gauge evaluation unexpectedly contains scale/shear')
    distances=[];angles=[]
    for name,entry in frames.items():
        if name==origin_anchor or name not in gt:
            continue
        pred=transform@rigid_matrix(entry['camera_to_metric_sfm_world'])
        reference=rigid_matrix(gt[name])
        distances.append(float(np.linalg.norm(pred[:3,3]-reference[:3,3])))
        delta=reference[:3,:3].T@pred[:3,:3]
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1)))))
    if len(distances)<20:
        raise DataError('Not enough GT-paired poses for heldout metric check')
    return {'evaluation_scope':'Only after sensor-scale solution is locked',
            'gt_pose_used_for_origin_and_orientation_only':origin_anchor,
            'gt_was_used_for_scale':False,
            'rigid_se3_no_scaling':True,'evaluated_nonanchor_frames':len(distances),
            'heldout_position_rmse_m':float(np.sqrt(np.mean(np.square(distances)))),
            'heldout_position_median_m':float(np.median(distances)),
            'heldout_position_p95_m':float(np.percentile(distances,95)),
            'heldout_rotation_median_degrees':float(np.median(angles)),
            'heldout_rotation_p95_degrees':float(np.percentile(angles,95))}


def experiment(args:argparse.Namespace,receipt:dict) -> None:
    work,report=args.work.resolve(),args.report_dir.resolve()
    work.mkdir(parents=True,exist_ok=False)
    archive=work/'source.tgz'
    download(archive,receipt)
    if file_sha256(archive)!=PINNED_ARCHIVE_SHA256:
        raise DataError('TUM source SHA differs from pinned official archive')
    receipt['dataset']={'url':URL,'sha256':file_sha256(archive),
                        'bytes':archive.stat().st_size}
    receipt['extraction']=extract(archive,work/'unpacked')
    tum=work/'unpacked'/ROOT
    source=prepare_video(tum,work,receipt)
    sfm=work/'rgb-only-colmap'
    cmd=[sys.executable,'-m','echo4d.video_sfm',
         '--video',str(work/'test-video-from-real-tum-rgb.mp4'),
         '--output',str(sfm),'--fps',str(VIDEO_FPS),
         '--max-frames',str(FRAME_COUNT),
         '--camera-model','PINHOLE',
         '--camera-params','525,525,319.5,239.5']
    try:
        invoke('estimate-camera-poses-from-RGB-ONLY',cmd,receipt,timeout=1800)
    except RuntimeError as failed:
        nested=sfm/'report.json'
        if nested.is_file():
            receipt['failed_rgb_sfm']=json.loads(nested.read_text())
        raise DataError('Real RGB-only COLMAP stage failed; see nested report') from failed
    sfm_receipt=json.loads((sfm/'report.json').read_text())
    if not sfm_receipt['status'].startswith('passed'):
        raise DataError('COLMAP did not produce a complete registered camera track')
    receipt['rgb_sfm_result']=sfm_receipt['result']
    trajectory=sfm/'trajectory-scale-unknown.json'
    track=json.loads(trajectory.read_text())
    if track.get('units')!='unknown' or track.get('metric_scale_estimated') is not False:
        raise DataError('RGB-only model is secretly pre-scaled')
    if sfm_receipt['result']['frames_registered']!=FRAME_COUNT:
        raise DataError('Unregistered anchor views; no replacing with GT poses')
    model=sfm/sfm_receipt['result']['sfm_model_dir']

    # FIRST SENSOR ACCESS: SfM is finished. This selects 3 independent measured
    # scale anchor depth maps and 7 holdout depth maps by raw source timestamps.
    measured=measured_spec(tum,source,receipt)
    depth_spec=work/'measured-depth-evidence.json'
    depth_spec.write_text(json.dumps(measured,indent=2,ensure_ascii=False)+'\n')
    metric=calibrate(trajectory,model,tum,depth_spec)
    output=report/'metric-trajectory-from-sensor.json'
    output.write_text(json.dumps(metric,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    receipt['metric_sensor_anchor']={
        'scale_m_per_sfm_unit':metric['scale_m_per_sfm_unit'],
        'anchors':metric['anchor_diagnostics'],
        'heldout':metric['heldout_depth_diagnostics'],
        'metric_output_sha256':file_sha256(output),
        'was_gt_read_to_estimate_scale':False}
    shutil.copyfile(depth_spec,report/'measured-depth-evidence.json')
    shutil.copyfile(trajectory,report/'trajectory-scale-unknown.json')
    # NOW (and only now) open GT camera poses; do not change metric scale.
    groundtruth,excluded=gt_after_sfm(tum,source)
    receipt['gt_loaded_only_after_metric_scale_was_locked']=True
    receipt['gt_groundtruth_txt_sha256']=file_sha256(tum/'groundtruth.txt')
    receipt['gt_missing_frame_names']=excluded
    anchor_name=source[ANCHORS[0]]['name']
    receipt['metric_se3_pose_evaluation']=metric_pose_check_using_one_gt_for_world_gauge(
        metric,groundtruth,anchor_name)
    # Separate posthoc reference scale is informational; never changes metric output.
    ev=evaluate_sfm_poses(track,groundtruth)
    receipt['reference_scale_for_audit_only']={
        'posthoc_gt_sim3_scale_m_per_unit':ev['evaluation_only_sim3_scale_m_per_sfm_unit'],
        'sensor_scale_m_per_unit':metric['scale_m_per_sfm_unit'],
        'relative_difference_sensor_vs_gt_sim3':metric['scale_m_per_sfm_unit']/
                          ev['evaluation_only_sim3_scale_m_per_sfm_unit']-1,
        'gt_sim3_never_used_in_calibration':True}
    receipt['report_hashes']={p.name:file_sha256(p) for p in report.iterdir() if p.is_file()}
    receipt['claimed_capability']='measured-depth-assisted metric trajectory; no estimated dense depth or mesh'


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args()
    report=args.report_dir.resolve()
    report.mkdir(parents=True,exist_ok=True)
    receipt={'experiment':'tum-rgb-sfm-plus-three-real-sensor-depth-anchors-v1',
             'started_utc':datetime.now(timezone.utc).isoformat(),
             'github_run_url':(f'https://github.com/{os.environ.get("GITHUB_REPOSITORY")}/actions/runs/{os.environ.get("GITHUB_RUN_ID")}'
                               if os.environ.get('GITHUB_RUN_ID') else None),
             'git_sha':os.environ.get('ECHO_HEAD_SHA',os.environ.get('GITHUB_SHA')),
             'status':'running','commands':[],
             'pre_registered_calibration_indices':ANCHORS,
             'pre_registered_heldout_indices':HOLDOUTS,
             'algorithmic_limits':['SfM from RGB and known intrinsics only',
               'Source RGB was rebundled as test MP4, not recorded natively as phone video',
               'Scale from three external measured registered TUM RGB-D sensor depth frames',
               'Holdout depth does not set scale',
               'GT camera poses read only after scale locked, used for one-frame SE3 world gauge and evaluation',
               'Sparse SfM points only; no dense 3D TSDF mesh',
               'Provenance claims not independently authenticated by core adapter']}
    start=time.perf_counter();code=0
    try:
        experiment(args,receipt)
        receipt['status']='passed_sensor_metric_calibration_not_rgb_only_metric'
    except Exception as e:
        receipt['status']='failed';code=1
        receipt['error']={'type':type(e).__name__,'message':str(e)}
        print('REAL_METRIC_ANCHOR_FAILED:',type(e).__name__,str(e),file=sys.stderr)
    finally:
        receipt['elapsed_seconds']=round(time.perf_counter()-start,3)
        (report/'report.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        metric=receipt.get('metric_sensor_anchor',{})
        gt=receipt.get('metric_se3_pose_evaluation',{})
        summary=[
            '# Real RGB SfM + independently measured sensor depth scale', '',
            f"- Status: **{receipt['status']}**", f"- [Actions Run]({receipt['github_run_url']})",
            f"- Pinned TUM SHA256: {receipt.get('dataset',{}).get('sha256','unavailable')}",
            f"- COLMAP actual RGB-only registered frames: {receipt.get('rgb_sfm_result',{}).get('frames_registered','not_done')}",
            f"- Measured-depth scale (m/SfM unit): {metric.get('scale_m_per_sfm_unit','not_done')}",
            f"- 7 non-calibration sensor depth checks: {len(metric.get('heldout',[]))}",
            f"- GT-independent-scale heldout SE3 position RMSE (m): {gt.get('heldout_position_rmse_m','not_done')}",
            f"- Error: {receipt.get('error',{}).get('message','none')}", '',
            '**Not monocular absolute scale; 3 external sensor depth anchors required.**',
            'Groundtruth.txt may only be read AFTER locking the measured-depth scale.',
            'See report.json and metric-trajectory-from-sensor.json for evidence and all frame values.']
        (report/'SUMMARY.md').write_text('\n'.join(summary)+'\n')
    return code


if __name__=='__main__':
    raise SystemExit(main())
