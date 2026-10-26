"""AB-V2-final: same 48 RGB/depth train frames, same 12 withheld Z views.

A uses only TUM known camera poses; B uses independently reconstructed SfM poses
scaled by three measured sensor-depth anchors. The original V1 benchmark is
NEVER overwritten: V2 source frame 25 has no timely GT and is excluded BEFORE
benchmarking, with source frame 60 substituted as a training frame.

This is measured RGB-D sensor depth, not inferred depth from RGB alone.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from echo4d.cli import reconstruct_command
from echo4d.metric_holdout import evaluate_sensor_metric_holdouts
from echo4d.model import DataError, file_sha256, load_recording, rigid_matrix
from echo4d.tum import read_rows
from tum_public_roundtrip import ROOT, PINNED_ARCHIVE_SHA256
from tum_rgb_sfm import gt_after_sfm
from tum_sensor_metric_mesh import (
    FAIR_AB_V2_SELECTED_INDICES, FAIR_AB_V2_HOLDOUT_INDICES,
    VOXEL_M, MAX_DEPTH_M, MAX_VOXELS, MIN_OBSERVATIONS, DOWNSAMPLE,
    depth_only_training_bounds, experiment as run_estimated_B,
)


def verify_v2_split(split: dict) -> tuple[list[str], list[str]]:
    """Must be the exact pre-registered V2; no cherry picking after A scores."""
    p = split.get('protocol', {})
    chosen = [f'frame_{i:05d}.png' for i in FAIR_AB_V2_SELECTED_INDICES]
    held = [f'frame_{i:05d}.png' for i in FAIR_AB_V2_HOLDOUT_INDICES]
    train = [x for x in chosen if x not in set(held)]
    observed_train = [x.get('name') for x in split.get('train', [])]
    observed_held = split.get('holdout')
    if (len(chosen) != 60 or len(train) != 48 or len(held) != 12
            or p.get('ab_version') != 'AB-V2-final'
            or list(p.get('selected_source_indices', [])) != list(FAIR_AB_V2_SELECTED_INDICES)
            or list(p.get('holdout_indices', [])) != list(FAIR_AB_V2_HOLDOUT_INDICES)
            or observed_train != train or observed_held != held
            or p.get('missing_gt_source_index_predeclared_excluded') != 25
            or 'frame_00025.png' in train + held):
        raise DataError('AB-V2-final 48/12 cohort differs from pre-registered protocol')
    return train, held


def strict_truth(groundtruth: dict, train: list[str], held: list[str]) -> dict:
    required = set(train + held)
    missing = sorted(required - set(groundtruth))
    if missing:
        raise DataError('A known camera poses unavailable for frozen views: ' + ', '.join(missing))
    if len(set(train + held)) != 60:
        raise DataError('A/B study train/holdout names overlap')
    for name in train + held:
        rigid_matrix(groundtruth[name])
    return {name:groundtruth[name] for name in held}


def make_known_pose_training(b_record: Path, b_provenance: Path, truth: dict,
                             train: list[str], output: Path) -> dict:
    """A and B have bit-identical training RGB and Z; ONLY camera poses differ."""
    manifest = json.loads(b_record.read_text(encoding='utf-8'))
    prov = json.loads(b_provenance.read_text(encoding='utf-8'))
    source_names = [x['name'] for x in prov['train']]
    if source_names != train or len(manifest['frames']) != 48:
        raise DataError('B mesh input manifest no longer matches pre-registered train order')
    if output.exists():
        raise DataError('Refuse to overwrite A reference recording')
    shutil.copytree(b_record.parent, output)
    new = json.loads((output / 'recording.json').read_text(encoding='utf-8'))
    checks=[]
    for frame, name in zip(new['frames'], train):
        pose=rigid_matrix(truth[name])
        frame['cameraToWorld']=pose.flatten(order='F').tolist()
        color=Path(frame['color'])
        depth=Path(frame['depth'])
        if color.is_absolute() or depth.is_absolute() or '..' in color.parts or '..' in depth.parts:
            raise DataError('A manifest contains unsafe input path')
        for rel in (color, depth):
            a=output / rel; b=b_record.parent / rel
            if not a.is_file() or not b.is_file() or file_sha256(a)!=file_sha256(b):
                raise DataError('A/B identical training image or measured depth invariant broken')
        checks.append({'name':name,'rgb_sha256':file_sha256(output/color),
                       'measured_z_sha256':file_sha256(output/depth)})
    new['name']='reference-known-pose-same-48-measured-depth'
    new['notes']='A reference: TUM known poses, SAME B RGB and sensor-Z bytes; GT not available to B solver.'
    (output/'recording.json').write_text(json.dumps(new, indent=2, allow_nan=False)+'\n',encoding='utf-8')
    for _ in load_recording(output/'recording.json').iter_frames():
        pass
    return {'frames':48,'training_data_sha256':checks,
            'a_manifest_sha256':file_sha256(output/'recording.json'),
            'b_manifest_sha256':file_sha256(b_record)}


def check_gate(name: str, metric: dict) -> dict:
    """Display all errors; scientific screening is not accuracy certification."""
    valid=int(metric['valid_measured_pixels'])
    overlap=int(metric['overlap_pixels'])
    coverage=float(metric['coverage'])
    rmse=float(metric['rmse_m'])
    if (int(metric['holdout_frames'])!=12 or valid<=0 or not 0<=overlap<=valid
            or not math.isfinite(coverage) or not math.isfinite(rmse) or rmse < 0
            or abs(coverage-overlap/valid)>1e-12):
        raise DataError(f'{name}: incomplete or inconsistent 12-view pixel accounting')
    return {'coverage_minimum':0.80,'rmse_maximum_m':0.15,
            'passed':bool(coverage>=.80 and rmse<=.15)}


def assert_same_holdout_inputs(a: dict, b: dict, names: list[str]) -> dict:
    """Verify source identity, not prediction equality, before claiming fair A/B.

    Adapted from PR #32's per-view identity guard to the reusable #25/#31
    report schema. Coverage may differ; source bytes and valid masks may not.
    """
    if len(names) != 12 or len(set(names)) != 12:
        raise DataError('A/B requires exactly twelve distinct frozen views')
    rows = []
    totals = []
    for label, report in (('A', a), ('B', b)):
        frames = report.get('per_frame')
        if (report.get('holdout_frames') != 12 or not isinstance(frames, list)
                or len(frames) != 12
                or any(not isinstance(row, dict) for row in frames)
                or [row.get('name') for row in frames] != names):
            raise DataError(f'{label}: missing, reordered or changed frozen views')
        validated = []
        for frame in frames:
            score = frame.get('score', {})
            if not isinstance(score, dict):
                raise DataError(f'{label}: missing per-view pixel accounting')
            count = score.get('sensor_valid_pixels')
            digest = frame.get('depth_sha256')
            filename = frame.get('depth_filename')
            stamp = frame.get('source_rgb_t_s')
            gap = frame.get('rgb_depth_delta_s')
            if (type(count) is not int or count <= 0
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)
                    or not isinstance(filename, str) or not filename
                    or isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                    or not math.isfinite(stamp)
                    or isinstance(gap, bool) or not isinstance(gap, (int, float))
                    or not math.isfinite(gap) or abs(gap) > .03):
                raise DataError(f'{label}: incomplete measured source identity for {frame["name"]}')
            validated.append({
                'name': frame['name'], 'depth_filename': filename,
                'depth_sha256': digest, 'sensor_valid_pixels': count,
                'source_rgb_t_s': stamp, 'rgb_depth_delta_s': gap,
            })
        if len({row['depth_filename'] for row in validated}) != 12:
            raise DataError(f'{label}: repeated heldout depth source')
        total = sum(row['sensor_valid_pixels'] for row in validated)
        if type(report.get('valid_measured_pixels')) is not int or report['valid_measured_pixels'] != total:
            raise DataError(f'{label}: per-view valid pixels disagree with aggregate')
        rows.append(validated)
        totals.append(total)
    if rows[0] != rows[1] or totals[0] != totals[1]:
        raise DataError('A/B heldout source hashes, timestamps or valid pixels differ')
    return {'verified': True, 'views': 12, 'valid_measured_pixels': totals[0],
            'per_frame': rows[0]}


def experiment(args: argparse.Namespace, receipt: dict) -> None:
    work=args.work.resolve();report=args.report_dir.resolve()
    report.mkdir(parents=True,exist_ok=True)
    b_receipt={'git_sha':receipt['git_sha']}
    # Crucially B is computed FIRST; this function does not read GT until B
    # has completed its own RGB-only SfM, depth anchoring, 48-frame mesh and 12
    # withheld-depth scoring. The prior #14 calibration evaluates GT for audit
    # internally but never supplies it to SfM, scale estimation, or B fusion.
    run_estimated_B(argparse.Namespace(work=work,report_dir=report,fair_ab_v2=True),b_receipt)
    receipt['b_result']={k:v for k,v in b_receipt.items() if k in (
        'metric_stage','dataset_conversion','split_sha256','input_manifest_sha256',
        'sensor_provenance_sha256','pre_registered_bounds','tsdf_config',
        'mesh_result','mesh_heldout','mesh_heldout_report_sha256')}
    receipt['b_stage_completed_before_A_GT_reference_opened']=True
    split=json.loads((report/'b-48train-12holdout.json').read_text())
    training,held=verify_v2_split(split)
    b_depth=report/'mesh-heldout-depth.json'
    b_hold=json.loads(b_depth.read_text(encoding='utf-8'))
    if b_hold['holdout_frames']!=12 or b_hold['gt_camera_poses_loaded'] is not False:
        raise DataError('B score no longer uses only estimated metric camera positions')
    metric=json.loads((report/'metric'/'metric-trajectory-from-sensor.json').read_text())
    metric_source=json.loads((report/'metric'/'report.json').read_text())
    source=metric_source['test_video']['source_rgb_manifest']
    tum=work/'unpacked'/ROOT
    if b_receipt['metric_stage']['source_sha256']!=PINNED_ARCHIVE_SHA256:
        raise DataError('AB dataset source hash mismatch')
    # No posthoc replacement of missing GT or hidden extra interpolation.
    truth,missing=gt_after_sfm(tum,source)
    receipt['gt_reference']={'groundtruth_sha256':file_sha256(tum/'groundtruth.txt'),
                             'missing_across_all_90':missing,
                             'pre_registered_excluded_from_AB':['frame_00025.png']}
    held_truth=strict_truth(truth,training,held)
    a_dir=work/'a-known-pose-training'
    inputs=make_known_pose_training(work/'b-sensor-recording'/'recording.json',
                                   work/'b-sensor-recording'/'sensor-provenance.json',
                                   truth,training,a_dir)
    receipt['identical_training_inputs']=inputs
    a_bounds=depth_only_training_bounds(a_dir/'recording.json')
    receipt['a_bounds_m_from_same_training_depth_only']=a_bounds
    out=report/'a-known-pose-mesh'
    a_result=reconstruct_command(argparse.Namespace(
        recording=a_dir/'recording.json',output=out,bounds=a_bounds,
        voxel=VOXEL_M,truncation=None,max_depth=MAX_DEPTH_M,max_voxels=MAX_VOXELS,
        min_observations=MIN_OBSERVATIONS,stride=1,limit=None,reference=None,
        threshold=.05,no_scene=False,save_volume=False,quiet=True))
    if a_result['frames']!=48 or a_result['vertices']<=0 or a_result['triangles']<=0:
        raise DataError('A 48-frame known-pose mesh failed to reconstruct')
    receipt['a_mesh_result']=a_result
    sfm=work/'rgb-only-colmap'
    sfm_info=json.loads((sfm/'report.json').read_text())
    colmap_model=sfm/sfm_info['result']['sfm_model_dir']
    # THE SAME evaluator scores A and B. The only difference is A's explicit
    # known-pose override, never provided to the B input or result.
    a_hold=evaluate_sensor_metric_holdouts(
        out/'mesh.ply',a_dir/'recording.json',metric,split,source,tum,
        colmap_model,downsample=DOWNSAMPLE,max_gap_s=.03,max_depth_m=MAX_DEPTH_M,
        known_poses_for_A=held_truth)
    receipt['heldout_input_identity'] = assert_same_holdout_inputs(a_hold, b_hold, held)
    (report/'a-known-pose-holdout.json').write_text(
        json.dumps(a_hold,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    receipt['a_heldout_report_sha256']=file_sha256(report/'a-known-pose-holdout.json')
    receipt['ab_comparison']={
        'version':'AB-V2-final','same_48_training_rgb_and_depth_files':True,
        'same_12_withheld_depth_views':True,
        'heldout_source_identity_verified':receipt['heldout_input_identity']['verified'],
        'split_sha256':file_sha256(report/'b-48train-12holdout.json'),
        'A_known_pose':{k:v for k,v in a_hold.items() if k not in ('per_frame','limitations')},
        'B_estimated_pose_sensor_scale':{k:v for k,v in b_hold.items() if k not in ('per_frame','limitations')},
        'A_gate':check_gate('A',a_hold),
        'B_gate':check_gate('B',b_hold),
        'interpretation':('Only one real TUM RGB-D sensor sequence, not an independent laser scan. '
                          'Difference combines pose, TSDF bounds and reconstruction randomness; '
                          'it cannot be uniquely assigned to pose error.'),
    }
    (report/'ab-comparison.json').write_text(
        json.dumps(receipt['ab_comparison'],indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    if not receipt['ab_comparison']['A_gate']['passed'] or not receipt['ab_comparison']['B_gate']['passed']:
        raise DataError('Pre-registered AB research screening failed; preserving raw metrics')


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    args=p.parse_args()
    args.report_dir.mkdir(parents=True,exist_ok=True)
    report={'version':'AB-V2-final','status':'running',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'git_sha':os.getenv('ECHO_TESTED_CHECKOUT_SHA') or os.getenv('GITHUB_SHA'),
            'tested_checkout_sha':os.getenv('ECHO_TESTED_CHECKOUT_SHA'),
            'pr_head_sha':os.getenv('ECHO_HEAD_SHA'),
            'base_sha':os.getenv('ECHO_BASE_SHA'),
            'run_url':f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}" if os.getenv('GITHUB_RUN_ID') else None,
            'purpose':'48 same real RGB+sensor-depth train, 12 same withheld-Z compare GT-pose A and SfM-pose B; never use GT for B',
            'limits':['First public AB trial on one TUM sequence only.',
                      'TUM GT frame 25 excluded from NEW V2 train and heldout before first trial.',
                      'Existing V1 48/12 reports are separate and must not be numerically compared to V2.',
                      'Depth comes from physical sensor, not RGB-only depth inference.',
                      'Both A/B have incomplete regions and share sensor biases; not independent laser ground truth.']}
    started=time.perf_counter()
    code=0
    try:
        experiment(args,report)
        report['status']='completed_AB_V2_both_research_gates_passed_not_independent_3D_truth'
    except Exception as ex:
        report['status']='failed'
        report['error']={'type':type(ex).__name__,'message':str(ex)}
        print('STATIC_AB_FAILED:',str(ex),file=sys.stderr)
        code=1
    finally:
        report['duration_seconds']=round(time.perf_counter()-started,3)
        (args.report_dir/'ab-run-report.json').write_text(
            json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        x=report.get('ab_comparison',{});a=x.get('A_known_pose',{});b=x.get('B_estimated_pose_sensor_scale',{})
        lines=['# Same-video, same-48/12 sensor-depth A/B (AB-V2-final)','',
               f"- status: {report['status']}",f"- [Actions]({report['run_url']})",
               f"- actual dataset SHA: {report.get('b_result',{}).get('metric_stage',{}).get('source_sha256','not completed')}",
               f"- A known-pose: coverage {a.get('coverage','n/a')}; RMSE {a.get('rmse_m','n/a')} m",
               f"- B estimated-pose: coverage {b.get('coverage','n/a')}; RMSE {b.get('rmse_m','n/a')} m",
               f"- error: {report.get('error',{}).get('message','none')}",'',
               'These are the SAME 48/12 and SAME measured sensor depths, with different pose sources.',
               'Original V1 frozen holdouts remain untouched. Sensor measurement is not a laser-scan geometry reference.']
        (args.report_dir/'AB_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return code


if __name__=='__main__':
    raise SystemExit(main())
