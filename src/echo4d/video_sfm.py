"""RGB video -> estimated COLMAP sparse SfM track, *scale unknown*.

No depth/metric pose is inferred. Do not use these coordinates as metres.
COLMAP is executed as an external process; it is not simulated or bypassed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from .colmap import _images
from .model import DataError, file_sha256, safe_path
from .video_pts import extract_source_pts

MAX_INPUT_BYTES = 2 * 1024**3
SUFFIXES = ('.png', '.jpg', '.jpeg')


def read_timestamps(path: Path) -> dict[str, float]:
    if not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise DataError('Missing/oversized timestamp CSV')
    stamps: dict[str, float] = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['name', 't']:
            raise DataError('Timestamp CSV header must be exactly name,t')
        for row in reader:
            name, raw = row.get('name'), row.get('t')
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(png|jpe?g)', name, re.I):
                raise DataError('Image filename is not a safe single-component PNG/JPEG basename')
            try:
                t = float(raw)
            except (ValueError, TypeError) as e:
                raise DataError('Invalid timestamp') from e
            if not math.isfinite(t) or t < 0 or name in stamps:
                raise DataError('Nonfinite/negative or duplicate timestamp/image name')
            stamps[name] = t
    if not 3 <= len(stamps) <= 2000 or len(set(stamps.values())) != len(stamps):
        raise DataError('Need 3-2000 frames with unique times')
    return stamps


def run_step(args: list[str], receipt: dict, label: str, timeout: int = 900) -> str:
    record = {'stage': label, 'command': args, 'status': 'running'}
    receipt.setdefault('commands', []).append(record)
    start = time.perf_counter()
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                              check=False, env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'})
        record.update({'seconds': round(time.perf_counter()-start, 3),
                       'returncode': proc.returncode,
                       'stdout_tail': proc.stdout[-6000:],
                       'stderr_tail': proc.stderr[-10000:]})
        if proc.returncode:
            raise DataError(f'{label} failed (exit {proc.returncode}): '
                            f'{proc.stderr[-1200:]}')
        record['status'] = 'passed'
        return proc.stdout
    except (subprocess.TimeoutExpired, OSError) as e:
        raise DataError(f'{label} could not complete: {type(e).__name__}: {e}') from e
    finally:
        if record['status'] != 'passed':
            record['status'] = 'failed'
            record['seconds'] = round(time.perf_counter()-start, 3)


def probe_video(video: Path, receipt: dict) -> dict:
    if not video.is_file() or not 1024 <= video.stat().st_size <= MAX_INPUT_BYTES:
        raise DataError('Video must be a nonempty regular file within the 2GiB limit')
    raw = run_step(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height,codec_name',
                    '-show_entries', 'format=duration', '-of', 'json', str(video)],
                   receipt, 'probe-video', timeout=30)
    try:
        info = json.loads(raw)
        stream = info['streams'][0]
        width, height = int(stream['width']), int(stream['height'])
        duration = float(info['format']['duration'])
    except (ValueError, TypeError, IndexError, KeyError) as e:
        raise DataError('Could not determine video dimensions and duration') from e
    if not (16 <= width <= 4096 and 16 <= height <= 4096 and
            width*height <= 4_000_000 and 0.4 <= duration <= 600):
        raise DataError('Video dimensions or duration outside processing limits')
    return {'width': width, 'height': height, 'duration_seconds': duration,
            'codec': stream.get('codec_name'), 'sha256': file_sha256(video)}


def extract_video(video: Path, image_dir: Path, fps: float, max_frames: int,
                  receipt: dict, timestamp_mode: str = 'resampled') -> dict[str, float]:
    video_info = probe_video(video, receipt)
    receipt['video'] = video_info
    if not math.isfinite(fps) or not 1 <= fps <= 12:
        raise DataError('Sample fps must be between 1 and 12')
    if not 3 <= max_frames <= 240:
        raise DataError('max_frames must be in [3,240]')
    if timestamp_mode == 'source-pts':
        return extract_source_pts(video, image_dir, max_frames, receipt, run_step)
    if timestamp_mode != 'resampled':
        raise DataError('Unknown video timestamp mode')
    image_dir.mkdir(parents=True)
    # Preserve fixed timebase, not arbitrary input video VFR packet PTS.
    run_step(['ffmpeg','-hide_banner','-loglevel','error','-nostdin',
              '-i',str(video), '-vf',f'fps={fps:.8g}',
              '-frames:v',str(max_frames),'-start_number','0',
              '-f','image2',str(image_dir/'frame_%05d.png')],
             receipt,'extract-frames',timeout=240)
    frames = sorted(image_dir.glob('frame_*.png'))
    if not 3 <= len(frames) <= max_frames:
        raise DataError('Video did not yield enough valid frames')
    # Sampling is a uniformly resampled *relative* timeline, not original camera PTS.
    return {p.name: i/fps for i,p in enumerate(frames)}


def copy_images(source: Path, dest: Path, timestamp_path: Path,
                max_frames: int, receipt: dict) -> dict[str,float]:
    stamp = read_timestamps(timestamp_path)
    names = sorted(stamp, key=lambda n: stamp[n])
    if len(names) > max_frames:
        raise DataError('Too many images: select frames explicitly in the timestamp CSV')
    dest.mkdir(parents=True)
    copied: dict[str,float] = {}
    manifest = []
    for i,name in enumerate(names):
        original = safe_path(source,name)
        out = dest/f'frame_{i:05d}{original.suffix.lower()}'
        shutil.copyfile(original,out)
        copied[out.name] = stamp[name]-stamp[names[0]]
        manifest.append({'source_name':name,'name':out.name,'t_source_s':stamp[name],
                         'sha256':file_sha256(out)})
    receipt['images_source'] = {'timestamp_csv_sha256':file_sha256(timestamp_path),
                                'frames':manifest}
    return copied


def _estimated_track(text_model: Path, timestamps: dict[str,float]) -> tuple[list[dict],int]:
    poses = _images(text_model/'images.txt')
    frames = []
    registered = set()
    for name, camera_id, pose in poses:
        if name not in timestamps or name in registered:
            raise DataError('COLMAP registered a missing or duplicated extracted frame')
        if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
                not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-5) or
                not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)):
            raise DataError('COLMAP camera pose is not a finite, proper rigid transform')
        registered.add(name)
        frames.append({'name':name,'t_s':timestamps[name],
                       'camera_id':camera_id,
                       'camera_to_sfm_world':pose.flatten(order='F').tolist()})
    if len(frames) < 3:
        raise DataError('COLMAP SfM registered fewer than three frames')
    frames.sort(key=lambda x:x['t_s'])
    return frames,len(registered)


def _model_camera_ids(path: Path) -> set[int]:
    """Check model-local camera IDs without narrowing the existing SfM model choices.

    In contrast, the downstream measured-depth bridge still intentionally
    requires undistorted PINHOLE inputs. SfM's SIMPLE_RADIAL default is kept.
    """
    if not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise DataError('Missing or oversized COLMAP cameras.txt')
    ids: set[int] = set()
    expected = {'PINHOLE': 4, 'SIMPLE_PINHOLE': 3, 'SIMPLE_RADIAL': 4}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.split()
        try:
            camera_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(v) for v in parts[4:]]
        except (IndexError, TypeError, ValueError) as exc:
            raise DataError('Invalid COLMAP cameras.txt record') from exc
        if (camera_id <= 0 or camera_id in ids or model not in expected or
                len(params) != expected[model] or
                width < 1 or height < 1 or not all(math.isfinite(v) for v in params) or
                params[0] <= 0 or (model == 'PINHOLE' and params[1] <= 0)):
            raise DataError('Invalid COLMAP camera calibration or duplicate camera ID')
        ids.add(camera_id)
    if not ids:
        raise DataError('COLMAP model has no valid cameras')
    return ids


def select_sparse_model(sparse: Path, target: Path, timestamps: dict[str, float],
                        receipt: dict) -> tuple[Path, list[dict], int]:
    """Select the most registered *valid* COLMAP model; preserve all failed candidates.

    Taken from the useful multi-model/failure idea in PR #13, while retaining
    PR #12's trajectory layout and downstream metric-sensor contract.
    Different disconnected models are NEVER stitched into one trajectory.
    """
    candidates = sorted(
        (p for p in sparse.iterdir()
         if p.is_dir() and not p.is_symlink() and p.name.isascii() and p.name.isdigit()
         and (p / 'images.bin').is_file()),
        key=lambda p: (int(p.name), p.name))
    receipt['model_candidates'] = []
    if not candidates:
        raise DataError('COLMAP mapper produced no numbered registered sparse model')
    if len(candidates) > 32:
        raise DataError('COLMAP produced more than 32 sparse models; manual review required')
    selected: tuple[Path, list[dict], int] | None = None
    for candidate in candidates:
        entry = {'model_id': candidate.name, 'status': 'rejected'}
        receipt['model_candidates'].append(entry)
        try:
            output = target / f'text-model-{candidate.name}'
            output.mkdir()
            run_step(['colmap', 'model_converter', '--input_path', str(candidate),
                      '--output_path', str(output), '--output_type', 'TXT'],
                     receipt, f'colmap-text-{candidate.name}', 90)
            cameras = _model_camera_ids(output / 'cameras.txt')
            frames, count = _estimated_track(output, timestamps)
            if any(frame['camera_id'] not in cameras for frame in frames):
                raise DataError('COLMAP model references a missing camera calibration')
            entry.update({'registered_images': count, 'status': 'valid'})
            if (selected is None or count > selected[2]):
                selected = (output, frames, count)
                receipt['selected_model_id'] = candidate.name
        except (DataError, OSError, ValueError) as error:
            entry['reason'] = f'{type(error).__name__}: {str(error)[:400]}'
    if selected is None:
        raise DataError('No valid COLMAP model with >=3 registered frames; '
                        'see report.json model_candidates')
    return selected


def run_pipeline(args: argparse.Namespace) -> dict:
    target = args.output.resolve()
    if target.exists():
        raise DataError('Output already exists; refusing overwrite')
    target.mkdir(parents=True,exist_ok=False)
    receipt = {'version':'echo-sfm-run-1','status':'running',
               'started_utc':datetime.now(timezone.utc).isoformat(),
               'declared_scale':'unknown','source':'RGB-only; no depths or GT poses for SfM',
               'commands':[],'repo_sha':os.environ.get('GITHUB_SHA')}
    start = time.perf_counter()
    try:
        images = target/'images'
        if args.video is not None:
            timestamps = extract_video(args.video.resolve(),images,args.fps,args.max_frames,receipt,
                                       timestamp_mode=args.timestamp_mode)
            receipt['input_kind'] = 'video'
            receipt['timestamp_note'] = (
                'Decoded-frame source PTS shifted relative to first decoded frame; see source_pts_sampling.frame_map.'
                if args.timestamp_mode == 'source-pts' else
                'Uniform FFmpeg fps-resampled relative times; original VFR camera packet times are not preserved.')
        else:
            timestamps = copy_images(args.images.resolve(),images,args.timestamps.resolve(),
                                     args.max_frames,receipt)
            receipt['input_kind'] = 'image_sequence'
            receipt['timestamp_note'] = 'Explicit image timestamp CSV; times shifted to first frame.'
        receipt['frames_extracted'] = len(timestamps)
        database = target/'database.db'
        sparse = target/'sparse'
        sparse.mkdir()
        camera_args = ['--ImageReader.camera_model',args.camera_model,
                       '--ImageReader.single_camera','1']
        if args.camera_params:
            camera_args += ['--ImageReader.camera_params',args.camera_params]
        run_step(['colmap','help'],receipt,'colmap-help-version-compatible',30)
        run_step(['colmap','feature_extractor','--database_path',str(database),
                  '--image_path',str(images),*camera_args,
                  '--SiftExtraction.use_gpu','0',
                  '--SiftExtraction.num_threads','2',
                  '--SiftExtraction.max_num_features','4096'],
                 receipt,'colmap-features',600)
        matching = getattr(args, 'matching_mode', 'sequential')
        if matching not in ['sequential', 'exhaustive']:
            raise DataError('Unknown COLMAP matching mode')
        matcher = ['colmap', matching+'_matcher', '--database_path', str(database)]
        if matching == 'sequential':
            matcher += ['--SequentialMatching.overlap','10']
        receipt['matching_mode'] = matching
        run_step([*matcher,'--SiftMatching.use_gpu','0',
                  '--SiftMatching.num_threads','2'],
                 receipt,'colmap-'+matching+'-matching',1200)
        run_step(['colmap','mapper','--database_path',str(database),
                  '--image_path',str(images),'--output_path',str(sparse)],
                 receipt,'colmap-sparse-mapper',1000)
        model, frames, registered_count = select_sparse_model(sparse, target, timestamps, receipt)
        with (target/'registered_timestamps.csv').open('w',newline='',encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['name','t'])
            writer.writerows([(entry['name'],entry['t_s']) for entry in frames])
        with (target/'all_input_timestamps.csv').open('w',newline='',encoding='utf-8') as fh:
            writer=csv.writer(fh)
            writer.writerow(['name','t'])
            writer.writerows(sorted(timestamps.items(),key=lambda x:x[1]))
        text = (model/'points3D.txt').read_text(encoding='utf-8')
        points = sum(bool(line.strip()) and not line.startswith('#') for line in text.splitlines())
        # These outputs deliberately do NOT conform to a metric RGB-D recording.
        track = {'version':'echo-sfm-trajectory-1','coordinate_frame':'arbitrary COLMAP SfM world',
                 'units':'unknown','metric_scale_estimated':False,
                 'physical_dimensions_available':False,
                 'depth_estimated':False,
                 'model_images_txt_sha256':file_sha256(model/'images.txt'),
                 'model_cameras_txt_sha256':file_sha256(model/'cameras.txt'),
                 'frames':frames}
        (target/'trajectory-scale-unknown.json').write_text(
            json.dumps(track,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
        receipt['result'] = {'frames_extracted':len(timestamps),
                             'frames_registered':registered_count,
                             'registration_fraction':registered_count/len(timestamps),
                             'sparse_points':points,'sfm_model_dir':model.name,
                             'scale_is_metric':False,
                             'trajectory_sha256':file_sha256(target/'trajectory-scale-unknown.json'),
                             'model_images_txt_sha256':file_sha256(model/'images.txt')}
        receipt['status'] = 'passed_rgb_only_sfm_not_metric_geometry'
    except Exception as e:
        receipt['status']='failed'
        receipt['error']={'type':type(e).__name__,'message':str(e)}
    finally:
        receipt['seconds']=round(time.perf_counter()-start,3)
        (target/'report.json').write_text(json.dumps(receipt,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    input_group=parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video',type=Path,help='MP4/MOV: FFmpeg uniformly resamples source frames')
    input_group.add_argument('--images',type=Path,help='Already-selected PNG/JPEG frames')
    parser.add_argument('--timestamps',type=Path,help='Required with --images; CSV: name,t')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fps',type=float,default=4.0)
    parser.add_argument('--timestamp-mode',choices=['resampled','source-pts'],default='resampled',
                        help='Video only: legacy fps resampling or decoded source PTS sampling')
    parser.add_argument('--max-frames',type=int,default=120)
    parser.add_argument('--matching-mode',choices=['sequential','exhaustive'],default='sequential')
    parser.add_argument('--camera-model',choices=['PINHOLE','SIMPLE_PINHOLE','SIMPLE_RADIAL'],default='SIMPLE_RADIAL')
    parser.add_argument('--camera-params',help='Optional real source intrinsics, comma-delimited in selected model order')
    args=parser.parse_args(argv)
    if (args.video is None)!=(args.timestamps is not None):
        parser.error('--timestamps is required with --images and forbidden with --video')
    if args.images is not None and args.timestamp_mode != 'resampled':
        parser.error('--timestamp-mode source-pts requires --video; --images uses explicit timestamp CSV')
    if not 3 <= args.max_frames <= 240:
        parser.error('--max-frames must be 3–240')
    if args.camera_params:
        try:
            vals=[float(v) for v in args.camera_params.split(',')]
        except ValueError as e:
            parser.error('Camera parameters must be numeric')
        expected={'PINHOLE':4,'SIMPLE_PINHOLE':3,'SIMPLE_RADIAL':4}[args.camera_model]
        if len(vals)!=expected or not all(math.isfinite(v) for v in vals):
            parser.error(f'{args.camera_model} expects {expected} finite values')
    try:
        report=run_pipeline(args)
    except (DataError,OSError,ValueError) as e:
        print(json.dumps({'status':'failed','error':str(e)}),file=sys.stderr)
        return 2
    print(json.dumps({'status':report['status'],'output':str(args.output.resolve()),
                      'result':report.get('result'),'error':report.get('error')},
                     ensure_ascii=False,allow_nan=False))
    return 0 if report['status'].startswith('passed') else 2


if __name__=='__main__':
    raise SystemExit(main())
