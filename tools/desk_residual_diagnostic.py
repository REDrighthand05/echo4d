"""Diagnose one frozen real Mesh. Never reconstruct, fit poses, or change gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile

import numpy as np

from echo4d.colmap import _cameras
from echo4d.holdout import read_echo_ply, rasterize_mesh_depth
from echo4d.metric_anchor import read_measured_depth
from echo4d.model import DataError, file_sha256, rigid_matrix


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
                    encoding='utf-8')


def require_hash(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise DataError(f'Frozen input hash mismatch: {path.name}')


def summarize(measured: np.ndarray, predicted: np.ndarray, mask: np.ndarray,
              lo: float, hi: float) -> dict:
    valid = mask & np.isfinite(measured) & (measured >= lo) & (measured <= hi)
    overlap = valid & np.isfinite(predicted) & (predicted >= lo) & (predicted <= hi)
    # Same subtraction order as the original metric_holdout global aggregation.
    residual = (predicted[overlap] - measured[overlap]).astype(np.float64)
    nv, no = int(valid.sum()), int(overlap.sum())
    sse = float(np.dot(residual, residual))
    return {'valid_pixels': nv, 'overlap_pixels': no, 'missing_surface_pixels': nv-no,
            'coverage': no/nv if nv else None, 'squared_error_m2': sse,
            'absolute_error_m': float(np.abs(residual).sum()),
            'signed_error_m': float(residual.sum()),
            'rmse_m': float(np.sqrt(sse/no)) if no else None,
            'mae_m': float(np.abs(residual).mean()) if no else None,
            'bias_m': float(residual.mean()) if no else None,
            'p95_abs_m': float(np.percentile(np.abs(residual), 95)) if no else None}


def edge_partition(measured: np.ndarray, lo: float, hi: float, jump: float) -> np.ndarray:
    """0=interior, 1=invalid-neighbour proxy, 2=valid depth-jump proxy.

    Never wrap rows or count outside-image neighbours as invalid sensor data.
    This does not establish occlusion or a causal explanation of residuals.
    """
    valid = np.isfinite(measured) & (measured >= lo) & (measured <= hi)
    invalid_neighbor = np.zeros(measured.shape, dtype=bool)
    discontinuity = np.zeros(measured.shape, dtype=bool)
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]),
                 (np.s_[:, :-1], np.s_[:, 1:])):
        va, vb = valid[a], valid[b]
        pair_edge = va & vb & (np.abs(measured[a]-measured[b]) > jump)
        discontinuity[a] |= pair_edge
        discontinuity[b] |= pair_edge
        invalid_neighbor[a] |= va & ~vb
        invalid_neighbor[b] |= vb & ~va
    labels = np.zeros(measured.shape, dtype=np.uint8)
    labels[invalid_neighbor] = 1
    labels[discontinuity] = 2
    return labels


def partitions(measured: np.ndarray, predicted: np.ndarray, protocol: dict) -> dict:
    if measured.ndim != 2 or predicted.shape != measured.shape:
        raise DataError('Expected same-size two-dimensional depth arrays')
    lo, hi = protocol['min_depth_m'], protocol['max_depth_m']
    h, w = measured.shape
    y, x = np.indices((h, w))
    rows, cols = protocol['image_grid_rows'], protocol['image_grid_columns']
    region = (y*rows//h)*cols + x*cols//w
    depth_edges = protocol['depth_bin_edges_m']
    # The validity mask in summarize excludes out-of-range/nonfinite values.
    depth = np.searchsorted(depth_edges[1:-1], measured, side='right')
    edge = edge_partition(measured, lo, hi, protocol['edge_proxy_jump_m'])
    residual = predicted.astype(np.float64)-measured.astype(np.float64)
    signed_edges = protocol['signed_residual_bin_edges_m']
    signed = np.searchsorted(signed_edges, residual, side='right')
    overlap = (np.isfinite(predicted) & (predicted >= lo) & (predicted <= hi))
    total = summarize(measured, predicted, np.ones((h,w), dtype=bool), lo, hi)
    result = {'total': total}
    for name, labels, names in (
        ('depth', depth, [f'[{a},{b}{"]" if i==len(depth_edges)-2 else ")"}'
                          for i,(a,b) in enumerate(zip(depth_edges,depth_edges[1:]))]),
        ('region', region, [f'row{i//cols}_col{i%cols}' for i in range(rows*cols)]),
        ('edge_proxy', edge, ['interior','invalid_neighbor','depth_jump']),
    ):
        bins = [{ 'bin': key, **summarize(measured, predicted, labels==i, lo, hi)}
                for i,key in enumerate(names)]
        if (sum(b['valid_pixels'] for b in bins) != total['valid_pixels'] or
                sum(b['overlap_pixels'] for b in bins) != total['overlap_pixels'] or
                not np.isclose(sum(b['squared_error_m2'] for b in bins),
                               total['squared_error_m2'], rtol=1e-12, atol=1e-10)):
            raise DataError('Partition failed pixel/SSE conservation')
        result[name] = bins
    # Sign bins describe overlap only; all missing pixels stay in total/other partitions.
    result['signed_residual'] = [
        {'bin_index': i, **summarize(measured, predicted, (signed==i)&overlap, lo, hi)}
        for i in range(len(signed_edges)+1)]
    if (sum(b['overlap_pixels'] for b in result['signed_residual']) != total['overlap_pixels'] or
            not np.isclose(sum(b['squared_error_m2'] for b in result['signed_residual']),
                           total['squared_error_m2'], rtol=1e-12, atol=1e-10)):
        raise DataError('Signed residual partition failed conservation')
    return result


def merge_summaries(items: list[dict]) -> dict:
    keys = ('valid_pixels','overlap_pixels','missing_surface_pixels',
            'squared_error_m2','absolute_error_m','signed_error_m')
    out = {k: sum(i[k] for i in items) for k in keys}
    nv, no = out['valid_pixels'], out['overlap_pixels']
    out.update(coverage=no/nv if nv else None,
               rmse_m=float(np.sqrt(out['squared_error_m2']/no)) if no else None,
               mae_m=out['absolute_error_m']/no if no else None,
               bias_m=out['signed_error_m']/no if no else None)
    return out


def unpack_evidence(archive: Path, target: Path, protocol: dict) -> None:
    require_hash(archive, protocol['artifact_zip_sha256'])
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if (len(members)>200 or len({m.filename for m in members})!=len(members) or
                sum(m.file_size for m in members)>300*1024**2):
            raise DataError('Evidence ZIP resource/duplicate member limit')
        if source.testzip() is not None:
            raise DataError('Evidence ZIP CRC failure')
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                raise DataError('Unsafe evidence ZIP path')
            if member.is_dir():
                continue
            dest = target.joinpath(*path.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as src, dest.open('xb') as sink:
                shutil.copyfileobj(src,sink)


def get_depths(archive: Path, output: Path, frames: list[dict], protocol: dict) -> None:
    if archive.stat().st_size != protocol['archive_bytes']:
        raise DataError('Official archive byte count changed')
    require_hash(archive, protocol['archive_sha256'])
    needed = {f['depth_filename']:f['depth_sha256'] for f in frames}
    if len(needed)!=12:
        raise DataError('Expected twelve distinct frozen depth sources')
    found = set()
    count = total = 0
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.name or
                    not path.parts or path.parts[0]!='rgbd_dataset_freiburg1_desk' or
                    not (member.isfile() or member.isdir()) or member.size>50*1024**2):
                raise DataError('Unsafe official archive member')
            count+=1; total+=member.size
            if count>25000 or total>3*1024**3:
                raise DataError('Official archive extraction resource limit')
            relative = '/'.join(path.parts[1:])
            if relative not in needed:
                continue
            if relative in found or not member.isfile():
                raise DataError('Duplicate/invalid frozen depth member')
            dest = output/Path(relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as src, dest.open('xb') as sink:
                shutil.copyfileobj(src,sink)
            require_hash(dest, needed[relative]); found.add(relative)
    if found != set(needed):
        raise DataError('Missing one or more frozen depth PNGs')


def download_archive(path: Path, protocol: dict) -> None:
    # Only public, pinned TUM bytes; never an authenticated request.
    with urlopen(protocol['archive_url'], timeout=90) as response:
        parsed = urlsplit(response.geturl())
        if parsed.scheme!='https' or parsed.hostname not in {'cvg.cit.tum.de','webshare.cvg.cit.tum.de'}:
            raise DataError('Dataset redirect outside official HTTPS source')
        size=0
        with path.open('xb') as sink:
            while chunk:=response.read(1024**2):
                size+=len(chunk)
                if size>protocol['archive_bytes']:
                    raise DataError('Official download exceeds pinned size')
                sink.write(chunk)
    require_hash(path,protocol['archive_sha256'])


def run(evidence_zip: Path, output: Path, protocol_path: Path, archive: Path | None) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(protocol_path.read_text())
    # Written before opening any heldout depth pixels or rendering residuals.
    write_json(output/'protocol.json',protocol)
    identity={'tested_checkout_sha':__import__('os').environ.get('ECHO_TESTED_CHECKOUT_SHA'),
              'protocol_sha256':file_sha256(protocol_path),
              'tool_sha256':file_sha256(Path(__file__)),
              'started_utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'execution-identity.json',identity)
    evidence=output/'frozen-inputs';evidence.mkdir()
    unpack_evidence(evidence_zip,evidence,protocol)
    for name,key in [('mesh/mesh.ply','mesh_sha256'),
                     ('desk-heldout-metrics.json','heldout_report_sha256'),
                     ('metric-trajectory-from-sensor.json','metric_trajectory_sha256'),
                     ('cameras.txt','cameras_sha256')]:
        require_hash(evidence/name,protocol[key])
    original=json.loads((evidence/'desk-heldout-metrics.json').read_text())
    frames=original['per_frame']
    if [int(f['name'][6:11]) for f in frames]!=protocol['holdout_indices']:
        raise DataError('Frozen holdout identities/order changed')
    if original['screening_gate_passed'] is not False:
        raise DataError('Original frozen experiment must remain FAIL')
    if archive is None:
        archive=output/'official-desk.tgz';download_archive(archive,protocol)
    get_depths(archive,output,frames,protocol)
    poses={f['name']:f for f in json.loads((evidence/'metric-trajectory-from-sensor.json').read_text())['frames']}
    cams=_cameras(evidence/'cameras.txt')
    if len(cams)!=1:
        raise DataError('Expected original single camera')
    original_camera=next(iter(cams.values()))
    camera=original_camera.scaled(protocol['downsample'])
    vertices,faces=read_echo_ply(evidence/'mesh/mesh.ply')
    per_frame=[]
    for frame in frames:
        name=frame['name']
        measured=read_measured_depth(output/frame['depth_filename'],
                    protocol['depth_numeric_units_per_meter'],
                    (original_camera.height,original_camera.width))
        measured=measured[::protocol['downsample'],::protocol['downsample']].astype(np.float32)
        pose=rigid_matrix(poses[name]['camera_to_metric_sfm_world'])
        predicted,raster=rasterize_mesh_depth(vertices,faces,camera,pose,far=protocol['max_depth_m'])
        result=partitions(measured,predicted,protocol)
        expected=frame['score'];actual=result['total']
        if (actual['valid_pixels']!=expected['sensor_valid_pixels'] or
                actual['overlap_pixels']!=expected['overlap_pixels'] or
                abs(actual['rmse_m']-expected['rmse_m'])>protocol['reproduction_rmse_abs_tolerance_m']):
            raise DataError(f'Frozen per-view scoring did not reproduce: {name}')
        arrays=output/(name+'.npz')
        np.savez_compressed(arrays,measured_z_m=measured,predicted_z_m=predicted,
                            edge_proxy=edge_partition(measured,protocol['min_depth_m'],
                                protocol['max_depth_m'],protocol['edge_proxy_jump_m']))
        per_frame.append({'name':name,'depth_sha256':frame['depth_sha256'],
                          'arrays_sha256':file_sha256(arrays),'raster':raster,**result})
    global_result={'total':merge_summaries([f['total'] for f in per_frame])}
    for dimension in ('depth','region','edge_proxy','signed_residual'):
        global_result[dimension]=[
            {k:v for k,v in per_frame[0][dimension][i].items() if k in ('bin','bin_index')} |
            merge_summaries([f[dimension][i] for f in per_frame])
            for i in range(len(per_frame[0][dimension]))]
        for row in global_result[dimension]:
            row['fraction_of_all_squared_error']=row['squared_error_m2']/global_result['total']['squared_error_m2']
    total=global_result['total']
    if (total['valid_pixels']!=protocol['original_valid_pixels'] or
            total['overlap_pixels']!=protocol['original_overlap_pixels'] or
            abs(total['rmse_m']-protocol['original_rmse_m'])>protocol['reproduction_rmse_abs_tolerance_m']):
        raise DataError('Frozen global scoring did not reproduce')
    report={'version':protocol['version'],'status':'DIAGNOSTIC_COMPLETE_ORIGINAL_GATE_STILL_FAIL',
            'identity':identity,'original_gate_result':protocol['original_screening_result'],
            'mesh_sha256':protocol['mesh_sha256'],'global':global_result,'per_frame':per_frame,
            'limits':protocol['constraints']}
    write_json(output/'residual-report.json',report)
    # Retain exact PNGs/arrays/inputs, but not another 344MB archive copy in artifact.
    if archive==output/'official-desk.tgz':archive.unlink()
    return report


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-zip',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path('docs/experiments/desk-v2-residual-protocol.json'))
    p.add_argument('--offline-archive',type=Path)
    args=p.parse_args()
    try:
        report=run(args.evidence_zip,args.output,args.protocol,args.offline_archive)
        print(json.dumps(report['global'],allow_nan=False));return 0
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/'diagnostic-failure.json',
                   {'status':'DIAGNOSTIC_FAILED','error':f'{type(error).__name__}: {error}',
                    'original_gate_result':'FAIL_FROZEN_15CM_RMSE_GATE'})
        raise


if __name__=='__main__':
    raise SystemExit(main())
