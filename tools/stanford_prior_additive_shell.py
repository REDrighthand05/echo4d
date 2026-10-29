"""Observed-first real Stanford room: add only genuinely missing PRE-EXISTING shell.

The prior replacement experiment removed 74,641 observed triangles and visibly
worsened some recorded views. This variant NEVER removes measured geometry.
The already-aligned reference is used only where observed vertices do not
support a broad outer wall/floor/ceiling. It remains explicitly prior geometry.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from stanford_observed_prior_shell import broad_prior_shell,merge_colored


def boundary_shell_mask(vertices,faces,*,margin_m=.38):
    """Select broad prior planes close to the scan's external bounds.

    This excludes large interior horizontal surfaces such as tables; it is a
    deliberately conservative shell, not a complete semantic room model.
    """
    raw,planes=broad_prior_shell(vertices,faces,min_area=1.5)
    mins=np.min(vertices,axis=0)
    maxs=np.max(vertices,axis=0)
    eligible=[]
    keep=np.zeros(len(faces),dtype=bool)
    tris=np.asarray(vertices)[np.asarray(faces)]
    for plane in planes:
        axis=plane['axis']
        pos=plane['coordinate_m']
        if min(abs(pos-mins[axis]),abs(pos-maxs[axis]))>margin_m:
            continue
        on_plane=np.max(np.abs(tris[:,:,axis]-pos),axis=1)<=.065
        keep|=raw&on_plane
        eligible.append(plane)
    return keep,eligible


def unsupported_prior_faces(observed_vertices,prior_vertices,prior_faces,*,gap_m=.075):
    """Require all three existing-scan vertices to lack nearby measured support.

    Reject overlap from the prior, not observed faces. This avoids removing
    furniture/wall observations even if the registration is only 4 cm RMS.
    """
    if not .04<=gap_m<=.15:
        raise ValueError('Unsafe overlap tolerance')
    observed=np.asarray(observed_vertices,dtype=float)
    prior=np.asarray(prior_vertices,dtype=float)
    faces=np.asarray(prior_faces)
    if not len(faces):
        return np.empty(0,dtype=bool),0
    if not np.isfinite(observed).all() or not np.isfinite(prior).all():
        raise ValueError('Nonfinite scene geometry')
    distances=cKDTree(observed).query(prior,k=1,workers=2)[0]
    keep=np.all(distances[faces]>gap_m,axis=1)
    return keep,int(np.count_nonzero(~keep))


def run(observed_path,prior_path,observed_report,prior_report,output):
    import trimesh
    if output.exists():
        raise FileExistsError('New output required')
    info=json.loads(observed_report.read_text())
    pinfo=json.loads(prior_report.read_text())['prior']
    if pinfo['state']!='ALIGNED_PRIOR_ONLY' or pinfo['registration']['fitness']<.85 or pinfo['registration']['rmse_m']>.09:
        raise ValueError('Reject unaligned existing scan')
    observed=trimesh.load(str(observed_path),force='mesh',process=False)
    prior=trimesh.load(str(prior_path),force='mesh',process=False)
    n=int(info['observed_triangles'])
    if len(observed.faces)!=info['output_triangles'] or n<1000 or n>=len(observed.faces):
        raise ValueError('Pinned observed face boundary mismatch')
    observed.update_faces(np.arange(n))
    observed.remove_unreferenced_vertices()
    outer,planes=boundary_shell_mask(np.asarray(prior.vertices),np.asarray(prior.faces))
    if len(planes)<2:
        raise RuntimeError('Insufficient outer scan planes; do not claim complete room')
    candidates=np.flatnonzero(outer)
    gap,discarded=unsupported_prior_faces(np.asarray(observed.vertices),np.asarray(prior.vertices),
                                          np.asarray(prior.faces)[candidates])
    mask=np.zeros(len(prior.faces),dtype=bool)
    mask[candidates[gap]]=True
    if mask.sum()<100:
        raise RuntimeError('No meaningful missing shell geometry from this existing scan')
    full,source,measured,shell=merge_colored(observed,prior,
                                              np.ones(len(observed.faces),dtype=bool),mask)
    output.mkdir(parents=True)
    full.export(str(output/'HYBRID-observed-plus-PRIOR-shell.glb'))
    source.export(str(output/'HYBRID-geometry-source.glb'))
    measured.export(str(output/'OBSERVED-kept.glb'))
    shell.export(str(output/'PRIOR-shell-only.glb'))
    report={'scene':'Stanford Lounge real 640x480 RGB-D + aligned PRE-EXISTING room scan',
            'method':'preserve every observed face; only unsupported outer prior shell',
            'observed_faces_retained':len(measured.faces),
            'original_observed_faces':n,
            'prior_outer_shell_candidates':len(candidates),
            'prior_overlap_faces_rejected':discarded,
            'prior_shell_faces':len(shell.faces),'hybrid_faces':len(full.faces),
            'outer_planes':planes,'prior_registration':pinfo['registration'],
            'prior_shell_is_PREEXISTING':True,
            'independently_recovered_prior_faces':False,
            'previous_replacement_experiment':'40827393230, visible degradation',
            'real_camera_visual_review':'NOT YET REVIEWED',
            'free_navigation':'NOT YET ACCEPTED',
            'issue_74':'FAIL unchanged','desk_v2':'FAIL 16.7779cm > 15cm unchanged'}
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('observed','prior','observed-report','prior-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:
        print(json.dumps(run(a.observed,a.prior,a.observed_report,a.prior_report,a.output)))
    except Exception as e:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'FAILED.txt').write_text(f'{type(e).__name__}: {e}\n')
        raise
