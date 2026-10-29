"""Stanford Lounge: preserve measured furniture/objects, add PRE-EXISTING shell.

Input 1: full-resolution observed RGB-D TSDF (426789 measured triangles).
Input 2: independently existing, already aligned and color-transferred scan.
Prior shell is detected by broad plane support, never called measured geometry.
Only measured triangles with strong coincident prior-shell support are removed
(to avoid duplicate walls/floors); unsupported measured objects remain intact.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def broad_prior_shell(vertices, faces, *, min_area=1.5, min_span=1.35,
                      long_span=2.4, plane_width=.065, normal_min=.91):
    """Detect extensive axis-aligned room planes, not desk/couch tops.

    The Stanford recorded room has approximately axis-aligned major walls.
    Per-axis area-weighted peaks identify physical planes; their candidate
    triangles must lie within 65 mm and face the plane. Return face mask and
    selected plane evidence. No inferred triangles are generated.
    """
    v=np.asarray(vertices,dtype=float)
    f=np.asarray(faces)
    if (v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or
            len(f)<20 or not np.isfinite(v).all() or f.dtype.kind not in 'iu' or
            np.any(f<0) or np.any(f>=len(v))):
        raise ValueError('Invalid prior scan geometry')
    if not .03<=plane_width<=.12 or not .8<=normal_min<=1.:
        raise ValueError('Unsafe broad-plane selection')
    triangles=v[f]
    centers=triangles.mean(axis=1)
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    areas=.5*np.linalg.norm(cross,axis=1)
    normals=cross/np.maximum(2*areas[:,None],1e-15)
    selected=np.zeros(len(f),dtype=bool)
    planes=[]
    for axis in range(3):
        aligned=(np.abs(normals[:,axis])>=normal_min)&(areas>1e-8)
        ids=np.flatnonzero(aligned)
        if len(ids)<20:
            continue
        coordinate=centers[ids,axis]
        lower=float(np.min(coordinate))
        bins=np.floor((coordinate-lower)/plane_width).astype(int)
        if np.max(bins)>5000:
            raise ValueError('Scan extends beyond bounded room size')
        weighted=np.bincount(bins,weights=areas[ids])
        for index in np.argsort(weighted)[::-1][:32]:
            if weighted[index]<min_area*.22:
                break
            seed=lower+(float(index)+.5)*plane_width
            near=aligned&(np.abs(centers[:,axis]-seed)<=plane_width*1.6)
            if not np.any(near):
                continue
            loc=float(np.median(centers[near,axis]))
            near=aligned&(np.max(np.abs(triangles[:,:,axis]-loc),axis=1)<=plane_width)
            if np.sum(areas[near])<min_area:
                continue
            points=centers[near][:,[j for j in range(3) if j!=axis]]
            span=np.quantile(points,.95,axis=0)-np.quantile(points,.05,axis=0)
            if min(span)<min_span or max(span)<long_span:
                continue
            if any(p['axis']==axis and abs(p['coordinate_m']-loc)<2*plane_width for p in planes):
                continue
            selected|=near
            planes.append({'axis':axis,'coordinate_m':loc,
                           'area_m2':float(np.sum(areas[near])),
                           'spans_m':span.tolist(), 'faces':int(np.sum(near))})
            if len(planes)>=9:
                break
    return selected,planes


def retain_observed_faces(observed_v, observed_f, prior_v, prior_f, *,
                          duplicate_distance=.085, normal_dot=.86):
    """Only suppress observed triangles coincident with the prior shell.

    Geometry in front of a wall (furniture) is preserved even if planar.
    Nearest shell face CENTROID plus normal agreement prevents blind removal.
    """
    if len(prior_f)==0:
        return np.ones(len(observed_f),dtype=bool),0
    obs=observed_v[observed_f]
    prior=prior_v[prior_f]
    oc=obs.mean(axis=1)
    pc=prior.mean(axis=1)
    on=np.cross(obs[:,1]-obs[:,0],obs[:,2]-obs[:,0])
    pn=np.cross(prior[:,1]-prior[:,0],prior[:,2]-prior[:,0])
    on/=np.maximum(np.linalg.norm(on,axis=1,keepdims=True),1e-12)
    pn/=np.maximum(np.linalg.norm(pn,axis=1,keepdims=True),1e-12)
    d,closest=cKDTree(pc).query(oc,workers=2)
    duplicate=(d<=duplicate_distance)&(np.abs(np.einsum('ij,ij->i',on,pn[closest]))>=normal_dot)
    return ~duplicate,int(duplicate.sum())


def merge_colored(observed, prior, keep_observed, keep_prior):
    import trimesh
    def part(mesh,face_mask):
        copy=mesh.copy()
        copy.update_faces(face_mask)
        copy.remove_unreferenced_vertices()
        return copy
    measured=part(observed,keep_observed)
    shell=part(prior,keep_prior)
    if len(shell.faces)==0:
        raise RuntimeError('No broad room shell found: refuse misleading hybrid')
    if len(measured.faces)==0:
        raise RuntimeError('No observed object geometry retained')
    full=trimesh.util.concatenate((measured,shell))
    color=np.zeros((len(full.vertices),4),dtype=np.uint8)
    color[:len(measured.vertices)]=[36,181,207,255]
    color[len(measured.vertices):]=[244,143,55,255]
    provenance=full.copy()
    provenance.visual.vertex_colors=color
    return full,provenance,measured,shell


def run(observed_path:Path,prior_path:Path,observed_report:Path,prior_report:Path,
        output:Path):
    import trimesh
    if output.exists():
        raise FileExistsError('Use a new output directory')
    info=json.loads(observed_report.read_text())
    prior_info=json.loads(prior_report.read_text())['prior']
    if prior_info['state']!='ALIGNED_PRIOR_ONLY' or prior_info['registration']['fitness']<.85 or prior_info['registration']['rmse_m']>.09:
        raise ValueError('Unreliable existing scan alignment')
    observed=trimesh.load(str(observed_path),force='mesh',process=False)
    prior=trimesh.load(str(prior_path),force='mesh',process=False)
    n=int(info['observed_triangles'])
    if len(observed.faces)!=info['output_triangles'] or n<1000 or n>=len(observed.faces):
        raise ValueError('Observed face boundary does not match pinned real Stanford result')
    # Remove previously inferred planar patches, never treat them as measured.
    observed.update_faces(np.arange(n))
    observed.remove_unreferenced_vertices()
    if len(prior.faces)<1000 or not np.isfinite(prior.vertices).all():
        raise ValueError('Missing aligned real pre-existing scan')
    shell_mask,planes=broad_prior_shell(prior.vertices,prior.faces)
    if len(planes)<2 or shell_mask.sum()<1000:
        raise RuntimeError('No convincing broad prior shell planes; cannot hybridize')
    keep,removed=retain_observed_faces(np.asarray(observed.vertices),np.asarray(observed.faces),
                                        np.asarray(prior.vertices),np.asarray(prior.faces)[shell_mask])
    hybrid,provenance,measured,shell=merge_colored(observed,prior,keep,shell_mask)
    output.mkdir(parents=True)
    hybrid.export(str(output/'HYBRID-observed-plus-PRIOR-shell.glb'))
    provenance.export(str(output/'HYBRID-geometry-source.glb'))
    measured.export(str(output/'OBSERVED-kept.glb'))
    shell.export(str(output/'PRIOR-shell-only.glb'))
    report={'scene':'Stanford Lounge, real 640x480 RGB-D + registered PRE-EXISTING scan',
            'geometry':'measured observed objects + existing scan planar shell',
            'input_measured_faces':n,'input_prior_preview_faces':len(prior.faces),
            'prior_registration':prior_info['registration'],
            'observed_faces_removed_as_duplicates':removed,
            'observed_faces_retained':len(measured.faces),
            'prior_shell_faces':len(shell.faces),
            'hybrid_faces':len(hybrid.faces),'prior_planes':planes,
            'all_shell_faces_are_PREEXISTING':True,
            'shell_faces_independently_recovered_from_video':False,
            'provenance':'cyan=observed RGB-D, orange=existing scan shell',
            'real_camera_visual_review':'NOT YET REVIEWED',
            'free_navigation':'NOT YET ACCEPTED',
            'issue_74':'FAIL unchanged','desk_v2':'FAIL 16.7779cm > 15cm unchanged'}
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--observed',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--observed-report',type=Path,required=True)
    p.add_argument('--prior-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        print(json.dumps(run(a.observed,a.prior,a.observed_report,a.prior_report,a.output)))
    except Exception as exc:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'FAILED.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        raise


if __name__=='__main__':
    main()
