"""Post-hoc evaluation of monocular SfM poses, never a deployment scale solver.

Sim(3) aligns estimated and ground-truth camera centers for *evaluation only*.
It must not be used to claim estimated intrinsic metric scale.
"""
from __future__ import annotations

import math
import numpy as np
from .model import DataError


def similarity_align(estimated: np.ndarray, reference: np.ndarray) -> tuple[float,np.ndarray,np.ndarray]:
    """Return s,R,t minimizing sum ||s R x_i + t - y_i||² for COLMAP centers."""
    x,y=np.asarray(estimated,dtype=float),np.asarray(reference,dtype=float)
    if (x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or len(x)<3 or
            not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise DataError('Need at least three finite, corresponded 3D camera center pairs')
    xc=x-x.mean(axis=0)
    yc=y-y.mean(axis=0)
    if np.linalg.matrix_rank(xc,tol=1e-8)<2 or np.linalg.matrix_rank(yc,tol=1e-8)<2:
        raise DataError('Cannot align degenerate/unmoving camera trajectory')
    var=float(np.sum(xc**2)/len(x))
    u,s,vt=np.linalg.svd(yc.T@xc/len(x))
    d=np.diag([1.,1., 1. if np.linalg.det(u@vt)>0 else -1.])
    rotation=u@d@vt
    scale=float(np.sum(s*np.diag(d))/var)
    if not np.isfinite(scale) or scale<=0:
        raise DataError('Invalid fitted evaluation-only scale')
    translation=y.mean(axis=0)-scale*(rotation@x.mean(axis=0))
    return scale,rotation,translation


def evaluate_sfm_poses(trajectory: dict, truth_by_name: dict[str,list[float]]) -> dict:
    """Input truth: 16 floats/pose (column-major cameraToWorld), matched by image name.

    The GT poses may ONLY be read after COLMAP has finished, and never passed
    to the mapper or the converter. Output scale is an evaluation alignment.
    """
    frames=trajectory.get('frames')
    if trajectory.get('units')!='unknown' or trajectory.get('metric_scale_estimated') is not False:
        raise DataError('Expected unscaled monocular SfM trajectory')
    if not isinstance(frames,list) or len(frames)<3:
        raise DataError('Empty/too-small estimated trajectory')
    x,y,estR,gtR,names=[],[],[],[],[]
    for frame in frames:
        name=frame['name']
        if name not in truth_by_name:
            continue
        a=np.asarray(frame['camera_to_sfm_world'],dtype=float).reshape(4,4,order='F')
        b=np.asarray(truth_by_name[name],dtype=float).reshape(4,4,order='F')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise DataError('Nonfinite camera pose in SfM/GT')
        x.append(a[:3,3]); y.append(b[:3,3]); estR.append(a[:3,:3]);gtR.append(b[:3,:3]);names.append(name)
    if len(names)<3:
        raise DataError('Cannot evaluate: <3 estimated poses with external GT')
    scale,rot,translation=similarity_align(np.stack(x),np.stack(y))
    pred=(scale*(rot@np.stack(x).T)).T+translation
    truth=np.stack(y)
    center_err=np.linalg.norm(pred-truth,axis=1)
    angle=[]
    for a,b in zip(estR,gtR):
        delta=b.T@rot@a
        cos=np.clip((np.trace(delta)-1)/2,-1,1)
        angle.append(float(np.degrees(np.arccos(cos))))
    result={'matched_frames':len(names),'estimated_frames':len(frames),
            'ground_truth_frames':len(truth_by_name),
            'matched_fraction_of_gt':len(names)/len(truth_by_name),
            'ate_rmse_m_after_sim3':float(np.sqrt(np.mean(center_err**2))),
            'ate_median_m_after_sim3':float(np.median(center_err)),
            'position_p95_m_after_sim3':float(np.percentile(center_err,95)),
            'rotation_median_degrees_after_sim3':float(np.median(angle)),
            'rotation_p95_degrees_after_sim3':float(np.percentile(angle,95)),
            'evaluation_only_sim3_scale_m_per_sfm_unit':scale,
            'evaluation_alignment_rotation_3x3_rowmajor':rot.flatten().tolist(),
            'evaluation_alignment_translation_m':translation.tolist(),
            'alignment_scope':'Fit and scored on the SAME matched camera poses (optimistic in-sample ATE).',
            'metric_scale_available_in_product':False,
            'pose_source':'COLMAP RGB SfM (model registration, no GT used in mapper)',
            'truth_source':'TUM motion capture poses, used only after solving SfM',
            'individual': [{'name':name,'position_error_m':float(e),'rotation_error_degrees':float(a)}
                           for name,e,a in zip(names,center_err,angle)]}
    return result