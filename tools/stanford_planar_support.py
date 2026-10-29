"""Stanford Lounge: measured-surface-supported planar gap patches (NOT a prior).

Start from the previously published real 360-frame RGB-D TSDF GLB. Fit broad
planes to its observed vertices, close only small interior gaps with measured
surrounding planar support, and export the inferred faces SEPARATELY.
No pre-existing room mesh is used. Inferred triangles are not measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def broad_planes(vertices, normals, *, trials=350, rounds=7, seed=20261029):
    """Deterministic normal-aware RANSAC for broad walls and floor; no furniture patches."""
    vertices=np.asarray(vertices,dtype=float)
    normals=np.asarray(normals,dtype=float)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or normals.shape!=vertices.shape
            or not np.isfinite(vertices).all() or not np.isfinite(normals).all()
            or len(vertices)<1500):
        raise ValueError("Need finite measured 3D vertices and normals")
    rng=np.random.default_rng(seed)
    active=np.ones(len(vertices),dtype=bool)
    result=[]
    for _ in range(rounds):
        ids=np.flatnonzero(active)
        if len(ids)<900:
            break
        sample=ids[rng.choice(len(ids),min(len(ids),30000),replace=False)]
        pts=vertices[sample]
        nrm=normals[sample]
        choices=rng.choice(len(sample),min(trials,len(sample)),replace=False)
        best=(0,None,None)
        for j in choices:
            normal=nrm[j]
            if np.max(np.abs(normal))<.75:
                continue
            count=int(np.count_nonzero((np.abs((pts-pts[j])@normal)<.035)
                                       &(np.abs(nrm@normal)>.80)))
            if count>best[0]:
                best=(count,normal,pts[j])
        if best[0]<700:
            break
        normal,center=best[1:]
        inliers=np.empty(0,dtype=int)
        for _ in range(3):
            inliers=ids[(np.abs((vertices[ids]-center)@normal)<.035)
                        &(np.abs(normals[ids]@normal)>.75)]
            if len(inliers)<600:
                break
            center=vertices[inliers].mean(axis=0)
            _,_,basis=np.linalg.svd(vertices[inliers]-center,full_matrices=False)
            normal=basis[-1]
        if len(inliers)<600:
            break
        active[inliers]=False
        uv=(vertices[inliers]-center)@basis[:2].T
        span=np.quantile(uv,.98,axis=0)-np.quantile(uv,.02,axis=0)
        if len(inliers)>=900 and min(span)>=1.0 and max(span)>=1.7:
            result.append((center,normal,basis[:2],inliers,span))
    return result


def planar_gap_tiles(vertices, colors, plane, *, grid=.07, radius=.65,
                     tree=None, max_grid_cells=250000):
    """Close only bounded holes near actual coplanar measured support.

    binary_closing cannot extend the occupied silhouette; a nearby real
    inlier is required; and an existing measured vertex vetoes duplicates.
    Output is independent inferred geometry, not a watertight guarantee.
    """
    center,normal,basis,ids,span=plane
    uv=(vertices[ids]-center)@basis.T
    lo=uv.min(axis=0)-grid
    hi=uv.max(axis=0)+grid
    shape=np.ceil((hi-lo)/grid).astype(int)+1
    if np.prod(shape)>max_grid_cells:
        raise ValueError("Planar support grid exceeds bounded scene budget")
    cell=np.floor((uv-lo)/grid).astype(int)
    cell=np.clip(cell,[0,0],shape-1)
    occupied=np.zeros(tuple(shape),dtype=bool)
    occupied[cell[:,0],cell[:,1]]=True
    steps=int(round(radius/grid))
    x,y=np.ogrid[-steps:steps+1,-steps:steps+1]
    disk=(x*x+y*y)<=steps*steps
    inferred=ndimage.binary_closing(occupied,structure=disk)&~occupied
    labels,n=ndimage.label(inferred)
    if n:
        sizes=np.bincount(labels.ravel())
        inferred&=(sizes[labels]<=max(10,int(math.pi*(radius/grid)**2*1.5)))
    cells=np.argwhere(inferred)
    if not len(cells):
        return np.empty((0,3)),np.empty((0,3),dtype=int),np.empty((0,3)),0
    centers2d=lo+(cells+.5)*grid
    support=cKDTree(uv)
    near,_=support.query(centers2d)
    world=center+centers2d@basis
    if tree is None:
        tree=cKDTree(vertices)
    measured_dist,_=tree.query(world)
    cells=cells[(near<=radius)&(measured_dist>=grid*.65)]
    if not len(cells):
        return np.empty((0,3)),np.empty((0,3),dtype=int),np.empty((0,3)),0
    corners=np.array([[0,0],[1,0],[1,1],[0,1]],dtype=int)
    all_corners=(cells[:,None,:]+corners).reshape(-1,2)
    unique,inv=np.unique(all_corners,axis=0,return_inverse=True)
    uv_vertices=lo+unique*grid
    out_vertices=center+uv_vertices@basis
    quads=inv.reshape(-1,4)
    faces=np.concatenate((quads[:,[0,1,2]],quads[:,[0,2,3]]),axis=0)
    _,nearest=support.query(uv_vertices)
    out_colors=colors[ids][nearest]
    return out_vertices,faces,out_colors,len(cells)


def reconstruct(source:Path,output:Path,*,grid=.07,radius=.65):
    import trimesh
    if output.exists():
        raise FileExistsError("New output directory required")
    if not (.04<=grid<=.12 and .12<=radius<=.85 and radius>=grid):
        raise ValueError("Bounded planar gap geometry only")
    measured=trimesh.load(str(source),force="mesh",process=False)
    xyz=np.asarray(measured.vertices,dtype=float)
    rgb=np.asarray(measured.visual.vertex_colors)[:,:3].astype(float)/255.
    if (len(measured.faces)<1000 or len(xyz)<1500 or rgb.shape!=xyz.shape
            or not np.isfinite(xyz).all()):
        raise ValueError("Requires a colored, genuine recorded RGB-D Mesh")
    normals=np.asarray(measured.vertex_normals,dtype=float)
    planes=broad_planes(xyz,normals)
    tree=cKDTree(xyz)
    vertices=[];faces=[];colors=[];records=[];offset=0
    for center,normal,basis,ids,span in planes:
        v,f,c,count=planar_gap_tiles(xyz,rgb,(center,normal,basis,ids,span),
                                     grid=grid,radius=radius,tree=tree)
        if not len(f):
            continue
        vertices.append(v);faces.append(f+offset);colors.append(c)
        offset+=len(v)
        records.append({"observed_support_vertices":len(ids),"inferred_grid_cells":count,
                        "normal":normal.tolist(),"support_span_m":span.tolist()})
    if not faces:
        raise RuntimeError("No supported planar holes: retain measured Mesh unchanged")
    v=np.concatenate(vertices)
    f=np.concatenate(faces)
    c=np.concatenate(colors)
    rgba=np.column_stack((np.rint(np.clip(c,0,1)*255).astype(np.uint8),
                          np.full(len(v),255,dtype=np.uint8)))
    patch=trimesh.Trimesh(vertices=v,faces=f,vertex_colors=rgba,process=False)
    combined=trimesh.util.concatenate([measured,patch])
    provenance=trimesh.util.concatenate([
        trimesh.Trimesh(vertices=measured.vertices,faces=measured.faces,
                        vertex_colors=np.tile(np.array([25,184,190,255],dtype=np.uint8),
                                              (len(measured.vertices),1)),process=False),
        trimesh.Trimesh(vertices=v,faces=f,
                        vertex_colors=np.tile(np.array([255,145,35,255],dtype=np.uint8),
                                              (len(v),1)),process=False)])
    output.mkdir(parents=True)
    patch.export(str(output/"INFERRED-planar-only.glb"))
    combined.export(str(output/"OBSERVED-plus-INFERRED.glb"))
    provenance.export(str(output/"OBSERVED-vs-INFERRED-provenance.glb"))
    report={"scene":"REAL Stanford Lounge, previously recorded RGB-D TSDF",
            "source_observed_mesh":str(source),"input_prior_used":False,
            "observed_triangles":len(measured.faces),"inferred_planar_triangles":len(f),
            "output_triangles":len(combined.faces),"inferred_area_m2":float(patch.area),
            "plane_patches":records,"method":"RANSAC broad planes, bounded 2D morphological hole fill",
            "inferred_geometry_is_measured":False,
            "source_colors":"nearest coplanar measured RGB-D vertex; no generated textures",
            "provenance":"cyan=measured RGB-D; orange=plane-inferred tiles",
            "three_real_two_free_views":"not yet rendered",
            "human_visual_improvement":"NOT VERIFIED",
            "issue_74":"FAIL unchanged","desk_v2":"FAIL 16.7779cm > 15cm unchanged"}
    (output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--grid",type=float,default=.07)
    p.add_argument("--radius",type=float,default=.65)
    a=p.parse_args()
    r=reconstruct(a.source,a.output,grid=a.grid,radius=a.radius)
    print(json.dumps({"observed":r["observed_triangles"],"inferred":r["inferred_planar_triangles"],"planes":len(r["plane_patches"])}),flush=True)


if __name__=="__main__":
    main()
