"""Stanford Lounge: add only real RGB-D-supported surface triangles in observed Mesh gaps.

Full 3000-frame Open3D estimated trajectory; original 640x480 measured RGB-D.
No prior scan, predicted depth, inferred walls or hidden geometry. This is a
measured local patch experiment, NOT TSDF re-integration.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from stanford_full_observed_speck_cleanup import load_poses, camera_rays, render, export_glb

W,H = 320,240
FX,FY,CX,CY = 525.,525.,319.5,239.5
K = np.array([[262.5,0,159.5],[0,262.5,119.5],[0,0,1]],np.float32)


def read_frame(color_path, depth_path):
    with Image.open(depth_path) as im:
        raw=np.asarray(im)
    with Image.open(color_path) as im:
        color=np.asarray(im.convert("RGB"))
    if raw.shape!=(480,640) or color.shape!=(480,640,3):
        raise ValueError("Expected aligned original 640x480 RGB-D")
    z=raw.astype(np.float32).reshape(H,2,W,2).transpose(0,2,1,3).reshape(H,W,4)
    z[z==0]=np.nan
    # A median over the actual 2x2 source pixels, not an invented depth map.
    with np.errstate(all="ignore"):
        d=np.nanmedian(z,axis=2)/1000.
    d=np.nan_to_num(d,nan=0.)
    rgb=color.reshape(H,2,W,2,3).mean(axis=(1,3)).astype(np.uint8)
    return d.astype(np.float32),rgb


def rays_for_pose(o3d,pose):
    return o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(K),o3d.core.Tensor(np.linalg.inv(pose).astype(np.float32)),W,H)


def world_points(depth,pose):
    yy,xx=np.indices(depth.shape,dtype=np.float32)
    z=depth.reshape(-1)
    xyz=np.column_stack(((xx.ravel()+.5-K[0,2])*z/K[0,0],
                         (yy.ravel()+.5-K[1,2])*z/K[1,1],z))
    return (xyz@pose[:3,:3].T+pose[:3,3]).astype(np.float32)


def missing_pixels(o3d,scene,pose,depth):
    rays=rays_for_pose(o3d,pose)
    hit=scene.cast_rays(rays)
    t=hit["t_hit"].numpy().reshape(-1)
    ray=rays.numpy().reshape(-1,6)
    with np.errstate(invalid="ignore"):
        world=ray[:,:3]+ray[:,3:]*t[:,None]
        # Compare CAMERA-Z, not ray parameter, against the depth sensor.
        z_mesh=(world-pose[:3,3])@pose[:3,:3][:,2]
    z=depth.reshape(-1)
    valid=(z>=.3)&(z<=3.)
    return (valid&(~np.isfinite(t)|(z_mesh>z+.08))).reshape(H,W)


def cross_view_support(points,mask,frame,poses,depth_at):
    ids=np.flatnonzero(mask.ravel())
    accepted=np.zeros(H*W,bool)
    if not len(ids):
        return accepted.reshape(H,W)
    xyz=points[ids].astype(np.float64)
    support=np.zeros(len(ids),np.uint8)
    contradicted=np.zeros(len(ids),bool)
    for offset in (-30,-12,12,30):
        j=frame+offset
        if j<0 or j>=len(poses):
            continue
        if np.linalg.norm(poses[j,:3,3]-poses[frame,:3,3])<.025:
            continue
        camera=(xyz-poses[j,:3,3])@poses[j,:3,:3]
        z=camera[:,2]
        u=np.rint(FX*camera[:,0]/np.maximum(z,1e-6)+CX).astype(np.int32)
        v=np.rint(FY*camera[:,1]/np.maximum(z,1e-6)+CY).astype(np.int32)
        valid=(z>=.3)&(z<=3.)&(u>=0)&(u<640)&(v>=0)&(v<480)
        ix=np.flatnonzero(valid)
        if not len(ix):
            continue
        measured=depth_at(j)[v[ix],u[ix]]
        valid_depth=(measured>=300)&(measured<=3000)
        residual=measured/1000.-z[ix]
        support[ix]+=(valid_depth&(np.abs(residual)<.055)).astype(np.uint8)
        contradicted[ix]|=valid_depth&(residual<-.09)
    accepted[ids]=(support>=1)&~contradicted
    return accepted.reshape(H,W)


def smooth_measured_faces(depth,mask):
    from scipy import ndimage
    components,n=ndimage.label(mask)
    if n:
        sizes=np.bincount(components.ravel())
        sizes[0]=0
        mask=mask&(sizes[components]>=32)
    grid=np.arange(H*W,dtype=np.int32).reshape(H,W)
    a,b,c,d=grid[:-1,:-1],grid[:-1,1:],grid[1:,:-1],grid[1:,1:]
    faces=np.concatenate((np.stack((a,b,c),-1).reshape(-1,3),
                          np.stack((b,d,c),-1).reshape(-1,3)))
    z=depth.ravel()[faces]
    ok=mask.ravel()[faces].all(axis=1)
    ok&=(z.max(axis=1)-z.min(axis=1)<.035+.015*z.mean(axis=1))
    return faces[ok],mask


def make_patch(o3d,points,rgb,faces):
    if not len(faces):
        return None
    used,remap=np.unique(faces.ravel(),return_inverse=True)
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(points[used].astype(np.float64))
    mesh.triangles=o3d.utility.Vector3iVector(remap.reshape(-1,3).astype(np.int32))
    mesh.vertex_colors=o3d.utility.Vector3dVector(rgb.reshape(-1,3)[used].astype(np.float64)/255.)
    return mesh


def run(source,upstream,output):
    import open3d as o3d
    source,upstream,output=map(lambda x:Path(x).resolve(),(source,upstream,output))
    if output.exists():
        raise FileExistsError("Fresh output directory required")
    output.mkdir(parents=True)
    mesh=o3d.io.read_triangle_mesh(str(source/"scene/integrated.ply"))
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise ValueError("Missing full original observed colored Mesh")
    poses=np.asarray(load_poses(source/"scene/trajectory.log"))
    if len(poses)!=3000:
        raise ValueError("Expected 3000 estimated camera poses")
    sys.path.insert(0,str(upstream/"examples/python"))
    from open3d_example import get_rgbd_file_lists
    dataset=Path(o3d.data.LoungeRGBDImages().extract_dir)
    colors,depths=get_rgbd_file_lists(str(dataset))
    if len(colors)!=len(depths) or len(colors)!=len(poses):
        raise ValueError(f"Original RGB-D inventory mismatch: {len(colors)}, {len(depths)}, {len(poses)}")
    original_faces=len(mesh.triangles)
    views=camera_rays(o3d,mesh,poses)
    before,before_cov=render(o3d,mesh,views)
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    @lru_cache(maxsize=48)
    def depth_at(j):
        with Image.open(depths[j]) as im:
            return np.asarray(im).copy()
    ranked=[]
    for i in np.linspace(0,len(poses)-1,100,dtype=int):
        depth,_=read_frame(colors[i],depths[i])
        ranked.append((int(missing_pixels(o3d,scene,poses[i],depth).sum()),int(i)))
    ranked.sort(reverse=True)
    selected=[]
    for score,i in ranked:
        if score<250:
            break
        if all(abs(i-j)>=40 for j in selected):
            selected.append(i)
        if len(selected)>=16:
            break
    records=[]
    added=0
    for i in selected:
        depth,rgb=read_frame(colors[i],depths[i])
        missing=missing_pixels(o3d,scene,poses[i],depth)
        points=world_points(depth,poses[i])
        supported=cross_view_support(points,missing,i,poses,depth_at)
        faces,kept=smooth_measured_faces(depth,missing&supported)
        if len(faces)<64:
            records.append({"frame":i,"missing_pixels":int(missing.sum()),
                            "supported_pixels":int(kept.sum()),"added_faces":0})
            continue
        patch=make_patch(o3d,points,rgb,faces)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(patch))
        mesh+=patch
        added+=len(faces)
        records.append({"frame":i,"missing_pixels":int(missing.sum()),
                        "supported_pixels":int(kept.sum()),"added_faces":int(len(faces))})
        if added>=300000:
            break
    after,after_cov=render(o3d,mesh,views)
    export_glb(mesh,output/"observed-plus-measured-gap-patches.glb")
    preview=mesh.simplify_quadric_decimation(300000) if len(mesh.triangles)>300000 else mesh
    export_glb(preview,output/"observed-plus-measured-gap-patches-preview.glb")
    canvas=Image.new("RGB",(960,34+5*268),(20,20,20))
    draw=ImageDraw.Draw(canvas)
    draw.text((5,10),"REAL RGB / free view",fill="white")
    draw.text((325,10),"ORIGINAL observed",fill="white")
    draw.text((645,10),"MEASURED gap patches",fill="white")
    for k,(name,_) in enumerate(views):
        y=34+k*268
        draw.text((5,y+5),name,fill="white")
        if k<3:
            with Image.open(source/"views"/name) as im:
                if im.size!=(640,268):
                    raise ValueError("Unexpected original source view layout")
                canvas.paste(im.convert("RGB").crop((0,28,320,268)),(0,y+28))
        canvas.paste(before[name],(320,y+28))
        canvas.paste(after[name],(640,y+28))
    canvas.save(output/"five-view-measured-gap-ab.png")
    receipt={"input":"Stanford Lounge original 640x480 measured RGB-D",
             "estimated_poses":3000,
             "source_run":"https://github.com/REDrighthand05/echo4d/actions/runs/40877823227",
             "existing_mesh_prior":False,"predicted_depth":False,"inferred_geometry":False,
             "method":"measured local depth triangles with second-view corroboration; NOT TSDF refusion",
             "original_faces":original_faces,"added_faces":added,"final_faces":len(mesh.triangles),
             "screened_frames":100,"selected_frames":selected,"patches":records,
             "coverage_before":before_cov,"coverage_after":after_cov,
             "visual_acceptance":"NOT REVIEWED; #74 FAIL unchanged"}
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True)
    p.add_argument("--upstream",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    run(a.source,a.upstream,a.output)
