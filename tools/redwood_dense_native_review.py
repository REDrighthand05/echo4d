"""Preserve full observed Redwood Bedroom Mesh, render 3+2 original views via CPU rays."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def barycentric_rgb(ids,uv,colors,faces):
    ids,uv,colors,faces=map(np.asarray,(ids,uv,colors,faces))
    if (ids.ndim!=1 or uv.shape!=(len(ids),2) or colors.ndim!=2 or
        colors.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or
        np.any(ids<0) or np.any(ids>=len(faces)) or not np.isfinite(uv).all() or
        np.any(uv < -1e-4) or np.any(uv.sum(axis=1)>1.0001)):
        raise ValueError("Invalid measured ray/triangle sample")
    u,v=uv[:,0],uv[:,1]
    c=colors[faces[ids]]
    return np.rint(np.clip((1-u-v)[:,None]*c[:,0]+u[:,None]*c[:,1]+v[:,None]*c[:,2],0,1)*255).astype(np.uint8)


def export_full(mesh,out):
    import trimesh
    v=np.asarray(mesh.vertices)
    f=np.asarray(mesh.triangles)
    c=np.asarray(mesh.vertex_colors)
    if (not mesh.has_triangles() or not mesh.has_vertex_colors() or
        v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or
        c.shape!=v.shape or not np.isfinite(v).all() or
        not np.isfinite(c).all() or np.any(f<0) or np.any(f>=len(v))):
        raise ValueError("Input is not a valid colored observed Mesh")
    rgba=np.column_stack((np.rint(np.clip(c,0,1)*255).astype(np.uint8),
                          np.full(len(v),255,dtype=np.uint8)))
    name="observed-full-600frames.glb"
    trimesh.Trimesh(vertices=v,faces=f,vertex_colors=rgba,process=False).export(str(out/name))
    check=trimesh.load(str(out/name),force="mesh",process=False)
    if len(check.faces)!=len(f):
        raise RuntimeError("Native GLB export discarded faces")
    return {"native_faces":len(f),"native_vertices":len(v),"glb_faces":len(check.faces),
            "glb":name,"glb_bytes":(out/name).stat().st_size,"decimated":False}


def raycast(o3d,scene,mesh,k,extrinsic):
    w,h,fx,fy,cx,cy=k
    if (w,h)!=(640,480):
        raise ValueError("Redwood source must be original calibrated 640x480")
    camera=np.array([[fx/2,0,cx/2],[0,fy/2,cy/2],[0,0,1]],dtype=np.float32)
    pose=np.asarray(extrinsic,dtype=np.float32)
    if pose.shape!=(4,4) or not np.isfinite(pose).all():
        raise ValueError("Invalid real camera pose")
    rays=o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(camera),
        extrinsic_matrix=o3d.core.Tensor(pose),width_px=320,height_px=240)
    hits=scene.cast_rays(rays)
    dist=hits["t_hit"].numpy().reshape(-1)
    ids=hits["primitive_ids"].numpy().reshape(-1)
    uv=hits["primitive_uvs"].numpy().reshape(-1,2)
    faces=np.asarray(mesh.triangles)
    good=np.isfinite(dist)&(dist>.15)&(dist<6.)&(ids<len(faces))
    rgb=np.zeros((320*240,3),dtype=np.uint8)
    if np.any(good):
        rgb[good]=barycentric_rgb(ids[good].astype(np.int64),uv[good],
                                  np.asarray(mesh.vertex_colors),faces)
    return rgb.reshape(240,320,3),int(good.sum())


def old_panel(path,free):
    with Image.open(path) as f:
        im=f.convert("RGB")
    if im.size!=((320 if free else 640),268):
        raise ValueError("Fixed old 3+2 review dimensions changed")
    return im.crop(((0 if free else 320),28,(320 if free else 640),268))


def save_comparison(path,photo,old,new,label):
    panels=([photo] if photo is not None else [])+[old,Image.fromarray(new,"RGB")]
    image=Image.new("RGB",(len(panels)*320,268),(25,25,25))
    ImageDraw.Draw(image).text((8,8),label,fill="white")
    for i,panel in enumerate(panels):
        image.paste(panel,(i*320,28))
    image.save(path)


def render_five(o3d,mesh,out,baseline,cache):
    from open3d_official_room_baseline import read_poses,spaced_real_frames
    from tum_graph_visual import look_at
    data=o3d.data.BedroomRGBDImages(data_root=str(cache))
    colors=list(data.color_paths)
    traj=o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k,poses=read_poses(traj,colors,data.trajectory_log_path)
    frames=spaced_real_frames(len(colors),360)
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    visual=out/"visual"
    visual.mkdir()
    result=[]
    for idx in (frames[0],frames[len(frames)//2],frames[-1]):
        name=f"photo-vs-mesh-{idx:05d}.png"
        old_path=baseline/"visual"/name
        with Image.open(old_path) as im:
            photo=im.convert("RGB").crop((0,28,320,268))
        old=old_panel(old_path,False)
        new,pixels=raycast(o3d,scene,mesh,k,poses[idx])
        save_comparison(visual/name,photo,old,new,"REAL PHOTO / OLD 30k / NATIVE MESH")
        result.append({"view":"real","frame":idx,"file":"visual/"+name,"native_hit_pixels":pixels})
    for j,(idx,dx) in enumerate(((frames[len(frames)//3],.35),
                                  (frames[2*len(frames)//3],-.35)),1):
        pose=np.linalg.inv(poses[idx])
        eye=pose[:3,3]+dx*pose[:3,0]
        free=look_at(eye,pose[:3,3]+1.8*pose[:3,2])
        new,pixels=raycast(o3d,scene,mesh,k,np.linalg.inv(free))
        name=f"free-view-{j}.png"
        old=old_panel(baseline/"visual"/name,True)
        save_comparison(visual/name,None,old,new,"FREE CAMERA / OLD 30k / NATIVE MESH")
        result.append({"view":"free","file":"visual/"+name,"native_hit_pixels":pixels})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-ply",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--baseline",required=True,type=Path)
    p.add_argument("--cache",type=Path)
    p.add_argument("--mode",choices=["export","views"],default="export")
    args=p.parse_args()
    import open3d as o3d
    mesh=o3d.io.read_triangle_mesh(str(args.source_ply))
    args.output.mkdir(parents=True,exist_ok=False)
    report=export_full(mesh,args.output)
    report.update({"scene":"Real Redwood Bedroom, 600 frames of 640x480 RGB-D",
                   "source_run":"https://github.com/REDrighthand05/echo4d/actions/runs/40778686280",
                   "preexisting_prior_used":False,"geometry_changed":False,
                   "previous_glb_faces":85000,"previous_render_faces":30000,
                   "visual_acceptance":"not yet reviewed","issue_74":"FAIL unchanged"})
    if args.mode=="views":
        if args.cache is None:
            p.error("--cache required for recorded camera poses")
        report["views"]=render_five(o3d,mesh,args.output,args.baseline,args.cache)
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    main()
