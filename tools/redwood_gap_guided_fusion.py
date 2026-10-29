"""Re-fuse real Redwood Bedroom depth views chosen to fill observed Mesh gaps.

Baseline 600-frame PLY is used for view SELECTION, never as geometry prior.
Official RGB-D, calibration and logged poses are the only fusion inputs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

from open3d_official_room_baseline import read_poses,spaced_real_frames
from redwood_dense_native_review import raycast,export_full
from tum_graph_visual import look_at


def surface_cells(points,cell_m=.08):
    xyz=np.asarray(points,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError("Invalid real measured XYZ")
    grid=np.floor(xyz/cell_m).astype(np.int64)
    if np.any(np.abs(grid)>32000):
        raise ValueError("Measured XYZ outside local room bounds")
    q=grid+32768
    return set(np.unique((q[:,0]<<32)|(q[:,1]<<16)|q[:,2]).tolist())


def measured_gaps(o3d,scene,depth_path,extrinsic,k):
    """Raycast frozen Mesh and locate true sensor depth closer than its surface."""
    w,h,fx,fy,cx,cy=k
    if (w,h)!=(640,480):
        raise ValueError("Original Redwood calibration must be 640x480")
    depth=np.asarray(o3d.io.read_image(depth_path))
    if depth.shape!=(h,w) or depth.dtype!=np.uint16:
        raise ValueError("Original sensor depth must be 16-bit millimeters")
    z=depth[4::8,4::8].astype(float)/1000.
    intr=np.array([[fx/8,0,cx/8],[0,fy/8,cy/8],[0,0,1]],dtype=np.float32)
    pose=np.asarray(extrinsic,dtype=float)
    rays=o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(intr),
        extrinsic_matrix=o3d.core.Tensor(pose.astype(np.float32)),
        width_px=80,height_px=60)
    hit=scene.cast_rays(rays)["t_hit"].numpy()
    valid_hit=np.isfinite(hit)&(hit>0)&(hit<100)
    raw=rays.numpy()
    hit_xyz=raw[...,:3]+np.where(valid_hit,hit,0)[...,None]*raw[...,3:]
    old_z=(hit_xyz @ pose[:3,:3].T+pose[:3,3])[...,2]
    good=(z>.25)&(z<4.5)
    gaps=good&(~valid_hit | (old_z>z+.09))
    conflict=good&valid_hit&(old_z<z-.09)
    yy,xx=np.nonzero(gaps)
    if not len(xx):
        return set(),{"missing_pixels":0,"conflict_pixels":int(conflict.sum())}
    u=(xx+.5)*8
    v=(yy+.5)*8
    zz=z[yy,xx]
    local=np.column_stack(((u-cx)*zz/fx,(v-cy)*zz/fy,zz))
    inv=np.linalg.inv(pose)
    world=local@inv[:3,:3].T+inv[:3,3]
    cells=surface_cells(world)
    return cells,{"missing_pixels":int(len(xx)),"conflict_pixels":int(conflict.sum()),
                  "unique_missing_cells":len(cells)}


def choose_gap_views(candidates,limit):
    if not 1<=limit<=240:
        raise ValueError("Extra view budget 1..240")
    remaining={i:set(s) for i,s in candidates.items()}
    covered=set()
    selected=[]
    gains=[]
    while remaining and len(selected)<limit:
        best=max(remaining,key=lambda i:(len(remaining[i]-covered),len(remaining[i]),-i))
        gain=len(remaining[best]-covered)
        if gain==0:
            break
        selected.append(best)
        gains.append(gain)
        covered.update(remaining.pop(best))
    return selected,gains


def integrate(o3d,data,indices,poses,k):
    volume=o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=.025,sdf_trunc=.1,volume_unit_resolution=16,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=1)
    intr=o3d.camera.PinholeCameraIntrinsic(*k)
    for index in sorted(indices):
        color=o3d.io.read_image(data.color_paths[index])
        depth=o3d.io.read_image(data.depth_paths[index])
        if np.asarray(color).shape[:2]!=(k[1],k[0]) or np.asarray(depth).shape[:2]!=(k[1],k[0]):
            raise ValueError("Real sensor frames mismatch calibration")
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,depth,depth_scale=1000.,depth_trunc=4.5,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd,intr,poses[index])
    mesh=volume.extract_triangle_mesh()
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise RuntimeError("Measured fusion produced no colored Mesh")
    mesh.compute_vertex_normals()
    return mesh


def panel(path,photo,before,after,title):
    images=([photo.resize((320,240))] if photo is not None else [])+[
        Image.fromarray(before,"RGB"),Image.fromarray(after,"RGB")]
    result=Image.new("RGB",(320*len(images),268),(26,26,26))
    ImageDraw.Draw(result).text((8,8),title,fill="white")
    for i,img in enumerate(images):
        result.paste(img,(i*320,28))
    result.save(path)


def compare(o3d,data,poses,k,old,new,out):
    a=o3d.t.geometry.RaycastingScene()
    b=o3d.t.geometry.RaycastingScene()
    a.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(old))
    b.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(new))
    frames=spaced_real_frames(len(data.color_paths),360)
    visual=out/"visual"
    visual.mkdir()
    views=[]
    for idx in (frames[0],frames[len(frames)//2],frames[-1]):
        prev,p1=raycast(o3d,a,old,k,poses[idx])
        now,p2=raycast(o3d,b,new,k,poses[idx])
        with Image.open(data.color_paths[idx]) as f:
            photo=f.convert("RGB").copy()
        name=f"photo-vs-mesh-{idx:05d}.png"
        panel(visual/name,photo,prev,now,"REAL RGB / ORIGINAL 600 / GAP-GUIDED")
        views.append({"kind":"recorded","frame":idx,"file":"visual/"+name,
                      "old_pixels":p1,"new_pixels":p2})
    for j,(idx,dx) in enumerate(((frames[len(frames)//3],.35),
                                  (frames[2*len(frames)//3],-.35)),1):
        pose=np.linalg.inv(poses[idx])
        eye=pose[:3,3]+dx*pose[:3,0]
        free=look_at(eye,pose[:3,3]+1.8*pose[:3,2])
        prev,p1=raycast(o3d,a,old,k,np.linalg.inv(free))
        now,p2=raycast(o3d,b,new,k,np.linalg.inv(free))
        name=f"free-view-{j}.png"
        panel(visual/name,None,prev,now,"FREE CAMERA / ORIGINAL 600 / GAP-GUIDED")
        views.append({"kind":"free","file":"visual/"+name,"old_pixels":p1,"new_pixels":p2})
    return views


def run(source,cache,out,extra):
    import open3d as o3d
    data=o3d.data.BedroomRGBDImages(data_root=str(cache))
    if len(data.color_paths)!=len(data.depth_paths):
        raise ValueError("Unmatched measured RGB-D frames")
    if any(Path(c).stem!=Path(d).stem for c,d in zip(data.color_paths,data.depth_paths)):
        raise ValueError("Measured RGB-D frame IDs differ")
    traj=o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k,poses=read_poses(traj,data.color_paths,data.trajectory_log_path)
    original=o3d.io.read_triangle_mesh(str(source))
    if not original.has_vertex_colors() or len(original.triangles)!=553208:
        raise ValueError("Wrong actual frozen 600-frame observed-only baseline")
    base=spaced_real_frames(len(data.color_paths),600)
    mids=sorted({(a+b)//2 for a,b in zip(base,base[1:])}-set(base))
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(original))
    candidates={}
    observations=[]
    for idx in mids:
        cells,stats=measured_gaps(o3d,scene,data.depth_paths[idx],poses[idx],k)
        candidates[idx]=cells
        observations.append({"frame":idx,**stats})
    selected,gains=choose_gap_views(candidates,extra)
    if not selected:
        raise RuntimeError("No uncovered measured depth in intermediate frames; no fabricated gain")
    out.mkdir(parents=True,exist_ok=False)
    new=integrate(o3d,data,set(base)|set(selected),poses,k)
    if not o3d.io.write_triangle_mesh(str(out/"gap-guided-full.ply"),new):
        raise RuntimeError("Failed to export actual measured full Mesh")
    exported=export_full(new,out)
    (out/exported["glb"]).rename(out/"gap-guided-observed.glb")
    exported["glb"]="gap-guided-observed.glb"
    visuals=compare(o3d,data,poses,k,original,new,out)
    report={"scene":"Redwood Bedroom, actual 640x480 RGB-D and logged poses",
            "change":"up to 120 unused measured depth views selected for missing surfaces",
            "old_faces":len(original.triangles),"new_faces":len(new.triangles),
            "baseline_frames":len(base),"candidate_frames":len(mids),
            "new_frames":len(selected),"selected_frame_indices":selected,
            "unique_gap_cell_gains":gains,
            "old_mesh_used_only_for_view_selection":True,
            "supplied_reference_prior_used":False,
            "geometry_engine":"Open3D ScalableTSDFVolume voxel 0.025m",
            "new_glb":exported["glb"],"native_export":exported,
            "visuals":visuals,"human_visual_acceptance":"NOT YET REVIEWED",
            "independent_surface_accuracy":"NOT MEASURED",
            "issue_74":"FAIL unchanged",
            "source_run":"https://github.com/REDrighthand05/echo4d/actions/runs/40778686280"}
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"candidate-gaps.json").write_text(json.dumps(observations)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-ply",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--extra",type=int,default=120)
    a=p.parse_args()
    if not 1<=a.extra<=240:
        p.error("Extra views must be 1..240")
    try:
        print(json.dumps(run(a.source_ply,a.cache,a.output,a.extra)))
    except Exception as e:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/"FAILURE.txt").write_text(f"{type(e).__name__}: {e}\n")
        raise


if __name__=="__main__":
    main()
