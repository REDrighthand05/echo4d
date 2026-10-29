"""Stanford Lounge: choose unused *measured* RGB-D views for absent surfaces.

The frozen 360-view observed mesh is used only to choose new frames and as
the before image. NEW geometry is fused afresh from recorded RGB-D + poses.
No provided laser scan, inferred plane, or prior mesh enters the TSDF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def candidate_indices(total, baseline, stride=4):
    if total < 3 or not 1 <= stride <= 32:
        raise ValueError("Invalid recorded frame range/stride")
    base = set(baseline)
    if not base or min(base) < 0 or max(base) >= total:
        raise ValueError("Invalid baseline frame indices")
    # Avoid near-duplicates of the 360 already fused views.
    return [i for i in range(0, total, stride)
            if i not in base and min(abs(i-j) for j in base) >= 3]


def select_views(cells, budget):
    if not 1 <= budget <= 240:
        raise ValueError("Extra measured view budget must be 1..240")
    remaining = {int(i): set(s) for i, s in cells.items()}
    chosen, gains, covered = [], [], set()
    while remaining and len(chosen) < budget:
        best = max(remaining, key=lambda i:
                   (len(remaining[i]-covered), len(remaining[i]), -i))
        gain = len(remaining[best]-covered)
        if gain < 3:
            break
        chosen.append(best)
        gains.append(gain)
        covered.update(remaining.pop(best))
    return chosen, gains


def depth_gap_cells(o3d, scene, depth_path, extrinsic, intrinsic, cell=.10):
    """Find valid 640x480 sensor depths missing from a frozen observed mesh."""
    w,h,fx,fy,cx,cy = intrinsic
    if (w,h) != (640,480) or not all(np.isfinite([fx,fy,cx,cy])) or min(fx,fy)<=0:
        raise ValueError("Require calibrated native Stanford 640x480 RGB-D")
    raw = np.asarray(o3d.io.read_image(str(depth_path)))
    if raw.shape != (h,w) or raw.dtype != np.uint16:
        raise ValueError("Depth frame is not native 16-bit 640x480")
    pose = np.asarray(extrinsic, dtype=float)
    if pose.shape != (4,4) or not np.isfinite(pose).all():
        raise ValueError("Invalid recorded camera extrinsic")
    # Sample sensor pixels (4+8u, 4+8v) exactly; adjust principal point
    # for the half-pixel convention of create_rays_pinhole.
    step, offset = 8, 4
    z = raw[offset::step,offset::step].astype(np.float32)/1000.
    valid = (z>.35)&(z<3.95)
    # Discard isolated depth spikes; they should not win frame selection.
    neighbours = [raw[3::8,4::8],raw[5::8,4::8],
                  raw[4::8,3::8],raw[4::8,5::8]]
    valid &= np.logical_and.reduce([np.abs(n.astype(float)/1000.-z)<.075
                                    for n in neighbours])
    kk = np.array([[fx/step,0,(cx-offset)/step+.5],
                   [0,fy/step,(cy-offset)/step+.5],
                   [0,0,1]],dtype=np.float32)
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(kk),
        extrinsic_matrix=o3d.core.Tensor(pose.astype(np.float32)),
        width_px=w//step,height_px=h//step)
    hit = scene.cast_rays(rays)["t_hit"].numpy()
    good = np.isfinite(hit)&(hit>.15)&(hit<20.)
    rr = rays.numpy()
    xyz = rr[...,:3]+np.where(good,hit,0)[...,None]*rr[...,3:]
    old_z = (xyz@pose[:3,:3].T+pose[:3,3])[...,2]
    missing = valid & (~good | (old_z>z+.10))
    yy,xx = np.nonzero(missing)
    if not len(xx):
        return set(),{"missing_pixels":0,"valid_pixels":int(valid.sum())}
    u=offset+step*xx
    v=offset+step*yy
    d=z[yy,xx]
    local=np.column_stack(((u-cx)*d/fx,(v-cy)*d/fy,d))
    inv=np.linalg.inv(pose)
    world=local@inv[:3,:3].T+inv[:3,3]
    q=np.floor(world/cell).astype(np.int64)
    if not np.isfinite(world).all() or np.any(np.abs(q)>30000):
        raise ValueError("Invalid/out-of-bounds measured world depth")
    q+=32768
    packed=(q[:,0]<<32)|(q[:,1]<<16)|q[:,2]
    cells=set(np.unique(packed).tolist())
    return cells,{"missing_pixels":int(len(xx)),"valid_pixels":int(valid.sum()),
                  "unique_missing_cells":len(cells)}


def observed_mesh(source, report):
    import trimesh
    model=trimesh.load(str(source),force="mesh",process=False)
    count=int(json.loads(Path(report).read_text())["observed_triangles"])
    if not 100000 < count <= len(model.faces):
        raise ValueError("Wrong frozen observed-only face count")
    model.update_faces(np.arange(count))
    model.remove_unreferenced_vertices()
    if len(model.faces)!=count:
        raise RuntimeError("Could not isolate the observed-only source geometry")
    return model


def to_open3d(o3d, model):
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(np.asarray(model.vertices))
    mesh.triangles=o3d.utility.Vector3iVector(np.asarray(model.faces,dtype=np.int32))
    rgba=np.asarray(model.visual.vertex_colors)
    if rgba.shape[0]!=len(model.vertices):
        raise ValueError("Frozen observed source has no vertex colors")
    mesh.vertex_colors=o3d.utility.Vector3dVector(rgba[:,:3].astype(float)/255.)
    return mesh


def free_render(o3d,scene,mesh,center,eye):
    from redwood_dense_native_review import barycentric_rgb
    rays=o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=78.,
        center=o3d.core.Tensor(center.astype(np.float32)),
        eye=o3d.core.Tensor(eye.astype(np.float32)),
        up=o3d.core.Tensor(np.array([0.,-1.,0.],dtype=np.float32)),
        width_px=320,height_px=240)
    hit=scene.cast_rays(rays)
    dist=hit["t_hit"].numpy().reshape(-1)
    ids=hit["primitive_ids"].numpy().reshape(-1)
    uv=hit["primitive_uvs"].numpy().reshape(-1,2)
    faces=np.asarray(mesh.triangles)
    good=np.isfinite(dist)&(dist>.15)&(dist<16.)&(ids<len(faces))
    pixels=np.zeros((320*240,3),dtype=np.uint8)
    if np.any(good):
        pixels[good]=barycentric_rgb(ids[good].astype(np.int64),uv[good],
                                     np.asarray(mesh.vertex_colors),faces)
    return pixels.reshape(240,320,3),int(good.sum())


def five_views(o3d,data,poses,k,before,after,out):
    from open3d_official_room_baseline import spaced_real_frames
    from redwood_dense_native_review import raycast
    scenes=[]
    for mesh in (before,after):
        scene=o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        scenes.append(scene)
    review=spaced_real_frames(len(data.color_paths),360)
    center=(np.asarray(before.get_min_bound())+
            np.asarray(before.get_max_bound()))/2
    span=np.asarray(before.get_max_bound())-np.asarray(before.get_min_bound())
    views=[("recorded",i,poses[i]) for i in
           (review[0],review[len(review)//2],review[-1])]
    views.extend([("free",1,center+span*np.array([-.50,-.90,-.70])),
                  ("free",2,center+span*np.array([.80,-1.20,.05]))])
    folder=out/"visual"
    folder.mkdir()
    result=[]
    for kind,index,pose in views:
        if kind=="recorded":
            old,n0=raycast(o3d,scenes[0],before,k,pose)
            new,n1=raycast(o3d,scenes[1],after,k,pose)
            with Image.open(data.color_paths[index]) as image:
                photo=image.convert("RGB").resize((320,240))
            panels=[photo,Image.fromarray(old),Image.fromarray(new)]
            name=f"photo-vs-mesh-{index:05d}.png"
        else:
            old,n0=free_render(o3d,scenes[0],before,center,pose)
            new,n1=free_render(o3d,scenes[1],after,center,pose)
            panels=[Image.fromarray(old),Image.fromarray(new)]
            name=f"free-view-{index}.png"
        canvas=Image.new("RGB",(320*len(panels),268),(24,24,24))
        ImageDraw.Draw(canvas).text((8,8),
            "REAL RGB / OBSERVED 360 / OBSERVED + MEASURED VIEWS"
            if kind=="recorded" else
            "ROOM-SCALE FREE / OBSERVED 360 / OBSERVED + MEASURED VIEWS",
            fill="white")
        for j,panel in enumerate(panels):
            canvas.paste(panel,(j*320,28))
        canvas.save(folder/name)
        result.append({"kind":kind,"file":"visual/"+name,
                       "before_hit_pixels":n0,"after_hit_pixels":n1})
    return result


def run(cache,source,source_report,output,extra,stride):
    import open3d as o3d
    from stanford_lounge_second_room import lounge_recorded_poses,integrate
    from open3d_official_room_baseline import spaced_real_frames
    if output.exists():
        raise FileExistsError("Use a fresh result directory")
    data=o3d.data.LoungeRGBDImages(data_root=str(cache))
    if len(data.color_paths)!=len(data.depth_paths):
        raise ValueError("Unsynchronized Stanford RGB-D")
    if any(Path(a).stem!=Path(b).stem for a,b in
           zip(data.color_paths,data.depth_paths)):
        raise ValueError("Stanford color/depth frame mismatch")
    traj=o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k,poses,mapping=lounge_recorded_poses(
        traj,list(data.color_paths),data.trajectory_log_path)
    before=to_open3d(o3d,observed_mesh(source,source_report))
    base=spaced_real_frames(len(data.color_paths),360)
    candidates=candidate_indices(len(data.color_paths),base,stride)
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(before))
    gaps={}
    stats=[]
    for i in candidates:
        cells,s=depth_gap_cells(o3d,scene,data.depth_paths[i],poses[i],k)
        gaps[i]=cells
        stats.append({"frame":i,**s})
    selected,gains=select_views(gaps,extra)
    output.mkdir(parents=True)
    (output/"selection.json").write_text(json.dumps({
        "candidates":stats,"selected":selected,"new_cell_gains":gains},indent=2)+"\n")
    if not selected:
        raise RuntimeError("No measured missing surfaces: retain old observed mesh")
    # Re-fuse ALL 360 baseline frames plus selected real frames. Do not
    # union two independently meshed surfaces or inject supplied scan faces.
    after=integrate(o3d,data,k,poses,sorted(set(base)|set(selected)),.025)
    if not o3d.io.write_triangle_mesh(str(output/"OBSERVED-gap-guided-full.ply"),after):
        raise RuntimeError("Failed to export native measured Mesh")
    from redwood_prior_first_room import export_colored
    export_colored(output/"OBSERVED-gap-guided-full.glb",
                   np.asarray(after.vertices),np.asarray(after.triangles),
                   np.asarray(after.vertex_colors))
    views=five_views(o3d,data,poses,k,before,after,output)
    result={
        "scene":"Stanford Lounge 640x480 real RGB-D",
        "source":"Open3D LoungeRGBDImages recorded frames and logged poses",
        "pose_mapping":mapping,"baseline_frames":len(base),
        "candidate_frames":len(candidates),"new_frames":len(selected),
        "selected_frame_indices":selected,
        "before_observed_faces":len(before.triangles),
        "after_observed_faces":len(after.triangles),
        "existing_prior_geometry_used":False,
        "frozen_observed_mesh_only_for_frame_selection_and_comparison":True,
        "method":"Open3D ScalableTSDFVolume .025m voxel, measured depth",
        "three_recorded_two_room_scale_free":views,
        "human_visual_acceptance":"NOT REVIEWED",
        "issue_74":"FAIL unchanged",
        "desk_v2":"FAIL 16.7779cm > 15cm unchanged"}
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",required=True,type=Path)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--source-report",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--extra",type=int,default=120)
    p.add_argument("--stride",type=int,default=4)
    args=p.parse_args()
    if not 1<=args.extra<=240:
        p.error("extra must be 1..240")
    try:
        run(args.cache,args.source,args.source_report,
            args.output,args.extra,args.stride)
    except Exception as exc:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/"FAILED.txt").write_text(
            f"{type(exc).__name__}: {exc}\n")
        raise


if __name__=="__main__":
    main()
