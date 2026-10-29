"""Mature Open3D Poisson continuity on real Redwood Bedroom observed RGB-D Mesh.

Original measured TSDF faces are kept unchanged. New faces are INFERRED
geometry, never represented as observed. Existing reference scan not used.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def supported_faces(vertices,faces,distance,density,lower,upper,
                    min_gap=.035,max_support=.16,max_edge=.24):
    v=np.asarray(vertices,dtype=float)
    f=np.asarray(faces,dtype=np.int64)
    d=np.asarray(distance,dtype=float)
    rho=np.asarray(density,dtype=float)
    lo=np.asarray(lower,dtype=float)
    hi=np.asarray(upper,dtype=float)
    if (v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or
            d.shape!=(len(v),) or rho.shape!=(len(v),) or
            lo.shape!=(3,) or hi.shape!=(3,) or
            not np.isfinite(v).all() or not np.isfinite(d).all() or
            not np.isfinite(rho).all() or np.any(d<0) or
            np.any(f<0) or np.any(f>=len(v)) or
            not (0<min_gap<max_support<1 and 0<max_edge<1)):
        raise ValueError("Invalid bounded Poisson support inputs")
    good=(d<=max_support)&(rho>=np.percentile(rho,15))&np.all(
        (v>=lo[None,:]-.04)&(v<=hi[None,:]+.04),axis=1)
    keep=np.all(good[f],axis=1)&(np.max(d[f],axis=1)>min_gap)
    for a,b in ((0,1),(1,2),(2,0)):
        delta=v[f[:,a]]-v[f[:,b]]
        keep &= np.einsum("ij,ij->i",delta,delta)<=max_edge**2
    return keep


def export_glb(mesh,path):
    import trimesh
    v=np.asarray(mesh.vertices)
    f=np.asarray(mesh.triangles)
    c=np.asarray(mesh.vertex_colors)
    if not mesh.has_triangles() or not mesh.has_vertex_colors() or c.shape!=v.shape:
        raise ValueError("Cannot export an uncolored/empty scene")
    rgba=np.column_stack((np.rint(np.clip(c,0,1)*255).astype(np.uint8),
                          np.full(len(c),255,dtype=np.uint8)))
    trimesh.Trimesh(vertices=v,faces=f,vertex_colors=rgba,process=False).export(str(path))


def save_panel(path,photo,rendered,label):
    panels=([photo.resize((320,240))] if photo is not None else [])+[
        Image.fromarray(x[0],"RGB") for x in rendered]
    image=Image.new("RGB",(320*len(panels),268),(25,25,25))
    ImageDraw.Draw(image).text((8,8),label,fill="white")
    for j,panel in enumerate(panels):
        image.paste(panel,(320*j,28))
    image.save(path)


def render_five(o3d,observed,proposed,marked,cache,out):
    from open3d_official_room_baseline import read_poses,spaced_real_frames
    from redwood_dense_native_review import raycast
    from tum_graph_visual import look_at
    data=o3d.data.BedroomRGBDImages(data_root=str(cache))
    trajectory=o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k,poses=read_poses(trajectory,data.color_paths,data.trajectory_log_path)
    frames=spaced_real_frames(len(data.color_paths),360)
    meshes=(observed,proposed,marked)
    scenes=[]
    for mesh in meshes:
        scene=o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        scenes.append(scene)
    visual=out/"visual"
    visual.mkdir()
    result=[]
    for frame in (frames[0],frames[len(frames)//2],frames[-1]):
        views=[raycast(o3d,scene,mesh,k,poses[frame])
               for scene,mesh in zip(scenes,meshes)]
        with Image.open(data.color_paths[frame]) as im:
            photo=im.convert("RGB").copy()
        name=f"photo-vs-mesh-{frame:05d}.png"
        save_panel(visual/name,photo,views,"REAL PHOTO / MEASURED TSDF / POISSON / ORANGE=INFERRED")
        result.append({"kind":"recorded","frame":frame,"image":"visual/"+name,
                       "covered_pixels":[v[1] for v in views]})
    for j,(frame,dx) in enumerate(((frames[len(frames)//3],.35),
                                  (frames[2*len(frames)//3],-.35)),1):
        pose=np.linalg.inv(poses[frame])
        free=look_at(pose[:3,3]+dx*pose[:3,0],pose[:3,3]+1.8*pose[:3,2])
        views=[raycast(o3d,scene,mesh,k,np.linalg.inv(free))
               for scene,mesh in zip(scenes,meshes)]
        name=f"free-view-{j}.png"
        save_panel(visual/name,None,views,"FREE VIEW / MEASURED TSDF / POISSON / ORANGE=INFERRED")
        result.append({"kind":"free","image":"visual/"+name,
                       "covered_pixels":[v[1] for v in views]})
    return result


def run(source,cache,out,depth,points,min_gap,max_support):
    import open3d as o3d
    from scipy.spatial import cKDTree
    if not 7<=depth<=9 or not 30000<=points<=220000:
        raise ValueError("Poisson depth/point budget out of bounds")
    if not (math.isfinite(min_gap) and math.isfinite(max_support)
            and .015<=min_gap<=.09 and .08<=max_support<=.25 and min_gap<max_support):
        raise ValueError("Unsafe inferred gap range")
    observed=o3d.io.read_triangle_mesh(str(source))
    if not observed.has_vertex_colors() or len(observed.triangles)!=561887:
        raise ValueError("Expected real observed-only 720-frame Redwood PLY")
    observed.compute_vertex_normals()
    o3d.utility.random.seed(17)
    samples=observed.sample_points_uniformly(number_of_points=points,use_triangle_normal=False)
    if not samples.has_normals():
        raise RuntimeError("No observed surface normals for mature Poisson")
    poisson,densities=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        samples,depth=depth,scale=1.05,linear_fit=False,n_threads=2)
    v=np.asarray(poisson.vertices)
    f=np.asarray(poisson.triangles)
    measured_xyz=np.asarray(observed.vertices)
    distances,nearest=cKDTree(measured_xyz).query(v,k=1,workers=2)
    bbox=observed.get_axis_aligned_bounding_box()
    mask=supported_faces(v,f,distances,np.asarray(densities),
                         bbox.get_min_bound(),bbox.get_max_bound(),
                         min_gap=min_gap,max_support=max_support)
    if not mask.any():
        raise RuntimeError("Poisson has no locally supported missing surfaces")
    patch=o3d.geometry.TriangleMesh()
    patch.vertices=o3d.utility.Vector3dVector(v.copy())
    patch.triangles=o3d.utility.Vector3iVector(f[mask].copy())
    patch.vertex_colors=o3d.utility.Vector3dVector(
        np.asarray(observed.vertex_colors)[nearest].copy())
    patch.remove_unreferenced_vertices()
    patch.compute_vertex_normals()
    orange=o3d.geometry.TriangleMesh(patch)
    orange.paint_uniform_color([1.,.35,.025])
    proposed=observed+patch
    marked=observed+orange
    out.mkdir(parents=True,exist_ok=False)
    if not o3d.io.write_triangle_mesh(str(out/"combined-inferred.ply"),proposed):
        raise RuntimeError("Native colored Mesh export failed")
    o3d.io.write_triangle_mesh(str(out/"inferred-only.ply"),patch)
    export_glb(proposed,out/"combined-colored.glb")
    export_glb(marked,out/"combined-provenance.glb")
    visuals=render_five(o3d,observed,proposed,marked,cache,out)
    report={"scene":"REAL Redwood Bedroom original 640x480 RGB-D",
            "input":"720 measured depth frames with recorded poses",
            "source_run":"https://github.com/REDrighthand05/echo4d/actions/runs/40798646737",
            "method":"stock Open3D Poisson; original observed Mesh preserved",
            "measured_faces":len(observed.triangles),
            "inferred_faces":len(patch.triangles),
            "combined_faces":len(proposed.triangles),
            "raw_poisson_faces":len(poisson.triangles),
            "poisson_depth":depth,"sampled_points":points,
            "minimum_gap_m":min_gap,"max_nearest_measured_vertex_m":max_support,
            "preexisting_scan_used":False,
            "inferred_surface_colors":"nearest observed vertex approximation",
            "provenance":"orange triangles inferred; all others observed",
            "visuals":visuals,"human_visual_acceptance":"NOT REVIEWED",
            "independent_surface_accuracy":"NOT MEASURED",
            "issue_74":"FAIL unchanged","old_desk_v2":"FAIL 16.7779cm > 15cm unchanged"}
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-ply",required=True,type=Path)
    p.add_argument("--cache",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--depth",type=int,default=8)
    p.add_argument("--points",type=int,default=150000)
    p.add_argument("--min-gap",type=float,default=.035)
    p.add_argument("--max-support",type=float,default=.16)
    a=p.parse_args()
    try:
        result=run(a.source_ply,a.cache,a.output,a.depth,a.points,a.min_gap,a.max_support)
        print(json.dumps({"faces":result["combined_faces"],
                          "inferred":result["inferred_faces"],
                          "views":len(result["visuals"])}))
    except Exception as exc:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__=="__main__":
    main()
