"""Actual Redwood Bedroom geometry improvement: measured RGB-D plus labeled prior.

Input A is a real 600-frame RGB-D-only mesh. Input B is the supplied
PRE-EXISTING Redwood Bedroom reconstruction. B is rigidly registered to A;
only B's triangles unsupported by nearby observed geometry are retained.
This is PRIOR-ASSISTED geometry, never an independent video reconstruction.
The two geometry sources are also exported as a color-coded source mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def missing_prior_faces(observed_xyz, prior_xyz, prior_faces, gap_m):
    """Select prior triangles wholly outside the observed vertex support."""
    from scipy.spatial import cKDTree
    a = np.asarray(observed_xyz, dtype=float)
    b = np.asarray(prior_xyz, dtype=float)
    f = np.asarray(prior_faces)
    if (a.ndim != 2 or a.shape[1] != 3 or len(a) < 3 or
            b.ndim != 2 or b.shape[1] != 3 or len(b) < 3 or
            f.ndim != 2 or f.shape[1] != 3 or not len(f) or
            f.dtype.kind not in "iu" or np.any(f < 0) or np.any(f >= len(b)) or
            not np.isfinite(a).all() or not np.isfinite(b).all() or
            not np.isfinite(gap_m) or not .03 <= gap_m <= .30):
        raise ValueError("Invalid measured/prior gap geometry")
    distances = cKDTree(a).query(b, k=1, workers=2)[0]
    return np.all(distances[f] > gap_m, axis=1), distances


def safe_registration(fitness, rmse, matrix):
    """Never silently paste a badly registered external mesh into a room."""
    m = np.asarray(matrix, dtype=float)
    if (m.shape != (4,4) or not np.isfinite(m).all() or
            not np.allclose(m[3],[0,0,0,1],atol=1e-5) or
            not np.allclose(m[:3,:3].T @ m[:3,:3],np.eye(3),atol=3e-3) or
            abs(np.linalg.det(m[:3,:3])-1.) > 3e-3 or
            np.linalg.norm(m[:3,3]) > 8 or
            not np.isfinite([fitness,rmse]).all() or
            fitness < .22 or rmse > .12):
        raise ValueError(f"Prior registration rejected: fitness={fitness:.3f}, rmse={rmse:.3f}")
    return m


def as_cloud(mesh, o3d):
    cloud=mesh.sample_points_uniformly(number_of_points=45000)
    cloud=cloud.voxel_down_sample(.06)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.18,max_nn=30))
    return cloud


def align_prior(prior, measured, o3d):
    """Open3D ICP; FPFH fast global registration only if local candidates fail."""
    reg=o3d.pipelines.registration
    o3d.utility.random.seed(21)
    source=as_cloud(prior,o3d)
    target=as_cloud(measured,o3d)
    src=np.asarray(source.points)
    dst=np.asarray(target.points)
    if min(len(src),len(dst)) < 100:
        raise ValueError("Too few real room surfaces to register")
    starts=[np.eye(4)]
    centroid=np.eye(4)
    centroid[:3,3]=np.mean(dst,axis=0)-np.mean(src,axis=0)
    starts.append(centroid)
    candidates=[]
    for start in starts:
        fit=reg.registration_icp(source,target,.22,start,
            reg.TransformationEstimationPointToPoint(),
            reg.ICPConvergenceCriteria(max_iteration=65))
        candidates.append(fit)
    if max(c.fitness for c in candidates) < .35:
        try:
            sfeat=reg.compute_fpfh_feature(
                source,o3d.geometry.KDTreeSearchParamHybrid(radius=.3,max_nn=90))
            tfeat=reg.compute_fpfh_feature(
                target,o3d.geometry.KDTreeSearchParamHybrid(radius=.3,max_nn=90))
            global_fit=reg.registration_fgr_based_on_feature_matching(
                source,target,sfeat,tfeat,
                reg.FastGlobalRegistrationOption(maximum_correspondence_distance=.22))
            refined=reg.registration_icp(source,target,.22,global_fit.transformation,
                reg.TransformationEstimationPointToPoint(),
                reg.ICPConvergenceCriteria(max_iteration=65))
            candidates.append(refined)
        except RuntimeError:
            pass  # retain and report the local candidates; no invented transform
    best=max(candidates,key=lambda c:(float(c.fitness),-float(c.inlier_rmse)))
    transform=safe_registration(float(best.fitness),float(best.inlier_rmse),
                                best.transformation)
    return transform, {"fitness":float(best.fitness),"rmse_m":float(best.inlier_rmse),
                       "candidate_count":len(candidates),
                       "prior_to_measured":transform.tolist()}


def load_measured(path, o3d):
    import trimesh
    t=trimesh.load(str(path),force="mesh",process=False)
    vertices=np.asarray(t.vertices,dtype=np.float64)
    faces=np.asarray(t.faces,dtype=np.int32)
    colors=np.asarray(t.visual.to_color().vertex_colors)
    if (len(faces) < 1000 or colors.shape != (len(vertices),4) or
            not np.isfinite(vertices).all()):
        raise ValueError("Expected real, colored, measured-room GLB")
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(vertices)
    mesh.triangles=o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors=o3d.utility.Vector3dVector(colors[:,:3].astype(float)/255.)
    return mesh


def merged_geometry(measured,prior,mask,o3d):
    observed_vertices=np.asarray(measured.vertices)
    observed_faces=np.asarray(measured.triangles)
    observed_colors=np.asarray(measured.vertex_colors)
    prior_vertices=np.asarray(prior.vertices)
    prior_faces=np.asarray(prior.triangles)[mask]
    used=np.unique(prior_faces)
    remap=np.full(len(prior_vertices),-1,dtype=np.int64)
    remap[used]=np.arange(len(used))
    prior_faces=remap[prior_faces]
    prior_vertices=prior_vertices[used]
    if prior.has_vertex_colors():
        prior_colors=np.asarray(prior.vertex_colors)[used]
    else:
        prior_colors=np.tile(np.array([.58,.67,.71]),(len(used),1))
    vertices=np.concatenate((observed_vertices,prior_vertices))
    faces=np.concatenate((observed_faces,prior_faces+len(observed_vertices))).astype(np.int32)
    colors=np.concatenate((observed_colors,prior_colors))
    provenance=np.concatenate((np.zeros(len(observed_vertices),dtype=np.uint8),
                               np.ones(len(used),dtype=np.uint8)))
    hybrid=o3d.geometry.TriangleMesh()
    hybrid.vertices=o3d.utility.Vector3dVector(vertices)
    hybrid.triangles=o3d.utility.Vector3iVector(faces)
    hybrid.vertex_colors=o3d.utility.Vector3dVector(np.clip(colors,0,1))
    hybrid.compute_vertex_normals()
    return hybrid, provenance, {"measured_faces":len(observed_faces),
                                "prior_gap_faces":len(prior_faces),
                                "measured_vertices":len(observed_vertices),
                                "prior_gap_vertices":len(used)}


def export_colored(path, mesh, colors):
    import trimesh
    xyz=np.asarray(mesh.vertices)
    faces=np.asarray(mesh.triangles)
    rgba=np.column_stack((
        np.rint(np.clip(colors,0,1)*255).astype(np.uint8),
        np.full(len(xyz),255,dtype=np.uint8)))
    trimesh.Trimesh(vertices=xyz,faces=faces,vertex_colors=rgba,
                    process=False).export(str(path))


def before_after(base, hybrid, out):
    out.mkdir()
    paths=[]
    for before in sorted((base/"visual").glob("*.png")):
        after=hybrid/"visual"/before.name
        if not after.is_file():
            raise RuntimeError("Review camera mismatch")
        with Image.open(before) as im:
            a=im.convert("RGB")
        with Image.open(after) as im:
            b=im.convert("RGB")
        if a.size != b.size:
            raise RuntimeError("A/B rendering dimensions mismatch")
        result=Image.new("RGB",(a.width+b.width,max(a.height,b.height)))
        result.paste(a,(0,0))
        result.paste(b,(a.width,0))
        dest=out/before.name
        result.save(dest)
        paths.append(str(dest.name))
    if len(paths)!=5:
        raise RuntimeError("Need exactly 3 same-camera and 2 free-camera A/B images")
    return paths


def run(observed_path, output, cache, gap):
    import open3d as o3d
    from open3d_official_room_baseline import read_poses,spaced_real_frames,render_views
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    dataset=o3d.data.BedroomRGBDImages(data_root=str(cache))
    observed=load_measured(observed_path,o3d)
    prior=o3d.io.read_triangle_mesh(dataset.reconstruction_path)
    if not prior.has_triangles():
        raise RuntimeError("Official pre-existing bedroom.ply has no faces")
    if len(prior.triangles)>170000:
        prior=prior.simplify_quadric_decimation(170000)
    transform,reg=align_prior(prior,observed,o3d)
    prior.transform(transform)
    keep,distances=missing_prior_faces(
        np.asarray(observed.vertices),np.asarray(prior.vertices),
        np.asarray(prior.triangles),gap)
    if not np.any(keep):
        raise RuntimeError("No prior-only missing geometry; no geometry improvement to export")
    hybrid,source_labels,counts=merged_geometry(observed,prior,keep,o3d)
    output.mkdir(parents=True)
    export_colored(output/"prior-assisted-viewable.glb",hybrid,
                   np.asarray(hybrid.vertex_colors))
    source_colors=np.where(source_labels[:,None]==0,
                           np.array([.24,.70,.76]),np.array([.98,.49,.16]))
    export_colored(output/"geometry-source-mask.glb",hybrid,source_colors)
    traj=o3d.io.read_pinhole_camera_trajectory(dataset.trajectory_log_path)
    intrinsic,extrinsics=read_poses(traj,list(dataset.color_paths),dataset.trajectory_log_path)
    review=spaced_real_frames(len(dataset.color_paths),360)
    base_dir=output/"observed-only"
    base_dir.mkdir()
    assisted_dir=output/"prior-assisted"
    assisted_dir.mkdir()
    render_views(dataset,review,extrinsics,intrinsic,observed,base_dir)
    render_views(dataset,review,extrinsics,intrinsic,hybrid,assisted_dir)
    comparison=before_after(base_dir,assisted_dir,output/"before-after")
    report={
        "scene":"Redwood Bedroom, real RGB-D, original 640x480",
        "observed_input":str(observed_path),
        "observed_input_role":"600 recorded RGB-D frames, Open3D fusion, NO PRIOR",
        "prior_input":"Open3D BedroomRGBDImages.reconstruction_path (bedroom.ply)",
        "prior_role":"PRE-EXISTING MODEL, not recovered by our video reconstruction",
        "method":"Open3D ICP (FPFH fallback), missing-prior-face gap union",
        "gap_threshold_m":gap,
        "alignment":reg,"geometry":counts,
        "prior_faces_considered":len(prior.triangles),
        "prior_faces_kept_ratio":float(np.mean(keep)),
        "prior_to_measured_vertex_distance_median":float(np.median(distances)),
        "observed_only_glb":"docs/real-room-review/redwood-bedroom-dense600/recorded-rgbd-preview.glb",
        "hybrid_glb":"prior-assisted-viewable.glb",
        "source_mask_glb":"geometry-source-mask.glb",
        "source_mask_colors":{"cyan":"observed RGB-D","orange":"existing prior-only gap geometry"},
        "before_after_images":comparison,
        "human_visual_acceptance":"NOT YET JUDGED",
        "independent_reconstruction_claim":False,
        "issue_74":"FAIL unchanged",
        "desk_v2_heldout":"16.7779 cm > 15 cm FAIL unchanged"}
    (output/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observed",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--gap-m",type=float,default=.09)
    a=p.parse_args()
    try:
        r=run(a.observed,a.output,a.cache,a.gap_m)
        print(json.dumps({"geometry":r["geometry"],"alignment":r["alignment"],
                          "visuals":len(r["before_after_images"])}))
    except Exception as ex:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/"FAILURE.txt").write_text(f"{type(ex).__name__}: {ex}\n")
        raise


if __name__=="__main__":
    main()
