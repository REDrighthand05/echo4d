"""Reference-FIRST Redwood room: existing room surface aligned and RGB-D colored.

This is explicitly PRIOR geometry, not a reconstruction from images.
Measured RGB-D mesh is used only to align and transfer observed color.
All unobserved/unsupported colors are neutral, not hallucinated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def transfer_observed_colors(prior, observed, *, max_distance_m=.09,
                             min_normal_dot=.35):
    """Nearest compatible measured surface colors; no invented geometry.

    Source labels: 1=pre-existing geometry with observed RGB-D color,
    2=pre-existing geometry with unknown color.
    """
    from scipy.spatial import cKDTree
    p=np.asarray(prior.vertices,dtype=float)
    q=np.asarray(observed.vertices,dtype=float)
    pc=np.asarray(prior.vertex_normals,dtype=float)
    qc=np.asarray(observed.vertex_normals,dtype=float)
    rgb=np.asarray(observed.vertex_colors,dtype=float)
    if (len(p)<3 or len(q)<3 or not np.isfinite(p).all() or
            not np.isfinite(q).all() or pc.shape!=p.shape or
            qc.shape!=q.shape or rgb.shape!=q.shape or
            not np.isfinite(rgb).all() or
            not .02<=max_distance_m<=.2 or
            not 0<=min_normal_dot<=1):
        raise ValueError("Invalid calibrated measured/prior color transfer")
    k=min(8,len(q))
    distance, indices=cKDTree(q).query(p,k=k,workers=2)
    if k==1:
        distance=distance[:,None]
        indices=indices[:,None]
    normal_dot=np.einsum("ij,ikj->ik",pc,qc[indices])
    valid=(distance<=max_distance_m)&(normal_dot>=min_normal_dot)
    # A nearby wall behind a desk is NOT a substitute for the desk surface.
    nearest=np.where(valid,distance,np.inf)
    selected=np.argmin(nearest,axis=1)
    supported=np.isfinite(nearest[np.arange(len(p)),selected])
    colors=np.tile(np.array([.53,.56,.59]),(len(p),1))
    colors[supported]=rgb[indices[np.arange(len(p))[supported],selected[supported]]]
    source=np.where(supported,1,2).astype(np.uint8)
    return colors,source


def export_colored(path,vertices,faces,colors):
    import trimesh
    rgba=np.column_stack((np.rint(np.clip(colors,0,1)*255).astype(np.uint8),
                          np.full(len(vertices),255,dtype=np.uint8)))
    trimesh.Trimesh(vertices=vertices,faces=faces,vertex_colors=rgba,
                    process=False).export(str(path))


def compose_ab(before,after,out):
    out.mkdir()
    files=[]
    for old in sorted((before/"visual").glob("*.png")):
        new=after/"visual"/old.name
        if not new.is_file():
            raise RuntimeError("Fixed 3+2 cameras do not match")
        with Image.open(old) as a, Image.open(new) as b:
            left,right=a.convert("RGB"),b.convert("RGB")
        if left.size!=right.size:
            raise RuntimeError("A/B image size mismatch")
        merged=Image.new("RGB",(left.width+right.width,left.height))
        merged.paste(left,(0,0))
        merged.paste(right,(left.width,0))
        merged.save(out/old.name)
        files.append("ab/"+old.name)
    if len(files)!=5:
        raise RuntimeError("Require three same-camera and two free-view renders")
    return files


def run(observed_path, output, cache, max_distance_m):
    import open3d as o3d
    from redwood_prior_gapfill import load_measured,align_prior
    from open3d_official_room_baseline import (
        read_poses,spaced_real_frames,render_views)
    if output.exists():
        raise FileExistsError("Output directory must be new")
    dataset=o3d.data.BedroomRGBDImages(data_root=str(cache))
    observed=load_measured(observed_path,o3d)
    prior=o3d.io.read_triangle_mesh(dataset.reconstruction_path)
    if not prior.has_triangles():
        raise RuntimeError("No real supplied existing room model")
    transform,registration=align_prior(prior,observed,o3d)
    prior.transform(transform)
    observed.compute_vertex_normals()
    prior.compute_vertex_normals()
    colors,source=transfer_observed_colors(prior,observed,
                                          max_distance_m=max_distance_m)
    prior.vertex_colors=o3d.utility.Vector3dVector(colors)
    vertices=np.asarray(prior.vertices)
    faces=np.asarray(prior.triangles)
    output.mkdir(parents=True)
    export_colored(output/"PRIOR-room-observed-colors.glb",vertices,faces,colors)
    mask=np.where(source[:,None]==1,np.array([.20,.68,.76]),
                  np.array([.98,.49,.16]))
    export_colored(output/"PRIOR-room-color-source.glb",vertices,faces,mask)
    # Preserve all ~10M original PRIOR faces in Actions artifacts. Publish only
    # an explicitly decimated viewable copy under GitHub's 100 MB file limit.
    preview=prior.simplify_quadric_decimation(170000)
    preview.compute_vertex_normals()
    preview_rgb,preview_source=transfer_observed_colors(
        preview,observed,max_distance_m=max_distance_m)
    preview_xyz=np.asarray(preview.vertices)
    preview_faces=np.asarray(preview.triangles)
    export_colored(output/"PRIOR-room-preview.glb",
                   preview_xyz,preview_faces,preview_rgb)
    preview_mask=np.where(preview_source[:,None]==1,
                          np.array([.20,.68,.76]),np.array([.98,.49,.16]))
    export_colored(output/"PRIOR-room-preview-color-source.glb",
                   preview_xyz,preview_faces,preview_mask)
    traj=o3d.io.read_pinhole_camera_trajectory(dataset.trajectory_log_path)
    intrinsic,poses=read_poses(traj,list(dataset.color_paths),
                               dataset.trajectory_log_path)
    review=spaced_real_frames(len(dataset.color_paths),360)
    baseline=output/"measured-only"
    baseline.mkdir()
    with_prior=output/"prior-first"
    with_prior.mkdir()
    baseline_views=render_views(dataset,review,poses,intrinsic,observed,baseline)
    prior_views=render_views(dataset,review,poses,intrinsic,prior,with_prior)
    images=compose_ab(baseline,with_prior,output/"ab")
    report={
        "scene":"Real Redwood Bedroom, 640x480 RGB-D and logged poses",
        "geometry_role":"ALL output faces are PRE-EXISTING room model PRIOR",
        "geometry_source":"Open3D BedroomRGBDImages.reconstruction_path",
        "observed_input_role":"720-frame RGB-D-only TSDF used for registration and color transfer",
        "prior_fused_from_video_independently":False,
        "method":"existing model rigid ICP aligned + normal-compatible observed RGB-D vertex color",
        "registration":registration,
        "measured_faces":len(observed.triangles),
        "prior_faces":len(faces),
        "prior_vertices":len(vertices),
        "full_prior_glb_in_actions_only":"PRIOR-room-observed-colors.glb",
        "github_preview_faces":len(preview_faces),
        "github_preview_glb":"PRIOR-room-preview.glb",
        "github_preview_provenance":"PRIOR-room-preview-color-source.glb",
        "color_from_measured_rgbd_vertices":int(np.count_nonzero(source==1)),
        "unknown_color_prior_vertices":int(np.count_nonzero(source==2)),
        "color_coverage_fraction":float(np.mean(source==1)),
        "max_color_transfer_distance_m":max_distance_m,
        "colors_for_unobserved":"neutral grey, not inferred",
        "source_mask":"cyan=prior with measured RGB color; orange=prior with unknown color; ALL faces prior",
        "colored_glb":"PRIOR-room-observed-colors.glb",
        "provenance_glb":"PRIOR-room-color-source.glb",
        "fixed_3plus2_comparisons":images,
        "observed_view_records":baseline_views,
        "prior_view_records":prior_views,
        "human_recognition":"NOT REVIEWED",
        "independent_geometry_accuracy":"NOT MEASURED",
        "issue_74":"FAIL unchanged",
        "desk_v2":"FAIL 16.7779cm > 15cm unchanged"
    }
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"prior_faces":len(faces),
                      "observed_color_fraction":report["color_coverage_fraction"],
                      "comparisons":len(images)}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observed",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--max-color-distance-m",type=float,default=.09)
    a=p.parse_args()
    run(a.observed,a.output,a.cache,a.max_color_distance_m)


if __name__=="__main__":
    main()
