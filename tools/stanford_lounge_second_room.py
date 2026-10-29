"""Real second room: Stanford Lounge 3000-frame RGB-D, separate existing PRIOR."""
import argparse,json
from pathlib import Path
import numpy as np


def integrate(o3d,data,k,poses,indices,voxel):
    if not .015<=voxel<=.06: raise ValueError("invalid voxel")
    engine=o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,sdf_trunc=4*voxel,volume_unit_resolution=16,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=1)
    camera=o3d.camera.PinholeCameraIntrinsic(*k)
    for i in indices:
        color=o3d.io.read_image(data.color_paths[i])
        depth=o3d.io.read_image(data.depth_paths[i])
        if np.asarray(color).shape[:2]!=(k[1],k[0]) or np.asarray(depth).shape[:2]!=(k[1],k[0]):
            raise ValueError(f"real frame {i} does not match logged intrinsics")
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,depth,depth_scale=1000.,depth_trunc=4.,
            convert_rgb_to_intensity=False)
        engine.integrate(rgbd,camera,poses[i])
    mesh=engine.extract_triangle_mesh()
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise RuntimeError("Measured RGB-D produced no colored surface")
    mesh.compute_vertex_normals()
    return mesh


def prior_variant(o3d,data,measured,out):
    from redwood_prior_gapfill import align_prior
    from redwood_prior_first_room import transfer_observed_colors,export_colored
    source=Path(data.reconstruction_path)
    if not source.is_file(): return None,{"state":"NO_PREEXISTING_MODEL"}
    prior=o3d.io.read_triangle_mesh(str(source))
    if not prior.has_triangles(): return None,{"state":"INVALID_PREEXISTING_MODEL"}
    original=len(prior.triangles)
    if original>170000: prior=prior.simplify_quadric_decimation(170000)
    try:
        transform,registration=align_prior(prior,measured,o3d)
    except (ValueError,RuntimeError) as exc:
        return None,{"state":"PRIOR_ALIGNMENT_REJECTED","reason":str(exc)}
    prior.transform(transform)
    prior.compute_vertex_normals()
    measured.compute_vertex_normals()
    colors,mask=transfer_observed_colors(prior,measured,max_distance_m=.09)
    prior.vertex_colors=o3d.utility.Vector3dVector(colors)
    xyz=np.asarray(prior.vertices)
    faces=np.asarray(prior.triangles)
    export_colored(out/"PREEXISTING-PRIOR-RGB.glb",xyz,faces,colors)
    provenance=np.where(mask[:,None]==1,np.array([.20,.68,.76]),np.array([.98,.49,.16]))
    export_colored(out/"PREEXISTING-PRIOR-provenance.glb",xyz,faces,provenance)
    return prior,{"state":"ALIGNED_PRIOR_ONLY","all_faces_preexisting":True,
                  "original_faces":original,"preview_faces":len(faces),
                  "observed_color_fraction":float(np.mean(mask==1)),
                  "registration":registration}


def run(output,cache,frames,voxel):
    import open3d as o3d
    from open3d_official_room_baseline import read_poses,spaced_real_frames,render_views
    from redwood_prior_first_room import export_colored,compose_ab
    if output.exists(): raise FileExistsError("New output directory required")
    data=o3d.data.LoungeRGBDImages(data_root=str(cache))
    n=len(data.color_paths)
    if n<3 or len(data.depth_paths)!=n: raise ValueError("Missing synchronized real RGB-D")
    trajectory=o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k,poses=read_poses(trajectory,list(data.color_paths),data.trajectory_log_path)
    indices=spaced_real_frames(n,min(n,frames))
    output.mkdir(parents=True)
    observed=integrate(o3d,data,k,poses,indices,voxel)
    if not o3d.io.write_triangle_mesh(str(output/"OBSERVED-full.ply"),observed):
        raise RuntimeError("Measured PLY export failed")
    preview=(observed.simplify_quadric_decimation(85000)
             if len(observed.triangles)>85000 else observed)
    export_colored(output/"OBSERVED-viewable.glb",np.asarray(preview.vertices),
                   np.asarray(preview.triangles),np.asarray(preview.vertex_colors))
    review=spaced_real_frames(n,min(n,360))
    observed_dir=output/"observed"
    observed_dir.mkdir()
    views=render_views(data,review,poses,k,observed,observed_dir)
    prior,prior_result=prior_variant(o3d,data,observed,output)
    ab=[]
    if prior is not None:
        prior_dir=output/"prior"
        prior_dir.mkdir()
        render_views(data,review,poses,k,prior,prior_dir)
        ab=compose_ab(observed_dir,prior_dir,output/"ab")
    report={"scene":"REAL Stanford Lounge (not Redwood Bedroom)",
            "dataset":"Open3D LoungeRGBDImages",
            "input":"actual measured RGB-D and logged camera poses",
            "calibrated_intrinsics":k,"available_frames":n,
            "fused_frames":len(indices),"observed_faces":len(observed.triangles),
            "observed_model":"OBSERVED-viewable.glb",
            "observed_faces_are_independently_fused":True,
            "prior":prior_result,"prior_faces_independently_recovered":False,
            "three_real_and_two_free_views":views,"prior_AB_images":ab,
            "human_visual_acceptance":"NOT REVIEWED",
            "issue_74":"FAIL unchanged","desk_v2":"FAIL 16.7779cm > 15cm unchanged"}
    (output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"scene":report["scene"],"observed_faces":report["observed_faces"],
                      "prior_state":prior_result["state"],"AB_images":len(ab)}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--cache",required=True,type=Path)
    p.add_argument("--frames",type=int,default=360)
    p.add_argument("--voxel",type=float,default=.025)
    a=p.parse_args()
    if not 3<=a.frames<=600: p.error("frames must be 3..600")
    try: run(a.output,a.cache,a.frames,a.voxel)
    except Exception as e:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/"FAILED.txt").write_text(f"{type(e).__name__}: {e}\n")
        raise


if __name__=="__main__": main()
