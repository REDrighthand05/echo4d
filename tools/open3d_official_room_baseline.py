"""Stock Open3D reconstruction on official real Redwood RGB-D footage.

The supplied existing reconstruction is a separately labeled PRIOR.
No acoustic sonar, estimated depth, or prior mesh is fused into this result.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def spaced_real_frames(total, count):
    if type(total) is not int or type(count) is not int or not 3 <= count <= min(total, 600):
        raise ValueError("Need 3-600 genuine, distinct recorded camera frames")
    result = [round(i * (total - 1) / (count - 1)) for i in range(count)]
    if len(set(result)) != count:
        raise ValueError("Sampled duplicate frame")
    return result


def trajectory_frame_ids(log_path, expected_count):
    """Match Redwood LOG first-column frame IDs, not list positions.

    Open3D's official BedroomRGBDImages enumerates image/000001.jpg onward
    while its trajectory.log can include camera 000000. Ignoring this shifts
    every RGB-D measurement onto the wrong camera or rejects the dataset.
    """
    rows = [line.strip() for line in Path(log_path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != expected_count * 5:
        raise ValueError(f"Redwood LOG rows {len(rows)} != 5 * poses {expected_count}")
    ids = []
    for index in range(expected_count):
        tokens = rows[index * 5].split()
        if len(tokens) != 3 or not all(token.lstrip("-").isdigit() for token in tokens):
            raise ValueError(f"Bad Redwood trajectory header {index}")
        ids.append(int(tokens[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate logged Redwood camera frame ID")
    return ids


def read_poses(trajectory, image_paths, trajectory_path):
    """Return world-to-camera matrices aligned to *actual* image filename IDs."""
    params = trajectory.parameters
    if not params:
        raise ValueError("Empty recorded camera trajectory")
    pose_ids = trajectory_frame_ids(trajectory_path, len(params))
    ids = [int(Path(path).stem) for path in image_paths]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source image frame ID")
    positions = {frame_id: index for index,frame_id in enumerate(pose_ids)}
    missing = sorted(set(ids) - positions.keys())
    if missing:
        raise ValueError(f"Recorded camera trajectory missing RGB-D frame IDs: {missing[:8]}")
    entry = params[0].intrinsic
    w, h = int(entry.width), int(entry.height)
    k = np.asarray(entry.intrinsic_matrix, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0,0], k[1,1]) <= 0:
        raise ValueError("Bad recorded intrinsics")
    ext = []
    for image_id in ids:
        p = params[positions[image_id]]
        x = np.asarray(p.extrinsic, dtype=float)
        if (x.shape != (4,4) or not np.isfinite(x).all() or
                not np.allclose(x[3], [0,0,0,1], atol=1e-5) or
                not np.allclose(x[:3,:3].T @ x[:3,:3],np.eye(3),atol=2e-3) or
                abs(np.linalg.det(x[:3,:3])-1.) > 2e-3):
            raise ValueError(f"Bad recorded world-to-camera extrinsic for image {image_id}")
        ext.append(x)
    return (w,h,float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2])), ext


def save_comparison(path, photo, pixels, title):
    mesh = Image.fromarray(pixels, "RGB")
    combined = Image.new("RGB", (mesh.width * (2 if photo is not None else 1),mesh.height+28), (25,25,25))
    if photo is not None:
        combined.paste(photo.resize(mesh.size), (0,28))
    combined.paste(mesh, ((mesh.width if photo is not None else 0),28))
    ImageDraw.Draw(combined).text((8,8), title,fill="white")
    combined.save(path)


def render_views(dataset, review_indices, transforms, intrinsic, mesh, destination):
    from echo4d.model import Camera
    from tum_graph_visual import rasterize_vertex_rgb,look_at
    visual=destination/"visual"
    visual.mkdir()
    viewmesh=(mesh.simplify_quadric_decimation(30000)
              if len(mesh.triangles)>30000 else mesh)
    xyz=np.asarray(viewmesh.vertices)
    faces=np.asarray(viewmesh.triangles,dtype=np.int32)
    colors=np.asarray(viewmesh.vertex_colors)
    if colors.shape != xyz.shape:
        raise RuntimeError("Stock Open3D preview lost vertex colors")
    rgb=np.rint(np.clip(colors,0,1)*255).astype(np.uint8)
    w,h,fx,fy,cx,cy=intrinsic
    camera=Camera(w,h,fx,fy,cx,cy).scaled(2)
    result=[]
    for frame in (review_indices[0],review_indices[len(review_indices)//2],review_indices[-1]):
        world_pose=np.linalg.inv(transforms[frame])
        image,occupied=rasterize_vertex_rgb(xyz,faces,rgb,camera,world_pose,near=.15,far=6.)
        with Image.open(dataset.color_paths[frame]) as file:
            photo=file.convert("RGB").copy()
        name=f"photo-vs-mesh-{frame:05d}.png"
        save_comparison(visual/name,photo,image,"REAL RGB (left) / OBSERVED MESH (right)")
        result.append({"view":"real camera","file":"visual/"+name,"covered_pixels":occupied})
    for i,(frame,dx) in enumerate(((review_indices[len(review_indices)//3],.35),
                                  (review_indices[2*len(review_indices)//3],-.35)),1):
        anchor=np.linalg.inv(transforms[frame])
        eye=anchor[:3,3]+dx*anchor[:3,0]
        pose=look_at(eye,anchor[:3,3]+1.8*anchor[:3,2])
        image,occupied=rasterize_vertex_rgb(xyz,faces,rgb,camera,pose,near=.15,far=6.)
        name=f"free-view-{i}.png"
        save_comparison(visual/name,None,image,"FREE THIRD PERSON / OBSERVED MESH")
        result.append({"view":"free camera","file":"visual/"+name,"covered_pixels":occupied})
    return result


def run(kind, out, cache, budget, voxel):
    import open3d as o3d
    import trimesh
    if not math.isfinite(voxel) or not .01 <= voxel <= .06:
        raise ValueError("Voxel out of bounds")
    dataset=(o3d.data.SampleRedwoodRGBDImages(data_root=str(cache))
             if kind=="sample" else o3d.data.BedroomRGBDImages(data_root=str(cache)))
    colors=list(dataset.color_paths)
    depths=list(dataset.depth_paths)
    if len(colors)!=len(depths):
        raise ValueError("Unregistered color/depth sequence lengths")
    traj=o3d.io.read_pinhole_camera_trajectory(dataset.trajectory_log_path)
    intrinsic,extrinsics=read_poses(traj,colors,dataset.trajectory_log_path)
    if any(Path(c).stem != Path(d).stem for c,d in zip(colors,depths)):
        raise ValueError("Color/depth frame IDs disagree")
    source_frame_count=len(colors)
    selected=spaced_real_frames(len(colors),min(budget,len(colors)))
    out.mkdir(parents=True,exist_ok=False)
    engine=o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,sdf_trunc=4*voxel,volume_unit_resolution=16,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=1)
    camera=o3d.camera.PinholeCameraIntrinsic(*intrinsic)
    for index in selected:
        rgb=o3d.io.read_image(colors[index])
        depth=o3d.io.read_image(depths[index])
        if (np.asarray(rgb).shape[:2]!=(intrinsic[1],intrinsic[0]) or
                np.asarray(depth).shape[:2]!=(intrinsic[1],intrinsic[0])):
            raise ValueError("Raw images disagree with recorded calibration")
        frame=o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,depth,depth_scale=1000.,depth_trunc=4.5,
            convert_rgb_to_intensity=False)
        engine.integrate(frame,camera,extrinsics[index])
    result=engine.extract_triangle_mesh()
    if not result.has_triangles() or not result.has_vertex_colors():
        raise RuntimeError("Mature Open3D failed to extract observed RGB Mesh")
    result.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(out/"recorded-rgbd-full.ply"),result):
        raise RuntimeError("Native PLY export failed")
    preview=(result.simplify_quadric_decimation(85000)
             if len(result.triangles)>85000 else result)
    vertex_colors=np.asarray(preview.vertex_colors)
    if vertex_colors.shape != np.asarray(preview.vertices).shape:
        raise RuntimeError("GLB preview cannot be claimed colorized")
    rgba=np.column_stack((np.rint(np.clip(vertex_colors,0,1)*255).astype(np.uint8),
                          np.full(len(vertex_colors),255,dtype=np.uint8)))
    trimesh.Trimesh(vertices=np.asarray(preview.vertices),
                    faces=np.asarray(preview.triangles),
                    vertex_colors=rgba,process=False).export(str(out/"recorded-rgbd-preview.glb"))
    prior=None
    supplied=Path(dataset.reconstruction_path)
    if supplied.is_file():
        import shutil
        prior="PREEXISTING-PRIOR-"+supplied.name
        shutil.copy2(supplied,out/prior)
    # Keep 360-frame baseline review cameras fixed when comparing denser fusion.
    fixed_review=spaced_real_frames(source_frame_count,min(360,source_frame_count))
    five=render_views(dataset,fixed_review,extrinsics,intrinsic,result,out)
    report={"scene":("Redwood real bedroom" if kind=="bedroom" else "Redwood 5-frame partial smoke"),
            "source":"Open3D public real RGB-D","geometry_engine":"stock Open3D 0.19 ScalableTSDFVolume",
            "rgbd_frame_count":source_frame_count,"frames_used":len(selected),
            "first_frame_id":int(Path(colors[0]).stem),
            "last_frame_id":int(Path(colors[-1]).stem),
            "trajectory_poses":len(traj.parameters),
            "raw_resolution":list(intrinsic[:2]),"logged_calibration":intrinsic,
            "mesh_triangles":len(result.triangles),"colored_glb":"recorded-rgbd-preview.glb",
            "review_frame_indices":[fixed_review[0],fixed_review[len(fixed_review)//2],fixed_review[-1]],
            "prior_file_NOT_USED_IN_RECONSTRUCTION":prior,"visuals":five,
            "independent_geometry_accuracy":"not tested","issue_74":"FAIL unchanged"}
    (out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["sample","bedroom"],default="bedroom")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--frames",type=int,default=360)
    parser.add_argument("--voxel",type=float,default=.025)
    args=parser.parse_args()
    if not 3 <= args.frames <= 600:
        parser.error("3-600 original RGB-D frames")
    try:
        print(json.dumps(run(args.dataset,args.output,args.cache,args.frames,args.voxel)))
    except Exception as ex:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/"FAILURE.txt").write_text(f"{type(ex).__name__}: {ex}\n")
        raise


if __name__=="__main__":
    main()
