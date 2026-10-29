"""Real Stanford Lounge 3000-frame observed Mesh: conservative disconnected speck cleanup.

No RGB/depth/pose substitution and no pre-existing scan. The input is the
actual Open3D full reconstruction (estimated poses) from run 40877823227.
Compare original and edited geometry at identical five camera rays.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Pre-registered conservative rule: BOTH criteria must hold.
MAX_SPECK_FACES = 128
MAX_SPECK_AREA_M2 = 0.0025


def camera_rays(o3d, mesh, poses):
    k = np.array([[525., 0, 319.5], [0, 525., 239.5], [0, 0, 1.]])
    small = k.copy()
    small[:2] /= 2
    small[0, 2] = (k[0, 2] + .5) / 2 - .5
    small[1, 2] = (k[1, 2] + .5) / 2 - .5
    out = []
    indices = (0, len(poses)//2, len(poses)-1)
    for i in indices:
        ext = np.linalg.inv(poses[i])
        out.append((f"recorded-{i:05d}.png",
            o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                o3d.core.Tensor(small.astype(np.float32)),
                o3d.core.Tensor(ext.astype(np.float32)), 320, 240)))
    center = (mesh.get_min_bound() + mesh.get_max_bound()) / 2
    span = mesh.get_max_bound() - mesh.get_min_bound()
    for j, offset in enumerate(([-.5,-.9,-.7],[.8,-1.2,.05]), 1):
        eye = center + span * np.asarray(offset)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            78., o3d.core.Tensor(center.astype(np.float32)),
            o3d.core.Tensor(eye.astype(np.float32)),
            o3d.core.Tensor(np.array([0.,-1.,0.],np.float32)), 320, 240)
        out.append((f"free-{j}.png", rays))
    return out


def render(o3d, mesh, rays):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    frames, coverage = {}, {}
    for name, ray in rays:
        hit = scene.cast_rays(ray)
        ids = hit["primitive_ids"].numpy().reshape(-1)
        uv = hit["primitive_uvs"].numpy().reshape(-1, 2)
        depth = hit["t_hit"].numpy().reshape(-1)
        valid = np.isfinite(depth) & (depth > .15) & (depth < 16.) & (ids < len(faces))
        rgb = np.zeros((320*240,3),np.uint8)
        if valid.any():
            weights = np.column_stack((1-uv[valid].sum(axis=1),uv[valid]))
            values = (colors[faces[ids[valid].astype(np.int64)]] * weights[:,:,None]).sum(axis=1)
            rgb[valid] = np.clip(values*255,0,255).astype(np.uint8)
        frames[name] = Image.fromarray(rgb.reshape(240,320,3))
        coverage[name] = float(valid.mean())
    del scene
    gc.collect()
    return frames, coverage


def load_poses(path):
    lines = path.read_text().splitlines()
    if len(lines) % 5:
        raise ValueError("Incomplete estimated camera trajectory")
    poses = []
    for j in range(0,len(lines),5):
        if list(map(int,lines[j].split())) != [j//5,j//5,j//5+1]:
            raise ValueError("Unexpected camera trajectory frame order")
        p = np.array([float(v) for row in lines[j+1:j+5] for v in row.split()]).reshape(4,4)
        if not np.isfinite(p).all() or not np.allclose(p[3],[0,0,0,1]):
            raise ValueError("Invalid estimated camera pose")
        poses.append(p)
    return poses


def export_glb(mesh, path):
    import trimesh
    color = np.clip(np.asarray(mesh.vertex_colors)*255,0,255).astype(np.uint8)
    model = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),vertex_colors=color,process=False)
    model.export(path)


def run(source, output):
    import open3d as o3d
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Use fresh output directory")
    output.mkdir(parents=True)
    mesh = o3d.io.read_triangle_mesh(str(source/"scene"/"integrated.ply"))
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise ValueError("Recovered full room has no observed colored Mesh")
    poses = load_poses(source/"scene"/"trajectory.log")
    if len(poses) != 3000:
        raise ValueError(f"Expected 3000 upstream registered frames, got {len(poses)}")
    original_faces = len(mesh.triangles)
    labels, counts, areas = mesh.cluster_connected_triangles()
    labels = np.asarray(labels, dtype=np.int32)
    counts = np.asarray(counts)
    areas = np.asarray(areas)
    remove_cluster = (counts <= MAX_SPECK_FACES) & (areas <= MAX_SPECK_AREA_M2)
    remove_mask = remove_cluster[labels]
    removed_faces = int(remove_mask.sum())
    removed_components = int(remove_cluster.sum())
    # Render the unmodified source FIRST; no camera motion allowed between A and B.
    rays = camera_rays(o3d,mesh,poses)
    before, cov_before = render(o3d,mesh,rays)
    mesh.remove_triangles_by_mask(remove_mask.tolist())
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise RuntimeError("Cleanup destroyed real observed geometry")
    after, cov_after = render(o3d,mesh,rays)
    export_glb(mesh,output/"observed-3000-speck-filtered.glb")
    preview = mesh.simplify_quadric_decimation(300000) if len(mesh.triangles)>300000 else mesh
    export_glb(preview,output/"observed-3000-speck-filtered-preview.glb")
    names = [name for name,_ in rays]
    canvas = Image.new("RGB",(960,34+5*268),(19,19,19))
    draw = ImageDraw.Draw(canvas)
    for i,name in enumerate(names):
        y = 34+i*268
        draw.text((5,y+5),name,fill="white")
        if i < 3:
            with Image.open(source/"views"/name) as original:
                if original.size != (640,268):
                    raise ValueError(f"Unexpected real view layout {name}: {original.size}")
                canvas.paste(original.convert("RGB").crop((0,28,320,268)),(0,y+28))
        canvas.paste(before[name],(320,y+28))
        canvas.paste(after[name],(640,y+28))
    draw.text((5,10),"REAL RGB / free view",fill="white")
    draw.text((325,10),"3000 observed: BEFORE",fill="white")
    draw.text((645,10),"3000 observed: AFTER speck filter",fill="white")
    canvas.save(output/"five-view-observed-ab.png")
    receipt = {
        "source":"Stanford Lounge original 640x480 RGB-D",
        "source_run":"https://github.com/REDrighthand05/echo4d/actions/runs/40877823227",
        "source_kind":"Open3D complete measured RGB-D reconstruction, estimated poses",
        "registered_frames":len(poses),
        "existing_mesh_prior":False,"predicted_depth":False,
        "geometry_change":"remove only tiny disconnected observed mesh components; no hole filling",
        "max_faces_per_removed_component":MAX_SPECK_FACES,
        "max_area_m2_per_removed_component":MAX_SPECK_AREA_M2,
        "original_faces":original_faces,"filtered_faces":len(mesh.triangles),
        "removed_faces":removed_faces,"removed_components":removed_components,
        "coverage_before":cov_before,"coverage_after":cov_after,
        "visual_acceptance":"NOT REVIEWED; #74 remains FAIL",
        "interpretation":"Coverage loss is expected; accept only if artifacts visibly disappear without losing real objects."
    }
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    run(args.source,args.output)
