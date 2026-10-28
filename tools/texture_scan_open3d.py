"""Texture an existing measured scan with calibrated RGB via Open3D's CPU baker.

No depth inference, hole filling, geometry optimization or camera optimization.
Manifest uses ECHO intrinsics and column-major cameraToWorld, RGB paths relative
its directory. Output GLB embeds the actual baked image and per-corner UVs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def texture_scan(mesh_path, manifest_path, output, texture_size=2048):
    import open3d as o3d
    import trimesh
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(manifest_path).resolve()
    spec = json.loads(manifest_path.read_text())
    cam = spec['intrinsics']
    K = np.array([[cam['fx'],0,cam['cx']], [0,cam['fy'],cam['cy']], [0,0,1]], np.float64)
    images, intrinsics, extrinsics, inputs = [], [], [], []
    for f in spec['frames']:
        path = (manifest_path.parent/f['color']).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError('RGB path leaves manifest directory')
        if sha(path) != f['rgb_sha256']:
            raise ValueError('RGB checksum mismatch')
        rgb = np.asarray(Image.open(path).convert('RGB'))
        if rgb.shape != (cam['height'],cam['width'],3):
            raise ValueError('RGB dimensions differ from calibration')
        pose = np.asarray(f['cameraToWorld'],np.float64).reshape(4,4,order='F')
        if (not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1])
            or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=2e-3)
            or np.linalg.det(pose[:3,:3]) < .99):
            raise ValueError('Expected rigid camera-to-world pose')
        images.append(o3d.t.geometry.Image(np.ascontiguousarray(rgb)))
        intrinsics.append(o3d.core.Tensor(K))
        extrinsics.append(o3d.core.Tensor(np.linalg.inv(pose)))
        inputs.append({'color':f['color'],'sha256':sha(path),'source_index':f.get('source_index')})
    if not images:
        raise ValueError('No calibrated RGB images')
    original = o3d.io.read_triangle_mesh(str(mesh_path))
    positions = np.asarray(original.vertices).copy()
    triangles = np.asarray(original.triangles).copy()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(original)
    start = time.perf_counter()
    print('UV atlas',len(triangles),'triangles',flush=True)
    atlas = mesh.compute_uvatlas(size=texture_size,parallel_partitions=4,nthreads=4)
    print('Projecting',len(images),'real images',flush=True)
    albedo = mesh.project_images_to_albedo(images,intrinsics,extrinsics,tex_size=texture_size)
    pixels = albedo.as_tensor().numpy()
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels*255,0,255).astype(np.uint8)
    image = Image.fromarray(pixels)
    image.save(output/'albedo.png')
    # Split only vertex indexing at UV seams. Triangle positions stay unchanged.
    uv = mesh.triangle.texture_uvs.numpy()
    vertices = positions[triangles.reshape(-1)]
    faces = np.arange(len(vertices)).reshape(-1,3)
    visual = trimesh.visual.TextureVisuals(uv=uv.reshape(-1,2), image=image)
    result = trimesh.Trimesh(vertices=vertices,faces=faces,visual=visual,process=False)
    result.export(output/'room-textured.glb')
    restored = trimesh.load(output/'room-textured.glb',force='mesh',process=False)
    assert np.allclose(restored.vertices[restored.faces],positions[triangles],rtol=0,atol=1e-6)
    np.savez_compressed(output/'surface-uv.npz',positions=positions,triangles=triangles,uv=uv)
    report = {'provider':'Open3D project_images_to_albedo','version':o3d.__version__,
              'mesh_sha256':sha(mesh_path),'manifest_sha256':sha(manifest_path),
              'geometry_modified':False,'texture_size':texture_size,'image_count':len(images),
              'source':'existing measured-depth scan + calibrated real RGB',
              'unknown_regions':'No geometry added; unpainted surface may appear black',
              'vertices':len(positions),'faces':len(triangles),'inputs':inputs,
              'seconds':time.perf_counter()-start,'atlas_result':list(atlas),
              'glb_sha256':sha(output/'room-textured.glb'),'albedo_sha256':sha(output/'albedo.png')}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--texture-size',type=int,default=2048,choices=(1024,2048,4096))
    a=p.parse_args()
    texture_scan(a.mesh,a.manifest,a.output,a.texture_size)
