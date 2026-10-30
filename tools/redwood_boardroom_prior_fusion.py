"""Real Redwood Boardroom RGB-D TSDF plus EXPLICIT pre-existing reconstruction PRIOR.
Park, Zhou & Koltun ICCV 2017; original 640x480 Asus Xtion sensor images.
The supplied scan/reconstruction is not independently recovered from these frames.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import traceback
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from PIL import Image
from echo4d.model import Camera, Frame, DataError
from tum_xyz_open3d_320 import fuse, export_glb, render, free_view, panel

PAGE = "https://www.redwood-data.org/indoor_lidar_rgbd/download.html"
CAMERA = Camera(320, 240, 262.5, 262.5, 159.75, 119.75)
LIMIT = {"rgbd": 6_000_000_000, "prior": 600_000_000, "poses": 80_000_000}


class TableLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None
    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = {"text": "", "links": []}
        elif tag == "a" and self.cell is not None:
            href = dict(attrs).get("href")
            if href:
                self.cell["links"].append(href)
    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data
    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.row is not None and self.cell is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def official_assets(html, page=PAGE):
    parser = TableLinks()
    parser.feed(html)
    rows = [r for r in parser.rows if r and "boardroom" in r[0]["text"].lower()]
    if len(rows) != 1 or len(rows[0]) < 7:
        raise DataError("Official Boardroom table row missing")
    row, assets = rows[0], {}
    for role, col in (("rgbd", 3), ("prior", 5), ("poses", 6)):
        links = row[col]["links"]
        if not links:
            raise DataError("Missing Boardroom official " + role)
        if role == "poses":
            links = sorted(links, key=lambda x: (".log" not in x.lower(), x))
        url = urljoin(page, links[0])
        parsed = urlparse(url)
        if parsed.scheme == "http" and parsed.hostname in ("redwood-data.org", "www.redwood-data.org"):
            url = parsed._replace(scheme="https").geturl()
            parsed = urlparse(url)
        if parsed.scheme != "https":
            raise DataError("Official asset is not HTTPS: " + url)
        assets[role] = url
    return assets


def official_page():
    for url in (PAGE, PAGE.replace("www.", "")):
        try:
            with urlopen(Request(url, headers={"User-Agent": "Echo4D/1.0"}), timeout=90) as r:
                return r.read(2_000_000).decode("utf-8")
        except Exception as e:
            print("Official index attempt:", str(e), flush=True)
    raise DataError("Official Boardroom index unavailable")


def download(role, url, path):
    with urlopen(Request(url, headers={"User-Agent": "Echo4D/1.0"}), timeout=180) as r:
        if urlparse(r.geturl()).scheme != "https":
            raise DataError("Non-HTTPS redirect for " + role)
        n = 0
        with path.open("xb") as out:
            while chunk := r.read(2 * 1024 * 1024):
                n += len(chunk)
                if n > LIMIT[role]:
                    raise DataError("Oversized " + role)
                out.write(chunk)
                if n // 200_000_000 != (n-len(chunk)) // 200_000_000:
                    print(role, n, "bytes downloaded", flush=True)
    return n


def unpack_one(path, extension, target):
    if not zipfile.is_zipfile(path):
        if path.stat().st_size < 100:
            raise DataError("Empty official asset")
        shutil.copyfile(path, target)
        return target
    with zipfile.ZipFile(path) as z:
        matches = [i for i in z.infolist() if i.filename.lower().endswith(extension)
                   and not i.is_dir() and not PurePosixPath(i.filename).is_absolute()
                   and ".." not in PurePosixPath(i.filename).parts]
        if not matches:
            raise DataError("No " + extension + " in official archive")
        chosen = max(matches, key=lambda x: x.file_size)
        if chosen.file_size > 1_000_000_000:
            raise DataError("Oversized prior/pose")
        with z.open(chosen) as src, target.open("xb") as out:
            shutil.copyfileobj(src, out)
    return target


def read_log(text):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    poses = {}
    for start in range(0, len(lines)-4, 5):
        head = lines[start].split()
        if len(head) != 3:
            raise DataError("Bad Redwood log header")
        try:
            idx = int(head[0])
            mat = np.array([[float(x) for x in lines[start+k].split()]
                            for k in range(1, 5)], dtype=float)
        except (ValueError, IndexError) as e:
            raise DataError("Bad Redwood log pose") from e
        if (mat.shape != (4, 4) or not np.isfinite(mat).all()
            or not np.allclose(mat[3], [0, 0, 0, 1], atol=1e-4)
            or not np.allclose(mat[:3,:3].T @ mat[:3,:3], np.eye(3), atol=.01)):
            continue
        if idx in poses:
            raise DataError("Duplicate Redwood pose")
        poses[idx] = mat
    if len(poses) < 100:
        raise DataError("Too few official poses")
    return poses


def index_zip(z):
    groups = {}
    for info in z.infolist():
        if info.is_dir() or info.file_size > 8_000_000:
            continue
        name = PurePosixPath(info.filename).name
        m = re.match(r"^(\d{5,7})[-_]", name)
        if not m:
            continue
        ext = PurePosixPath(name).suffix.lower()
        role = "color" if ext in (".jpg", ".jpeg") else "depth" if ext == ".png" else None
        if role:
            group = groups.setdefault(int(m.group(1)), {})
            if role in group:
                raise DataError("Duplicate RGB-D image")
            group[role] = info.filename
    return {i: pair for i, pair in groups.items() if set(pair) == {"color", "depth"}}


def choose(available, count=240):
    if len(available) < 100:
        raise DataError("Insufficient paired real depth and camera poses")
    return [available[int(i)] for i in np.unique(
        np.linspace(0, len(available)-1, min(count, len(available)), dtype=int))]


def read_frames(z, index, poses, ids):
    frames, originals = [], {}
    for n, i in enumerate(ids):
        with z.open(index[i]["color"]) as stream, Image.open(stream) as image:
            color = image.convert("RGB")
            if color.size != (640, 480):
                raise DataError("RGB not 640x480")
            if n in (0, len(ids)//2, len(ids)-1):
                originals[i] = color.copy()
            rgb = np.asarray(color, dtype=np.uint8)[::2,::2].copy()
        with z.open(index[i]["depth"]) as stream, Image.open(stream) as image:
            depth = np.asarray(image)
        if depth.shape != (480, 640) or depth.dtype.kind not in "ui":
            raise DataError("Depth not original 16-bit 640x480")
        meters = depth[::2,::2].astype(np.float32) * .001
        meters[(meters < .2) | (meters > 4.)] = 0
        frames.append(Frame(t=i/30, camera_to_world=poses[i],
                            depth=meters, color=rgb).validate(CAMERA))
        if (n+1) % 40 == 0:
            print("Loaded measured RGB-D", n+1, "of", len(ids), flush=True)
    return frames, originals


def sampled_world(frames, invert=False):
    xyzs = []
    for f in frames[::max(1, len(frames)//8)]:
        yy, xx = np.mgrid[0:240:12, 0:320:12]
        d = f.depth[::12,::12]
        ok = (d > .3) & (d < 4.)
        xyz = np.stack(((xx[ok]-CAMERA.cx)*d[ok]/CAMERA.fx,
                        (yy[ok]-CAMERA.cy)*d[ok]/CAMERA.fy, d[ok]), axis=-1)
        pose = np.linalg.inv(f.camera_to_world) if invert else f.camera_to_world
        xyzs.append(xyz @ pose[:3,:3].T + pose[:3,3])
    return np.concatenate(xyzs)


def align_pose_convention(frames, prior):
    from scipy.spatial import cKDTree
    xyz = np.asarray(prior.vertices)
    if len(xyz) > 100_000:
        xyz = xyz[np.linspace(0, len(xyz)-1, 100_000, dtype=int)]
    tree = cKDTree(xyz)
    distances = [float(np.median(tree.query(sampled_world(frames, inv), workers=2)[0]))
                 for inv in (False, True)]
    inv = int(np.argmin(distances))
    if distances[inv] > .35:
        raise DataError("Prior and official poses do not register: " + str(distances))
    if inv:
        frames = [replace(f, camera_to_world=np.linalg.inv(f.camera_to_world)) for f in frames]
    return frames, {"inverted_official_log": bool(inv), "median_prior_distance_m": distances}


def hybrid_prior(measured, prior, o3d):
    from scipy.spatial import cKDTree
    if len(prior.triangles) > 450_000:
        prior = prior.simplify_quadric_decimation(450_000)
    p = np.asarray(prior.vertices)
    m = np.asarray(measured.vertices)
    if not len(p) or not len(m):
        raise DataError("Empty measured or PRIOR mesh")
    dist, nearest = cKDTree(m).query(p, workers=2)
    if prior.has_vertex_colors():
        rgb = np.asarray(prior.vertex_colors).copy()
    else:
        rgb = np.tile([.53,.56,.59], (len(p),1))
        supported = dist < .10
        rgb[supported] = np.asarray(measured.vertex_colors)[nearest[supported]]
    prior.vertex_colors = o3d.utility.Vector3dVector(np.clip(rgb,0,1))
    faces = np.asarray(prior.triangles)
    missing = np.min(dist[faces], axis=1) > .055
    if not missing.any():
        raise DataError("No prior-only gap geometry")
    used, inverse = np.unique(faces[missing], return_inverse=True)
    supplement = o3d.geometry.TriangleMesh()
    supplement.vertices = o3d.utility.Vector3dVector(p[used])
    supplement.triangles = o3d.utility.Vector3iVector(inverse.reshape(-1,3))
    supplement.vertex_colors = o3d.utility.Vector3dVector(rgb[used])
    combined = measured + supplement
    source = o3d.geometry.TriangleMesh(combined)
    colors = np.empty((len(combined.vertices),3))
    colors[:len(measured.vertices)] = [.18,.82,.37]
    colors[len(measured.vertices):] = [.98,.48,.16]
    source.vertex_colors = o3d.utility.Vector3dVector(colors)
    return prior, combined, source, {
        "measured_faces": len(measured.triangles),
        "prior_total_faces": len(faces),
        "prior_gap_faces_added": int(missing.sum()),
        "hybrid_faces": len(combined.triangles)}


def run(work, output, result):
    import open3d as o3d
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    assets = official_assets(official_page())
    result["official_assets"] = assets
    paths = {}
    for role in ("poses", "prior", "rgbd"):
        path = work / (role + ".download")
        result.setdefault("download_bytes", {})[role] = download(role, assets[role], path)
        paths[role] = path
    pose_path = unpack_one(paths["poses"], ".log", work/"official-poses.log")
    prior_path = unpack_one(paths["prior"], ".ply", work/"official-PRIOR.ply")
    poses = read_log(pose_path.read_text())
    prior = o3d.io.read_triangle_mesh(str(prior_path))
    if len(prior.triangles) < 1000:
        raise DataError("Official PRIOR has no triangle surface")
    with zipfile.ZipFile(paths["rgbd"]) as z:
        index = index_zip(z)
        ids = choose(sorted(set(index) & set(poses)))
        frames, originals = read_frames(z, index, poses, ids)
    result["measured_frames"] = len(frames)
    result["source_geometry"] = "Original 640x480 measured RGB-D and official estimated camera poses"
    result["prior_role"] = "Official pre-existing Boardroom reconstructed mesh; all supplemental faces PRIOR"
    frames, check = align_pose_convention(frames, prior)
    result["registration"] = check
    measured, _ = fuse(o3d, frames, CAMERA, pixel_stride=1, voxel=.025)
    prior, combined, source, counts = hybrid_prior(measured, prior, o3d)
    result["geometry"] = counts
    for name, mesh in (("measured-only.glb", measured),
                       ("PRIOR-assisted.glb", combined),
                       ("PRIOR-provenance.glb", source),
                       ("PRIOR-only.glb", prior)):
        export_glb(mesh, output/name)
    panels, views = [], []
    for n in (0, len(ids)//2, len(ids)-1):
        i = ids[n]
        photo = originals[i]
        photo.save(output/f"real-boardroom-{i:05d}-640x480.png")
        real = np.asarray(photo.resize((320,240), Image.Resampling.LANCZOS))
        pose = frames[n].camera_to_world
        a, b = render(o3d, measured, CAMERA, pose), render(o3d, combined, CAMERA, pose)
        name = f"recorded-{i:05d}.png"
        panels.append(panel(output/name, [real,a,b],
                            ["REAL RGB", "MEASURED TSDF", "PRIOR+MEASURED"]))
        views.append({"kind":"recorded","frame":i,"file":name})
    for n, side in ((len(ids)//3,-1.),(len(ids)*2//3,1.)):
        pose = free_view(frames[n].camera_to_world,side)
        a, b = render(o3d, measured, CAMERA, pose), render(o3d, combined, CAMERA, pose)
        name = f"free-{int(side):+d}.png"
        panels.append(panel(output/name, [np.full_like(a,20),a,b],
                            ["FREE / NO PHOTO","MEASURED TSDF","PRIOR+MEASURED"]))
        views.append({"kind":"free","frame":ids[n],"file":name,
                      "camera_shift_m":.22,"yaw_deg":12*side})
    board = Image.new("RGB",(960,267*len(panels)))
    for n, image in enumerate(panels):
        board.paste(image,(0,267*n))
    board.save(output/"boardroom-five-view-measured-vs-PRIOR.png")
    result.update(views=views, visual_verdict="PENDING human review",
                  free_orbit="FAIL until visually accepted",
                  issue_74="FAIL unchanged",
                  historical_desk="FAIL 16.7779cm > 15cm unchanged",
                  provenance="green measured sensor TSDF; orange pre-existing PRIOR")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result={"scene":"Real Redwood Boardroom","source":PAGE,"status":"running",
            "run_url":(f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                       f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None)}
    try:
        run(args.work,args.output,result)
        result["status"]="real_mesh_and_five_views_exported_review_pending"
        code=0
    except Exception as e:
        traceback.print_exc()
        result.update(status="failed",error=f"{type(e).__name__}: {e}")
        code=1
    finally:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({"status":result["status"],"geometry":result.get("geometry"),
                          "error":result.get("error")}),flush=True)
    return code

if __name__=="__main__":
    raise SystemExit(main())
