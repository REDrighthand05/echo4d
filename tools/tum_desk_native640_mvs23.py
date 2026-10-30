"""TUM Desk: replace one AI depth with real RGB multi-view geometry at fixed poses.

Stacked on PR #171. Target measured depths are excluded from inference and
TSDF, and opened only after both GLBs are exported for post-hoc checking.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from echo4d.model import Camera, DataError, Frame, load_recording, rigid_matrix, safe_path
from tum_desk_sfm import URL, ROOT, get_archive, extract_selected
from tum_desk_full_mesh import extract_sensor_depths
from tum_public_roundtrip import invoke
from tum_desk_known_pose_open3d import MAX_IMPORTED, RECORDED, FREE, FR1_INTRINSICS, split_indices
from tum_desk_native640_depth_ladder import nested_measured_indices
from tum_desk_native640_tsdf import NATIVE_VOXEL_M, MAX_TRIANGLES, recorded_poses
from tum_xyz_open3d_320 import fuse, export_glb, render, panel, free_view
from tum_desk_rgb_mvs import estimate_target

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REV = "32d03942121d29edb49de4e2cc15831558af3f36"
WEIGHTS_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"


def select_rgb_targets(training, measured, count=12):
    """Time-spread, disjoint RGB targets selected without inspecting depth."""
    if tuple(sorted(set(training))) != tuple(training) or tuple(sorted(set(measured))) != tuple(measured):
        raise ValueError("Frame indices must be sorted and unique")
    if not set(measured).issubset(training):
        raise ValueError("Measured cohort not in training")
    remaining = tuple(i for i in training if i not in set(measured))
    if count < 2 or count > len(remaining):
        raise ValueError("Insufficient RGB targets")
    selected = tuple(remaining[round(j * (len(remaining)-1)/(count-1))] for j in range(count))
    if len(set(selected)) != count or set(selected) & set(measured):
        raise ValueError("RGB target selection collision")
    return selected


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CPUDepthAnything:
    """Previously executed Apache-2.0 CPU model from PR #165, SHA-pinned."""
    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        torch.set_num_threads(2)
        path = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REV,
            allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"]))
        if sha256(path / "model.safetensors") != WEIGHTS_SHA256:
            raise RuntimeError("Model weights SHA256 mismatch")
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(str(path))
        self.model = AutoModelForDepthEstimation.from_pretrained(
            str(path), use_safetensors=True).to("cpu").eval()

    def infer(self, rgb):
        import torch.nn.functional as F
        batch = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        with self.torch.inference_mode():
            prediction = self.model(**batch).predicted_depth.unsqueeze(1)
            prediction = F.interpolate(prediction, size=rgb.shape[:2],
                                       mode="bicubic", align_corners=False)
        output = prediction[0, 0].cpu().numpy().astype(np.float32)
        if not np.isfinite(output).all():
            raise RuntimeError("Nonfinite relative depth")
        return output


def project_measured_depth(frames, camera, target_pose, stride=4):
    """Project only retained sensor frames, z-buffering in the target RGB view."""
    h, w = camera.height, camera.width
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    xx, yy = xx.ravel(), yy.ravel()
    result = np.full(h*w, np.inf, np.float32)
    target_inverse = np.linalg.inv(target_pose)
    for frame in frames:
        d = frame.depth[::stride, ::stride].ravel()
        valid = np.isfinite(d) & (d > .4) & (d < 3.8)
        if not np.any(valid):
            continue
        z = d[valid]
        points = np.stack(((xx[valid]-camera.cx)*z/camera.fx,
                           (yy[valid]-camera.cy)*z/camera.fy, z), axis=1)
        transform = target_inverse @ frame.camera_to_world
        points = points @ transform[:3,:3].T + transform[:3,3]
        zz = points[:,2]
        good = (zz > .4) & (zz < 3.8)
        points, zz = points[good], zz[good]
        u = np.rint(points[:,0]*camera.fx/zz+camera.cx).astype(np.int32)
        v = np.rint(points[:,1]*camera.fy/zz+camera.cy).astype(np.int32)
        inside = (u>=0)&(u<w)&(v>=0)&(v<h)
        np.minimum.at(result, v[inside]*w+u[inside], zz[inside])
    return result.reshape(h,w)


def calibrate_relative_depth(relative, projected, support_radius=90):
    """Robust inverse-depth metric fit from retained sensor reprojections only."""
    if relative.ndim != 2 or relative.shape != projected.shape:
        raise ValueError("Mismatched depth image shapes")
    anchors = np.isfinite(projected) & np.isfinite(relative) & (relative > 0)
    y,x = np.where(anchors)
    if len(x) < 200:
        raise RuntimeError(f"Too few measured projections: {len(x)}")
    x,y = x[::3],y[::3]
    z = projected[y,x].astype(np.float64)
    p = relative[y,x].astype(np.float64)
    if np.std(p) < 1e-5:
        raise RuntimeError("Constant relative depth")
    A = np.column_stack((p,np.ones_like(p)))
    target = 1/z
    weights = np.ones_like(z)
    for _ in range(7):
        coef = np.linalg.lstsq(A*weights[:,None],target*weights,rcond=None)[0]
        residual = A@coef-target
        robust = max(float(np.median(np.abs(residual)))*1.4826,.02)
        weights = np.clip(1.5*robust/np.maximum(np.abs(residual),1e-8),.05,1)
    if not np.isfinite(coef).all() or coef[0] <= 0:
        raise RuntimeError("Inverse-depth scale fit failed")
    estimate = 1/np.maximum(A@coef,1e-6)
    inliers = np.abs(estimate-z) < np.maximum(.28,.22*z)
    if inliers.sum() < 150 or inliers.mean() < .45:
        raise RuntimeError(f"Poor metric support: {inliers.sum()}/{len(z)}")
    coef = np.linalg.lstsq(A[inliers],target[inliers],rcond=None)[0]
    if coef[0] <= 0:
        raise RuntimeError("Refitted scale invalid")
    denominator = relative.astype(np.float64)*coef[0]+coef[1]
    valid = (distance_transform_edt(~anchors) <= support_radius) & (denominator>1/3.8) & (denominator<1/.4)
    metric = np.zeros(relative.shape,np.float32)
    metric[valid] = (1/denominator[valid]).astype(np.float32)
    if valid.mean() < .02:
        raise RuntimeError("Less than 2% calibrated depth")
    return metric, {"projected_sensor_pixels":int(anchors.sum()),
                    "calibration_inliers":int(inliers.sum()),
                    "valid_fraction":float(valid.mean()),
                    "scale_source":"24 retained TUM sensor depths and TUM reference poses"}


def run(args, result):
    import open3d as o3d
    import trimesh
    args.work.mkdir(parents=True,exist_ok=False)
    archive = args.work/"desk-source.tgz"
    get_archive(archive,result,offline=args.archive)
    root = args.work/"dataset"
    result["rgb_pose_extraction"] = extract_selected(archive,root)
    result["measured_depth_extraction"] = extract_sensor_depths(archive,root)
    imported = args.work/"imported-native"
    invoke("import-TUM-Desk-native640-24-plus12",[
        sys.executable,"-m","echo4d","import-tum",str(root/ROOT),
        "--output",str(imported),"--intrinsics",*map(str,FR1_INTRINSICS),
        "--stride","8","--limit",str(MAX_IMPORTED),
        "--downsample","1","--world-up","z"],result,timeout=1200)
    recording = load_recording(imported/"recording.json")
    camera = recording.camera
    if camera.to_dict() != Camera(640,480,*FR1_INTRINSICS).to_dict():
        raise DataError("Native camera calibration differs from PR #171")
    entries = list(recording.entries)
    train,holdout = split_indices(len(entries))
    measured_ids = nested_measured_indices(train)[24]
    targets = select_rgb_targets(train,measured_ids)
    poses = recorded_poses(entries)
    result.update(source_scene="TUM Freiburg1 Desk",source_rgb_frames=len(entries),
        reference_poses="TUM provided, not estimated",
        metric_scale="24 measured TUM sensor depths in metres; AI predicts relative depth",
        camera=camera.to_dict(),voxel_m=NATIVE_VOXEL_M,
        measured_indices=list(measured_ids),rgb_only_indices=list(targets),
        holdout_indices=list(holdout),mesh_prior=False,
        target_sensor_depth_used_for_inference=False,
        checkpoint=MODEL_ID,checkpoint_revision=MODEL_REV,
        weights_sha256=WEIGHTS_SHA256,checkpoint_license="Apache-2.0",
        gpu=False,views=[],meshes={},predictions=[])
    recording.entries = [entries[i] for i in measured_ids]
    measured = [frame for _,frame,_ in recording.iter_frames()]
    if len(measured) != 24:
        raise DataError("Expected 24 sensor RGB-D frames")
    recording.entries = entries
    args.output.mkdir(parents=True,exist_ok=True)
    render_camera = camera.scaled(2)
    rendered = {}
    control,_ = fuse(o3d,measured,camera,pixel_stride=1,voxel=NATIVE_VOXEL_M)
    if not 100 <= len(control.triangles) <= MAX_TRIANGLES:
        raise RuntimeError("Invalid 24-measured Mesh")
    export_glb(control,args.output/"desk-measured24-control.glb")
    rendered["control"] = [render(o3d,control,render_camera,poses[i]) for i in RECORDED]
    rendered["control"].extend(render(o3d,control,render_camera,free_view(poses[i],side)) for side,i in FREE)
    result["meshes"]["control"] = {"glb":"desk-measured24-control.glb","measured":24,"predicted":0}
    del control
    gc.collect()
    predictor = CPUDepthAnything()
    inferred = []
    predictions_by_index = {}
    for index in targets:
        spec = entries[index]
        # Read target RGB only. No target sensor depth enters inference or fusion.
        with Image.open(safe_path(recording.root,spec["color"])) as image:
            rgb = np.asarray(image.convert("RGB"),dtype=np.uint8)
        if rgb.shape != (camera.height,camera.width,3):
            raise DataError(f"RGB target {index} shape mismatch")
        pose = rigid_matrix(spec["cameraToWorld"])
        relative = predictor.infer(rgb)
        projected = project_measured_depth(measured,camera,pose)
        try:
            metric,details = calibrate_relative_depth(relative,projected)
        except RuntimeError as exc:
            result["predictions"].append({"index":index,"used":False,"reason":str(exc)})
            print(f"Skipped target {index}: {exc}",flush=True)
            continue
        inferred.append(Frame(float(spec["t"]),pose,metric,rgb).validate(camera))
        predictions_by_index[index] = metric
        Image.fromarray(np.clip(metric*1000,0,65535).astype(np.uint16)).save(
            args.output/f"predicted-{index:02d}-millimetres.png")
        result["predictions"].append({"index":index,"used":True,**details})
        print(f"Target {index}: {details['valid_fraction']:.1%} anchored",flush=True)
    del predictor
    gc.collect()
    if len(inferred) < 4:
        raise RuntimeError(f"Only {len(inferred)} calibrated RGB targets")
    hybrid,_ = fuse(o3d,measured+inferred,camera,pixel_stride=1,voxel=NATIVE_VOXEL_M)
    if not 100 <= len(hybrid.triangles) <= MAX_TRIANGLES:
        raise RuntimeError("Invalid hybrid Mesh")
    export_glb(hybrid,args.output/"desk-measured24-plus-ai.glb")
    rendered["hybrid"] = [render(o3d,hybrid,render_camera,poses[i]) for i in RECORDED]
    rendered["hybrid"].extend(render(o3d,hybrid,render_camera,free_view(poses[i],side)) for side,i in FREE)
    result["meshes"]["hybrid"] = {"glb":"desk-measured24-plus-ai.glb","measured":24,"predicted":len(inferred)}
    del hybrid
    gc.collect()
    # Geometry-only ablation: replace exactly one AI-predicted target by
    # multi-view RGB triangulation. The target sensor depth is not loaded.
    target_index = 23
    if target_index not in predictions_by_index:
        raise RuntimeError("Target 23 AI baseline unavailable")
    metric_mvs, mvs_receipt = estimate_target(entries, recording, camera,
                                              target_index, train)
    result["mvs"] = mvs_receipt
    if mvs_receipt["valid_fraction"] < .005:
        raise RuntimeError("RGB multi-view support <0.5%; no false Mesh success")
    Image.fromarray(np.clip(metric_mvs*1000,0,65535).astype(np.uint16)).save(
        args.output/"mvs-23-millimetres.png")
    used_indices = [p["index"] for p in result["predictions"] if p["used"]]
    if len(used_indices) != len(inferred):
        raise RuntimeError("Prediction cohort mismatch")
    spec23 = entries[target_index]
    with Image.open(safe_path(recording.root,spec23["color"])) as im:
        rgb23 = np.asarray(im.convert("RGB"),dtype=np.uint8)
    frame23 = Frame(float(spec23["t"]),rigid_matrix(spec23["cameraToWorld"]),
                    metric_mvs,rgb23).validate(camera)
    candidate = [frame for idx,frame in zip(used_indices,inferred)
                 if idx != target_index] + [frame23]
    changed,_ = fuse(o3d,measured+candidate,camera,pixel_stride=1,
                     voxel=NATIVE_VOXEL_M)
    if not 100 <= len(changed.triangles) <= MAX_TRIANGLES:
        raise RuntimeError("Invalid MVS-replacement Mesh")
    export_glb(changed,args.output/"desk-measured24-plus11ai-mvs23.glb")
    rendered["mvs"] = [render(o3d,changed,render_camera,poses[i]) for i in RECORDED]
    rendered["mvs"].extend(render(o3d,changed,render_camera,
                                  free_view(poses[i],side)) for side,i in FREE)
    result["meshes"]["mvs"] = {
        "glb":"desk-measured24-plus11ai-mvs23.glb",
        "measured":24,"predicted_ai":len(candidate)-1,"rgb_mvs":1,
        "mvs_target":target_index,"prior_mesh":False}
    del changed
    gc.collect()
    for spec in result["meshes"].values():
        path = args.output/spec["glb"]
        mesh = trimesh.load(str(path),force="mesh",process=False)
        if len(mesh.faces) < 100 or mesh.visual.kind != "vertex":
            raise RuntimeError(f"Invalid exported color Mesh: {path}")
        spec["faces"] = int(len(mesh.faces))
        spec["sha256"] = sha256(path)
        del mesh
    association = json.loads((imported/"association.json").read_text())
    boards = []
    for row,index in enumerate(RECORDED):
        path = root/ROOT/association["frames"][index]["rgb_path"]
        with Image.open(path) as image:
            original = image.convert("RGB")
            original.save(args.output/f"real-desk-{index:02d}-640x480.png")
            photo = np.asarray(original.resize((320,240),Image.Resampling.LANCZOS))
        name = f"recorded-{index:02d}-ai-vs-mvs23.png"
        boards.append(panel(args.output/name,
            [photo,rendered["hybrid"][row],rendered["mvs"][row]],
            ["REAL SOURCE RGB","24 + 12 AI","24 + 11 AI + RGB MVS"]))
        result["views"].append({"kind":"recorded","index":index,"file":name})
    for offset,(side,index) in enumerate(FREE):
        row = len(RECORDED)+offset
        before,after = rendered["hybrid"][row],rendered["mvs"][row]
        name = f"free-{'left' if side < 0 else 'right'}-ai-vs-mvs23.png"
        boards.append(panel(args.output/name,[np.full_like(before,20),before,after],
            ["FREE / NO RGB","24 + 12 AI","24 + 11 AI + RGB MVS"]))
        result["views"].append({"kind":"free","index":index,"file":name,
                                "shift_m":.22,"yaw_deg":12*side})
    board = Image.new("RGB",(960,267*len(boards)))
    for i,canvas in enumerate(boards):
        board.paste(canvas,(0,267*i))
    board.save(args.output/"desk-ai-vs-mvs23-five-views.png")
    # Oracle sensor depths opened ONLY AFTER both meshes have been exported.
    target_pairs = [(i,entries[i]) for i in predictions_by_index]
    recording.entries = [entry for _,entry in target_pairs]
    validation = []
    mvs_validation = None
    for (index,_),(_,sensor,_) in zip(target_pairs,recording.iter_frames()):
        estimate = predictions_by_index[index]
        valid = (sensor.depth>.4)&(sensor.depth<3.8)&(estimate>0)
        if np.any(valid):
            if index == target_index:
                mvs_mask = (sensor.depth>.4)&(sensor.depth<3.8)&(metric_mvs>0)
                if np.any(mvs_mask):
                    mvs_validation = {"index":index,"pixels":int(mvs_mask.sum()),
                        "median_abs_depth_error_m":float(np.median(np.abs(
                            sensor.depth[mvs_mask]-metric_mvs[mvs_mask])))}
            validation.append({"index":index,"median_abs_depth_error_m":
                float(np.median(np.abs(sensor.depth[valid]-estimate[valid]))),
                "pixels":int(valid.sum())})
    result["posthoc_target_sensor_validation"] = validation
    result["posthoc_mvs23_sensor_validation"] = mvs_validation
    result["target_sensor_depth_opened_after_mesh_export"] = True
    result["visual_verdict"] = "PENDING human inspection of same-camera 3+2 AI-versus-MVS views"
    result["full_room"] = "FAIL until continuous room confirmed"
    result["historical_fail"] = "Desk 16.7779cm > 15cm; #74 FAIL"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--archive",type=Path)
    args = p.parse_args()
    result = {"status":"running","source":URL,"commands":[],
              "run_url":f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
              if os.getenv("GITHUB_RUN_ID") else None}
    try:
        run(args,result)
        result["status"] = "REAL_THREE_MESHES_EXPORTED_VISUAL_REVIEW_PENDING"
        return 0
    except Exception as exc:
        traceback.print_exc()
        result.update(status="FAIL",error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
        print(json.dumps({"status":result["status"],"meshes":result.get("meshes"),
                          "error":result.get("error")}),flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
