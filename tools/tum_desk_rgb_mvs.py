"""CPU calibrated-pose multi-view RGB plane sweep for TUM Desk.

Unlike Depth Anything, depth is triangulated from real RGB reprojection and
TUM reference camera translations (metric scale). No target sensor depth,
pre-existing mesh, GPU, or learned weights enter this module.
"""
import numpy as np
import cv2
from PIL import Image
from echo4d.model import Camera, rigid_matrix, safe_path


def candidate_neighbours(entries, recording, camera, target_index, allowed, count=4):
    """Select RGB-only source views using poses/overlap, never sensor depth."""
    target = rigid_matrix(entries[target_index]["cameraToWorld"])
    centre = target[:3, 3]
    axis = target[:3, 2]
    corners = np.array([[.1,.1,1.5],[camera.width-.1,.1,1.5],
                        [.1,camera.height-.1,1.5],
                        [camera.width-.1,camera.height-.1,1.5],
                        [camera.cx,camera.cy,1.5]], dtype=np.float64)
    rays = np.column_stack(((corners[:,0]-camera.cx)*corners[:,2]/camera.fx,
                            (corners[:,1]-camera.cy)*corners[:,2]/camera.fy,
                            corners[:,2]))
    world = rays @ target[:3,:3].T + centre
    ranked = []
    for index in allowed:
        if index == target_index:
            continue
        pose = rigid_matrix(entries[index]["cameraToWorld"])
        baseline = float(np.linalg.norm(pose[:3,3]-centre))
        alignment = float(np.dot(axis, pose[:3,2]))
        if not .025 <= baseline <= .55 or alignment < .82:
            continue
        other = (world-pose[:3,3]) @ pose[:3,:3]
        z = other[:,2]
        u = other[:,0]*camera.fx/np.maximum(z,1e-6)+camera.cx
        v = other[:,1]*camera.fy/np.maximum(z,1e-6)+camera.cy
        overlap = np.mean((z>.3)&(u>=0)&(u<camera.width)&(v>=0)&(v<camera.height))
        if overlap < .6:
            continue
        score = abs(np.log(baseline/.13)) + 2*(1-overlap) + 2*(1-alignment)
        ranked.append((score,index,baseline,float(overlap)))
    ranked.sort()
    return [i for _,i,_,_ in ranked[:count]], [
        {"index":int(i),"baseline_m":float(b),"overlap":float(o)}
        for _,i,b,o in ranked[:count]]


def plane_sweep_depth(target_rgb, target_pose, neighbour_rgbs, neighbour_poses,
                      camera, planes=72, min_depth=.45, max_depth=3.8):
    """Metric RGB-only depth from >=2 known-pose neighbour views.

    A per-plane 5x5 photometric/gradient reprojection cost is aggregated over
    the two most consistent neighbours to tolerate partial occlusion.
    Confidence and texture gates deliberately leave uncertain pixels empty.
    """
    if len(neighbour_rgbs) < 2 or len(neighbour_rgbs) != len(neighbour_poses):
        raise ValueError("At least two calibrated neighbour RGB images required")
    h,w = target_rgb.shape[:2]
    if (camera.height,camera.width)!=(h,w):
        raise ValueError("Camera and image dimensions differ")
    if planes < 12 or min_depth <= 0 or max_depth <= min_depth:
        raise ValueError("Invalid inverse-depth search")
    ref = cv2.cvtColor(target_rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    grad = cv2.Sobel(ref,cv2.CV_32F,1,0,ksize=3)*.125
    grad += cv2.Sobel(ref,cv2.CV_32F,0,1,ksize=3)*.125
    mean = cv2.boxFilter(ref,-1,(7,7))
    texture = np.maximum(cv2.boxFilter(ref*ref,-1,(7,7))-mean*mean,0)
    yy,xx = np.mgrid[:h,:w].astype(np.float32)
    rayx = (xx-camera.cx)/camera.fx
    rayy = (yy-camera.cy)/camera.fy
    transforms = []
    for rgb,pose in zip(neighbour_rgbs,neighbour_poses):
        if rgb.shape != target_rgb.shape:
            raise ValueError("Neighbour image size differs")
        rel = np.linalg.inv(pose) @ target_pose
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        g = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)*.125
        g += cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)*.125
        transforms.append((rel[:3,:3].astype(np.float32),
                           rel[:3,3].astype(np.float32),gray,g))
    depth_values = 1/np.linspace(1/max_depth,1/min_depth,planes,dtype=np.float32)
    volume = np.empty((planes,h,w),np.float32)
    supports = np.empty((planes,h,w),np.uint8)
    for j,z in enumerate(depth_values):
        x,y = rayx*z,rayy*z
        errors=[]
        for R,t,gray,g in transforms:
            qx=R[0,0]*x+R[0,1]*y+R[0,2]*z+t[0]
            qy=R[1,0]*x+R[1,1]*y+R[1,2]*z+t[1]
            qz=R[2,0]*x+R[2,1]*y+R[2,2]*z+t[2]
            u=(qx*camera.fx/np.maximum(qz,1e-5)+camera.cx).astype(np.float32)
            v=(qy*camera.fy/np.maximum(qz,1e-5)+camera.cy).astype(np.float32)
            valid=(qz>.3)&(u>=2)&(u<w-2)&(v>=2)&(v<h-2)
            warped=cv2.remap(gray,u,v,cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
            warped_grad=cv2.remap(g,u,v,cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            err=.75*np.abs(ref-warped)+.25*np.abs(grad-warped_grad)
            err=cv2.boxFilter(err,-1,(5,5))
            errors.append(np.where(valid,err,np.inf))
        ranked=np.sort(np.stack(errors),axis=0)
        support=np.isfinite(ranked[1])
        volume[j]=np.where(support,(ranked[0]+ranked[1])*.5,np.inf)
        supports[j]=support
    best=np.argmin(volume,axis=0)
    best_cost=np.take_along_axis(volume,best[None],axis=0)[0]
    far=np.abs(np.arange(planes)[:,None,None]-best[None])>=3
    second=np.min(np.where(far,volume,np.inf),axis=0)
    confidence=second-best_cost
    valid=(np.isfinite(best_cost)&np.isfinite(second)&(texture>0.00012)&
           (best_cost<.14)&(confidence>.003))
    depth=np.where(valid,depth_values[best],0).astype(np.float32)
    return depth,{"valid_fraction":float(valid.mean()),
                  "valid_pixels":int(valid.sum()),
                  "median_cost":float(np.median(best_cost[valid])) if valid.any() else None,
                  "median_margin":float(np.median(confidence[valid])) if valid.any() else None,
                  "planes":planes,"scale_source":"TUM metric reference camera translations",
                  "method":"CPU RGB multi-view inverse-depth plane sweep"}


def estimate_target(entries, recording, camera, target_index, allowed):
    indices, geometry=candidate_neighbours(entries,recording,camera,target_index,allowed)
    if len(indices)<2:
        raise RuntimeError("Insufficient spatially separated overlapping RGB views")
    half=Camera(camera.width//2,camera.height//2,camera.fx/2,camera.fy/2,
                camera.cx/2,camera.cy/2)
    def rgb(i):
        with Image.open(safe_path(recording.root,entries[i]["color"])) as im:
            return np.asarray(im.convert("RGB").resize(
                (half.width,half.height),Image.Resampling.BILINEAR))
    target=rgb(target_index)
    sources=[rgb(i) for i in indices]
    pose=rigid_matrix(entries[target_index]["cameraToWorld"])
    other=[rigid_matrix(entries[i]["cameraToWorld"]) for i in indices]
    small,receipt=plane_sweep_depth(target,pose,sources,other,half)
    metric=cv2.resize(small,(camera.width,camera.height),
                      interpolation=cv2.INTER_NEAREST)
    receipt.update(target_index=target_index,neighbours=geometry,
                   target_sensor_depth_read=False,model_weights_used=False)
    return metric,receipt
