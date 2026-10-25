"""Independent analytic ray-box sensor fixture, never used by the fusion kernel."""
import math
from pathlib import Path
import numpy as np
from PIL import Image
from .model import Camera, DataError
from .output import output_transaction, write_json

BOXES = [
    ([-1.6,-.08,-1.7], [1.6,0,1.3], [95,130,137]),
    ([-1.65,0,-1.75], [1.65,1.6,-1.65], [122,150,155]),
    ([-1.65,0,-1.7], [-1.55,1.6,1.3], [100,135,145]),
    ([-1.1,0,-1.3], [.35,.5,-.55], [83,153,150]),
    ([-.45,0,-.1], [.55,.55,.6], [156,149,119]),
    ([.8,0,-1.35], [1.35,1.2,-.85], [147,138,118]),
]


def create_demo(output: Path, frames: int = 16, noise: float = 0) -> None:
    if type(frames) is not int or not 2 <= frames <= 300 or not math.isfinite(noise) or not 0 <= noise <= .2:
        raise DataError("Demo frames must be 2-300; noise half-width in [0,.2] metres")
    camera = Camera(80, 60, 62., 62., 39.5, 29.5)
    y, x = np.indices((camera.height, camera.width))
    rays = np.stack([(x-camera.cx)/camera.fx, (y-camera.cy)/camera.fy, np.ones_like(x)], axis=-1)
    rng = np.random.default_rng(14)
    with output_transaction(output) as root:
        (root / "depth").mkdir()
        (root / "color").mkdir()
        entries = []
        for i in range(frames):
            angle = -.7 + 1.4*i/(frames-1)
            position = np.array([math.sin(angle)*2., 1.35, math.cos(angle)*2.])
            forward = np.array([0.,.6,-.45])-position
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0,1,0]); right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            rotation = np.column_stack([right, down, forward])
            directions = rays @ rotation.T
            nearest = np.full((camera.height, camera.width), np.inf)
            colors = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
            for lo, hi, color in BOXES:
                with np.errstate(divide="ignore", invalid="ignore"):
                    a, b = (np.array(lo)-position)/directions, (np.array(hi)-position)/directions
                enter, leave = np.minimum(a,b).max(axis=-1), np.maximum(a,b).min(axis=-1)
                distance = np.where(enter > 0, enter, leave)
                hit = (leave >= np.maximum(enter,0)) & (distance > 0) & (distance < nearest)
                nearest[hit] = distance[hit]
                colors[hit] = color
            valid = np.isfinite(nearest)
            depth = np.zeros_like(nearest)
            depth[valid] = np.maximum(0, nearest[valid] + rng.uniform(-noise,noise,valid.sum()))
            Image.fromarray(np.rint(depth*1000).astype(np.uint16)).save(root / "depth" / f"{i:04d}.png")
            Image.fromarray(colors).save(root / "color" / f"{i:04d}.png")
            pose = np.eye(4); pose[:3,:3] = rotation; pose[:3,3] = position
            entries.append({"t": i*.2, "depth": f"depth/{i:04d}.png", "color": f"color/{i:04d}.png",
                            "cameraToWorld": pose.flatten(order="F").tolist()})
        write_json(root / "recording.json", {"version":"echo-dataset-1", "source":"synthetic", "units":"meters",
            "name":"Analytic box fixture", "duration":(frames-1)*.2, "depth_scale":1000,
            "intrinsics":camera.to_dict(), "frames":entries})
        write_json(root / "groundtruth.json", {"boxes":BOXES, "note":"Never read by fusion; analytic fixture only."})
