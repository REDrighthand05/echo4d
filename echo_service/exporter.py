"""Export utilities for colored voxels (GLB/PLY), camera trajectory, and batch exports."""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any


def ensure_voxel_export(job: Path) -> Dict[str, Any]:
    """Ensure that voxel_color.glb and voxel_color.ply are generated for a finished job."""
    glb = job / "voxel_color.glb"
    ply = job / "voxel_color.ply"
    if glb.exists() and ply.exists():
        return {"ok": True, "cached": True}

    try:
        from echo_scene.voxel_export import export_colored_voxels
        return export_colored_voxels(job)
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def export_scene_to_dir(job: Path, dest_dir: Path, fmt: str = "glb") -> Dict[str, Any]:
    """Export the colored solidified model (GLB or PLY) and info json to dest_dir."""
    job = Path(job)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    build_info = ensure_voxel_export(job)
    if not build_info.get("ok"):
        return {"ok": False, "reason": build_info.get("reason", "固化失败")}

    fmt = fmt.lower() if fmt.lower() in ("glb", "ply") else "glb"
    chosen = job / f"voxel_color.{fmt}"
    info_file = job / "voxel_color_info.json"

    copied: List[str] = []
    try:
        for src in (chosen, info_file):
            if src.exists():
                shutil.copy2(src, dest_dir / src.name)
                copied.append(src.name)
        if not copied:
            return {"ok": False, "reason": "未找到可复制的导出文件"}
        return {"ok": True, "files": copied, "dest": str(dest_dir)}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def export_camera_route_to_dir(job: Path, dest_dir: Path) -> Dict[str, Any]:
    """Export recovered camera route (poses.json and route.glb) to dest_dir."""
    job = Path(job)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    route_files = [job / "poses.json", job / "route.glb"]
    copied: List[str] = []
    try:
        for src in route_files:
            if src.exists():
                shutil.copy2(src, dest_dir / src.name)
                copied.append(src.name)
        return {"ok": len(copied) > 0, "files": copied, "dest": str(dest_dir)}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def batch_export_to_dir(jobs: List[Path], dest_dir: Path, fmt: str = "glb") -> Dict[str, Any]:
    """Export multiple jobs into subfolders within dest_dir."""
    results: Dict[str, Any] = {}
    for job in jobs:
        sub = dest_dir / job.name
        results[job.name] = export_scene_to_dir(job, sub, fmt=fmt)
    return {"ok": True, "results": results}


# Aliases for clean public API
export_voxel_model = export_scene_to_dir
export_camera_trajectory = export_camera_route_to_dir
export_batch = batch_export_to_dir

