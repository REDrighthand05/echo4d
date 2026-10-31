"""Demo kernels — light stand-ins with the same kind-signature as the
real pipeline stages (extract / poses / depth / fuse / present).

They prove the routing on any machine in milliseconds. Wrapping the
real stage_* functions as kernels is a one-line registration each,
using exactly these consumes/produces declarations.
"""
from __future__ import annotations

from .registry import Registry


def build_demo_registry() -> Registry:
    reg = Registry()

    @reg.register("extract", ["video"], "frames")
    def _(video):
        return {"frames_from": video.data}

    @reg.register("photos_to_frames", ["photos"], "frames")
    def _(photos):
        return {"frames_from": list(photos.data)}

    @reg.register("poses", ["frames"], "poses")
    def _(frames):
        return {"poses_for": frames.data}

    @reg.register("trajectory_plain", ["poses"], "trajectory", cost=1.0)
    def _(poses):
        return {"trajectory": poses.data, "stabilized": False}

    @reg.register("trajectory_imu", ["poses", "imu"], "trajectory", cost=0.5)
    def _(poses, imu):
        return {"trajectory": poses.data, "stabilized": True, "imu": imu.data}

    @reg.register("depth", ["frames"], "depth")
    def _(frames):
        return {"depth_for": frames.data}

    @reg.register("fuse", ["frames", "trajectory", "depth"], "points")
    def _(frames, trajectory, depth):
        return {"points": "fused", "trajectory": trajectory.data,
                "depth": depth.data}

    @reg.register("present", ["points"], "scene", cost=1.0)
    def _(points):
        return {"scene_from": points.data}

    @reg.register("points_to_mesh", ["points"], "mesh", cost=1.0)
    def _(points):
        return {"mesh_from": points.data}

    @reg.register("mesh_to_scene", ["mesh"], "scene", cost=1.0)
    def _(mesh):
        return {"scene_from_mesh": mesh.data}

    return reg
