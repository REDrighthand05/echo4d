import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echo_super import Plan, Registry, make_super, parse_sexp, wrap
from echo_super.demo import build_demo_registry


class SuperTests(unittest.TestCase):
    def setUp(self):
        self.go = make_super(build_demo_registry())

    def test_video_full_route(self):
        scene = self.go("room.mp4")
        self.assertEqual(scene.kind, "scene")
        names = [s[1] for s in scene.meta["plan"][1:-1]]
        self.assertEqual(names, ["extract", "poses", "trajectory_plain",
                                 "depth", "fuse", "present"])

    def test_variable_inputs_match_better_path(self):
        # video + imu: same length route, cheaper stabilized trajectory wins
        scene = self.go("room.mp4", "imu_log.csv")
        names = [s[1] for s in scene.meta["plan"][1:-1]]
        self.assertIn("trajectory_imu", names)
        self.assertTrue(scene.data["scene_from"]["trajectory"]["stabilized"])

    def test_satisfied_steps_are_skipped(self):
        pts = wrap({"points": 42}, kind="points")
        scene = self.go(pts)
        names = [s[1] for s in scene.meta["plan"][1:-1]]
        self.assertEqual(names, ["present"])

    def test_given_depth_is_not_recomputed(self):
        scene = self.go(("frames", ["f1"]), ("depth", {"d": 1}))
        names = [s[1] for s in scene.meta["plan"][1:-1]]
        self.assertNotIn("depth", names)
        self.assertNotIn("extract", names)

    def test_photos_route(self):
        scene = self.go(["a.jpg", "b.jpg"])
        names = [s[1] for s in scene.meta["plan"][1:-1]]
        self.assertEqual(names[0], "photos_to_frames")

    def test_plan_is_data_sexp_and_json_roundtrip(self):
        scene = self.go("room.mp4")
        plan_data = scene.meta["plan"]
        self.assertEqual(parse_sexp(scene.meta["sexp"]), plan_data)
        p = Plan(steps=[("extract", ("video",), "frames")], target="scene",
                 inputs=["video"])
        self.assertEqual(Plan.from_json(p.to_json()).steps, p.steps)

    def test_restart_reroutes_around_failing_kernel(self):
        reg = build_demo_registry()

        def boom(points):
            raise RuntimeError("present failed")

        reg.kernels["present"].fn = boom
        go = make_super(reg)
        scene = go(wrap({"points": 1}, kind="points"))
        names = [s[1] for s in scene.meta["plan"][1:-1]]
        self.assertEqual(names, ["points_to_mesh", "mesh_to_scene"])
        self.assertIn("scene_from_mesh", scene.data)


if __name__ == "__main__":
    unittest.main()
