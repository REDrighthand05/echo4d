"""Comprehensive automated test suite for the ECHO 4-Layer Micro-Modular Architecture."""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Add app directory to sys.path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


class TestMicroModularConstraints(unittest.TestCase):
    """Verify that every refactored source file strictly adheres to the <= 300 lines rule."""

    def test_line_counts_under_300(self):
        packages = ["echo_engine", "echo_service", "echo_cli", "echo_desktop"]
        for pkg in packages:
            pkg_path = APP_DIR / pkg
            for root, _, files in os.walk(pkg_path):
                for f in files:
                    if f.endswith(".py"):
                        p = Path(root) / f
                        with open(p, "r", encoding="utf-8") as fp:
                            count = len(fp.readlines())
                        rel = p.relative_to(APP_DIR)
                        self.assertLessEqual(
                            count,
                            300,
                            f"File {rel} has {count} lines, exceeding the 300-line constraint!",
                        )

        for facade in ["app.py", "web_app.py"]:
            p = APP_DIR / facade
            with open(p, "r", encoding="utf-8") as fp:
                count = len(fp.readlines())
            self.assertLessEqual(
                count,
                150,
                f"Facade {facade} has {count} lines, exceeding the 150-line facade limit!",
            )


class TestLayer1Engine(unittest.TestCase):
    """Verify Layer 1: echo_engine pure 3D kernel algorithms and job helpers."""

    def test_engine_exports(self):
        import echo_engine as engine

        required_symbols = [
            "run_pipeline",
            "run_depth_video",
            "play_file",
            "scene_file",
            "replay_file",
            "replay_available",
            "job_finished",
            "job_mtime",
            "list_previous_jobs",
            "scale_status_line",
            "job_display_name",
            "job_file_flags",
            "PLAIN_STATUS",
            "STAGE_COUNT",
            "STAGE_ORDER",
        ]
        for sym in required_symbols:
            self.assertTrue(hasattr(engine, sym), f"Missing engine symbol: {sym}")

        self.assertEqual(len(engine.PLAIN_STATUS), 12)
        self.assertEqual(engine.STAGE_COUNT, 12)
        self.assertEqual(len(engine.STAGE_ORDER), 12)


class TestLayer2Service(unittest.TestCase):
    """Verify Layer 2: echo_service task queue, cache, validators, and REST API."""

    def test_throttled_cache(self):
        from echo_service import ThrottledCache

        cache = ThrottledCache(ttl_seconds=0.2)
        calls = [0]

        def scanner():
            calls[0] += 1
            return calls[0]

        val1 = cache.get_or_scan(scanner)
        self.assertEqual(val1, 1)
        val2 = cache.get_or_scan(scanner)
        self.assertEqual(val2, 1)  # Cached within TTL

        cache.invalidate()
        val3 = cache.get_or_scan(scanner)
        self.assertEqual(val3, 2)  # Refreshed on invalidate

    def test_video_validator(self):
        from echo_service import check_video

        # Non-video extension
        bad_ext = Path("test.txt")
        self.assertIsNotNone(check_video(bad_ext))

        # Missing file
        missing_mp4 = Path("non_existent_12345.mp4")
        self.assertIsNotNone(check_video(missing_mp4))


class TestLayer3CLI(unittest.TestCase):
    """Verify Layer 3: echo_cli argument parsing and headless execution dispatch."""

    def test_cli_subcommands(self):
        from echo_cli.parser import build_parser

        parser = build_parser()

        # Test run
        args_run = parser.parse_args(["run", "input.mp4", "--export"])
        self.assertEqual(args_run.subcommand, "run")
        self.assertEqual(args_run.video, "input.mp4")
        self.assertTrue(args_run.export)

        # Test depth
        args_depth = parser.parse_args(["depth", "input.mp4"])
        self.assertEqual(args_depth.subcommand, "depth")
        self.assertEqual(args_depth.video, "input.mp4")

        # Test serve
        args_serve = parser.parse_args(["serve", "--port", "9000"])
        self.assertEqual(args_serve.subcommand, "serve")
        self.assertEqual(args_serve.port, 9000)

        # Test status
        args_status = parser.parse_args(["status", "--limit", "5"])
        self.assertEqual(args_status.subcommand, "status")
        self.assertEqual(args_status.limit, 5)


class TestLayer4Desktop(unittest.TestCase):
    """Verify Layer 4: echo_desktop bridge and geometry."""

    def test_hit_test_resize(self):
        from echo_desktop.win32_types import (
            _HTBOTTOMRIGHT,
            _HTCLIENT,
            _HTLEFT,
            hit_test_resize,
        )

        rect = (100, 100, 500, 400)
        # Center inside
        self.assertEqual(hit_test_resize(300, 250, rect), _HTCLIENT)
        # Left edge
        self.assertEqual(hit_test_resize(102, 250, rect), _HTLEFT)
        # Bottom-right corner
        self.assertEqual(hit_test_resize(498, 398, rect), _HTBOTTOMRIGHT)

    def test_bridge_state_and_polling(self):
        from echo_desktop import Bridge

        with tempfile.TemporaryDirectory() as tmp_dir:
            bridge = Bridge(jobs_root=Path(tmp_dir))
            snap = bridge.get_snapshot()
            self.assertIn("version", snap)
            self.assertIn("history", snap)
            self.assertIn("status", snap)
            self.assertFalse(snap["busy"])

            polled = bridge.poll()
            self.assertIn("status", polled)
            self.assertIn("frac", polled)
            self.assertFalse(polled["busy"])


class TestFacadesBackwardCompatibility(unittest.TestCase):
    """Verify that root app.py and web_app.py expose all legacy functions."""

    def test_app_facade(self):
        import app

        for sym in ["run_pipeline", "list_previous_jobs", "scene_file", "main", "JOBS_ROOT"]:
            self.assertTrue(hasattr(app, sym), f"app.py missing {sym}")

    def test_webapp_facade(self):
        import web_app

        for sym in ["Bridge", "main", "_hit_test_resize", "_png_complete", "run_pipeline"]:
            self.assertTrue(hasattr(web_app, sym), f"web_app.py missing {sym}")


if __name__ == "__main__":
    unittest.main()
