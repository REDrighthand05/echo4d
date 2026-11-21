"""Unit tests for Windows task lifecycle state machine, cancellation, and checkpoints."""

import tempfile
import time
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
import sys
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from echo_desktop.bridge import Bridge
from echo_service import (
    CheckpointData,
    CheckpointManager,
    JobManager,
    TaskState,
)


class TestTaskLifecycle(unittest.TestCase):
    def test_task_state_machine_init(self):
        with tempfile.TemporaryDirectory() as td:
            jm = JobManager(jobs_root=Path(td))
            self.assertEqual(jm.task_state, TaskState.CREATED)
            snap = jm.get_state()
            self.assertEqual(snap["taskState"], "CREATED")

    def test_cancellation_cooperative(self):
        with tempfile.TemporaryDirectory() as td:
            jm = JobManager(jobs_root=Path(td))
            # No task running
            ok, msg = jm.request_cancel()
            self.assertFalse(ok)

            # Set busy manually to simulate running task
            jm.busy = True
            jm.task_state = TaskState.RUNNING
            ok, msg = jm.request_cancel()
            self.assertTrue(ok)
            self.assertEqual(jm.task_state, TaskState.CANCEL_REQUESTED)
            self.assertTrue(jm._cancel_event.is_set())

            # Pipeline log checks cancellation
            with self.assertRaises(InterruptedError):
                jm._pipeline_log("正在抽帧…")

    def test_pause_cooperative(self):
        with tempfile.TemporaryDirectory() as td:
            jm = JobManager(jobs_root=Path(td))
            jm.busy = True
            jm.task_state = TaskState.RUNNING

            ok, msg = jm.request_pause()
            self.assertTrue(ok)
            self.assertEqual(jm.task_state, TaskState.PAUSE_REQUESTED)
            self.assertTrue(jm._pause_event.is_set())

    def test_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as td:
            job_p = Path(td)
            dummy_vid = job_p / "input.mp4"
            dummy_vid.write_bytes(b"sample video bytes")

            # Create dummy artifacts
            art1 = job_p / "mesh.ply"
            art1.write_text("ply")

            ckpt = CheckpointData(
                job_id="test_ckpt",
                input_sha256="fake_sha",
                completed_stages=["stage_extract", "stage_poses"],
                active_stage="stage_depth",
                output_artifacts=["mesh.ply"],
            )
            CheckpointManager.save_checkpoint(job_p, ckpt)
            self.assertTrue((job_p / "checkpoint.json").exists())

            # Without matching input sha check
            loaded = CheckpointManager.load_and_validate(job_p)
            self.assertIsNotNone(loaded)
            self.assertTrue(loaded.validated)

            # Invalidate
            CheckpointManager.invalidate(job_p, "user_cancelled")
            data = CheckpointManager.load_and_validate(job_p)
            self.assertIsNone(data)  # Invalidated checkpoint must be rejected

    def test_bridge_task_controls(self):
        with tempfile.TemporaryDirectory() as td:
            bridge = Bridge(jobs_root=Path(td))
            bridge.job_manager.busy = True
            bridge.job_manager.task_state = TaskState.RUNNING

            cancel_res = bridge.cancel_job()
            self.assertTrue(cancel_res["ok"])
            self.assertEqual(bridge.job_manager.task_state, TaskState.CANCEL_REQUESTED)


if __name__ == "__main__":
    unittest.main()
