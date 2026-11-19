"""Check batch dispatch, without downloading models or requiring a GPU."""
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo_scene import depth_batch


class Session:
    def __init__(self, dimension, reject_batches=False):
        self.dimension = dimension
        self.reject_batches = reject_batches
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name="pixel_values", shape=[self.dimension, 3, 518, 518])]

    def run(self, outputs, inputs):
        tensor = inputs["pixel_values"]
        self.calls.append(len(tensor))
        if self.reject_batches and len(tensor) > 1:
            raise ValueError("batch unsupported")
        return [tensor[:, 0]]


class DepthDispatch(unittest.TestCase):
    def test_fixed_dynamic_and_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for i in range(5):
                path = Path(temp) / f"{i}.png"
                Image.new("RGB", (12 + i, 20), (40 * i, 0, 0)).save(path)
                paths.append(path)
            fixed = Session(1)
            golden = depth_batch.estimate_depth_batch(paths, fixed)
            self.assertEqual(fixed.calls, [1] * 5)
            for reject, expected in [(False, [4, 1]), (True, [4, 1, 1, 1, 1, 1])]:
                session = Session("N", reject)
                with patch.object(depth_batch, "_preprocess_image",
                                  wraps=depth_batch._preprocess_image) as prep:
                    result = depth_batch.estimate_depth_batch(paths, session)
                self.assertEqual(session.calls, expected)
                self.assertEqual(prep.call_count, len(paths))
                for wanted, got in zip(golden, result):
                    np.testing.assert_array_equal(wanted, got)
            mixed = depth_batch.estimate_depth_batch([paths[0], Path(temp)/"missing", paths[2]], Session("N"))
            self.assertIsNone(mixed[1])
            np.testing.assert_array_equal(golden[2], mixed[2])

    def test_fixed_groups_prefetch_and_dynamic_downgrade(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{i}.png" for i in range(9)]
            for i, path in enumerate(paths):
                Image.new("RGB", (13 + i, 20), (13 * i, 0, 0)).save(path)
            golden = depth_batch.estimate_depth_batch(paths, Session(1))
            for session, expected in [
                    (Session(1), [1] * 9),
                    (Session(2), [2] * 5),
                    (Session("N", True), [4] + [1] * 9)]:
                timings = {}
                result = list(depth_batch.iter_depth_estimates(
                    paths, session, prefetch=True, timings=timings))
                self.assertEqual(session.calls, expected)
                self.assertEqual(timings["ort_run_calls"], len(expected))
                for index, output in result:
                    np.testing.assert_array_equal(output, golden[index])

    def test_profile_distinguishes_cuda_transfers_from_model_execution(self):
        import json
        from echo_scene.depth import end_provider_profile

        def event(op, provider):
            return {"cat": "Node", "name": op + "_kernel_time",
                    "args": {"op_name": op, "provider": provider}}

        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.json"
            data = [event("MemcpyFromHost", "CUDAExecutionProvider"),
                    event("Conv", "CPUExecutionProvider")]
            trace.write_text(json.dumps(data), encoding="utf-8")
            session = SimpleNamespace(end_profiling=lambda: str(trace))
            self.assertEqual(end_provider_profile(session, keep_file=True),
                             {"CPUExecutionProvider": 1})
            data.append(event("MatMul", "CUDAExecutionProvider"))
            trace.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(end_provider_profile(session),
                             {"CPUExecutionProvider": 1, "CUDAExecutionProvider": 1})
            self.assertFalse(trace.exists())


if __name__ == "__main__":
    unittest.main()
