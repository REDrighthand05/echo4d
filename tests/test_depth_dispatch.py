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


if __name__ == "__main__":
    unittest.main()
