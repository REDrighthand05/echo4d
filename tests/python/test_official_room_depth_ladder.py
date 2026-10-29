"""Offline contract tests for frame identity and upstream trajectory recovery."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / 'tools' / 'official_room_depth_ladder.py'
spec = importlib.util.spec_from_file_location('official_room_depth_ladder', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestOfficialLoungeDepthLadder(unittest.TestCase):
    def test_trajectory_and_frame_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'color').mkdir()
            (root/'depth').mkdir()
            records = []
            for i in range(3):
                (root/'color'/f'{i+1:06d}.png').touch()
                (root/'depth'/f'{i+1:06d}.png').touch()
                records.extend([f'{i} {i} {i+1}', *(' '.join(map(str, row)) for row in np.eye(4))])
            (root/'trajectory.log').write_text('\n'.join(records)+'\n')
            poses = mod.load_trajectory(root/'trajectory.log')
            pairs = mod.native_pairs(root/'color', root/'depth', poses)
            self.assertEqual([p[0] for p in pairs], [0, 1, 2])
            self.assertEqual(pairs[0][1].name, '000001.png')
            self.assertEqual(len(mod.even_indices(3000, 600)), 600)
            self.assertEqual(len(mod.even_indices(3000, 150)), 150)

    def test_reject_missing_depth_or_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'color').mkdir()
            (root/'depth').mkdir()
            (root/'color'/'000000.png').touch()
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                mod.native_pairs(root/'color', root/'depth', {0: np.eye(4)})
            with self.assertRaises(ValueError):
                mod.even_indices(3, 4)


if __name__ == '__main__':
    unittest.main()
