"""DS WO-A acceptance: batched depth inference (echo_scene.depth_batch).

Gate for merging a DS-written kernel: outputs must match the
single-frame estimate_depth within tolerance on real frames, keep
input order and shapes, isolate a bad frame to its own slot, and the
batch_size=1 path must equal plain sequential calls.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo_scene import depth as depth_mod
from echo_scene.depth_batch import estimate_depth_batch
from echo_scene.paths import require_depth_model


def main() -> int:
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        paths = []
        for i, (w, h) in enumerate([(320, 240), (256, 256), (400, 300),
                                    (320, 240), (288, 216), (352, 288)]):
            p = root / f"f{i}.jpg"
            arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(p, quality=92)
            paths.append(p)
        model = require_depth_model(prefer_base=False)
        session = depth_mod._get_session(model)

        golden = [depth_mod.estimate_depth(p, model) for p in paths]
        batched = estimate_depth_batch(paths, session, batch_size=4)
        assert len(batched) == len(paths)
        for g, b in zip(golden, batched):
            assert b is not None and b.shape == g.shape, (b, g.shape)
            med = float(np.median(np.abs(g - b)))
            assert med < 1e-4, f"median diff {med}"

        singles = estimate_depth_batch(paths, session, batch_size=1)
        for g, s in zip(golden, singles):
            assert s is not None
            assert float(np.median(np.abs(g - s))) < 1e-4

        mixed = estimate_depth_batch(
            [paths[0], root / "nope.jpg", paths[2]], session, batch_size=4)
        assert mixed[0] is not None and mixed[1] is None
        assert mixed[2] is not None

        assert estimate_depth_batch([], session) == []
    print("PASS test_depth_batch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
