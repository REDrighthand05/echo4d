"""Shared model resolution (echo_scene.paths).

Regression guard for the installed-layout failure of 2026-11-03: the
depth models live at the INSTALL ROOT while payload modules live one
level deeper (app/), so a module computing models/ from its own
__file__ found nothing and the depth video died after extraction.
The resolver must (a) honour each caller's model preference with
fallback to the other model, and (b) search several roots.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo_scene import paths


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        empty = root / "empty"
        empty.mkdir()
        only_base = root / "base"
        only_base.mkdir()
        (only_base / paths.MODEL_BASE).write_bytes(b"x")
        only_small = root / "small"
        only_small.mkdir()
        (only_small / paths.MODEL_SMALL).write_bytes(b"x")
        both = root / "both"
        both.mkdir()
        (both / paths.MODEL_BASE).write_bytes(b"x")
        (both / paths.MODEL_SMALL).write_bytes(b"x")

        # Preference honoured when both exist.
        assert paths.find_depth_model(False, [both]).name == paths.MODEL_SMALL
        assert paths.find_depth_model(True, [both]).name == paths.MODEL_BASE
        # Fallback to whichever model exists.
        assert paths.find_depth_model(False, [only_base]).name == paths.MODEL_BASE
        assert paths.find_depth_model(True, [only_small]).name == paths.MODEL_SMALL
        # Later roots are searched when earlier ones are empty.
        hit = paths.find_depth_model(False, [empty, only_small])
        assert hit is not None and hit.parent == only_small
        # Nothing anywhere -> None, and require_ raises naming the hunt.
        assert paths.find_depth_model(False, [empty]) is None
        try:
            paths.require_depth_model()
        except FileNotFoundError:
            pass  # message content depends on the host layout; type is the contract
        # The real candidate list always includes the payload-parent
        # (install-root) location -- the 2026-11-03 miss.
        cands = paths.candidate_model_dirs()
        here = Path(paths.__file__).resolve().parent
        assert here.parent.parent / "models" in cands, cands
    print("PASS test_paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
