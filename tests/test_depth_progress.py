"""Depth video: progress reporting + shared model resolution.

The depth render used to give its host no progress at all (a frozen
bar for minutes) and resolved its model from its own __file__, which
is wrong in the installed layout. This runs the real feature on a
tiny synthetic clip and pins both behaviours.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo_scene.depth_video import make_depth_video
from echo_scene.pipeline import _tool_path


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "clip.mp4"
        subprocess.run(
            [_tool_path("ffmpeg"), "-y", "-v", "error",
             "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
             "-pix_fmt", "yuv420p", str(src)], check=True)
        calls = []
        out = root / "clip_depth.mp4"
        info = make_depth_video(
            src, out, fps=5,
            progress=lambda s, d, t: calls.append((s, d, t)))
        assert info.get("ok"), info
        assert out.exists() and out.stat().st_size > 0
        stages = [c[0] for c in calls]
        assert stages[0] == "frames", stages
        assert "depth" in stages and stages[-1] == "encode", stages
        depth_calls = [c for c in calls if c[0] == "depth"]
        assert depth_calls[-1][1] == depth_calls[-1][2] == info["frames"]
        assert all(c[2] == info["frames"] for c in depth_calls)
    print("PASS test_depth_progress")
    return 0


if __name__ == "__main__":
    sys.exit(main())
