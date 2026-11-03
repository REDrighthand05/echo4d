"""DS WO-D acceptance: ffmpeg hwaccel probing (echo_scene.hwaccel).

The probe must never raise, must only return whitelisted names in
the spec's preference order, and extract_args follows its contract.
On a GPU-less dev box the probe legitimately returns [].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo_scene.hwaccel import extract_args, probe_hwaccel
from echo_scene.pipeline import _tool_path

ORDER = ["d3d11va", "cuda", "qsv", "dxva2", "vaapi", "videotoolbox"]


def main() -> int:
    assert extract_args("d3d11va") == ["-hwaccel", "d3d11va"]
    assert extract_args(None) == []
    assert extract_args("") == []
    got = probe_hwaccel(_tool_path("ffmpeg"))
    assert isinstance(got, list)
    assert all(name in ORDER for name in got), got
    assert got == sorted(got, key=ORDER.index), got
    assert probe_hwaccel("/nonexistent/ffmpeg") == []
    # The pipeline's cached arg helper agrees with the probe and
    # never raises on a GPU-less box (empty list = software decode).
    from echo_scene import pipeline

    args = pipeline._hwaccel_args()
    assert args == extract_args(got[0] if got else None), (args, got)
    assert pipeline._hwaccel_args() is args  # cached, probed once
    print("PASS test_hwaccel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
