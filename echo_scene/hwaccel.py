"""ffmpeg hardware-decode probing utilities."""

import subprocess

_WHITELIST_ORDER = (
    "d3d11va",
    "cuda",
    "qsv",
    "dxva2",
    "vaapi",
    "videotoolbox",
)


def _parse_hwaccel_candidates(stdout: str) -> set[str]:
    lines = stdout.splitlines()
    start = 1 if lines else 0

    for index, line in enumerate(lines):
        if "hardware acceleration" in line.lower():
            start = index + 1
            break

    candidates = set()
    for line in lines[start:]:
        for token in line.strip().split():
            token = token.strip().lower()
            if token:
                candidates.add(token)
    return candidates


def probe_hwaccel(ffmpeg_path: str, timeout: float = 20.0) -> list[str]:
    try:
        probe = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-hwaccels"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )

        if probe.returncode != 0:
            return []

        candidates = _parse_hwaccel_candidates(probe.stdout or "")
        available = []

        for name in _WHITELIST_ORDER:
            if name not in candidates:
                continue

            try:
                check = subprocess.run(
                    [
                        ffmpeg_path,
                        "-hide_banner",
                        "-hwaccel",
                        name,
                        "-f",
                        "lavfi",
                        "-i",
                        "testsrc=duration=1:size=320x240:rate=10",
                        "-frames:v",
                        "1",
                        "-f",
                        "null",
                        "-",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout,
                )
                if check.returncode == 0:
                    available.append(name)
            except Exception:
                continue

        return available
    except Exception:
        return []


def extract_args(hwaccel) -> list[str]:
    if isinstance(hwaccel, str) and hwaccel:
        return ["-hwaccel", hwaccel]
    return []
