"""Render Studio — project model (no UI, no rendering).

The Studio is a workbench over the reconstruction's own products: a source
timeline (the build process of one finished job, replayed from its
real intermediate products) that the user directs like a tiny video
editor:

  * attachment segments — over any source-time range, the camera is
    attached to one subject: "route" (ride the recovered camera
    path), "photo" (watch the newest photo land) or "orbit" (slow
   环视 of the whole scene). Gaps default to "orbit".
  * slow segments — a source range plays slower by a factor
    (factor 2 = half speed), stretching the output timeline.
  * element switches — grid / route line / photos on or off.
  * crush track — the photo-crush amplitude over time, a keyframe
    track sampled by the DS batch-4 kernel `sample_track`
    (echo_super.ds_batch4, bench-verified, registered); the Studio
    only ever *samples* it, the curve math is the kernel's.

Everything here is deterministic and serialisable to one JSON file
(studio.json inside the job folder), so a project can be reopened,
re-rendered and diffed. Time mapping: the output timeline is the
source timeline with slow segments stretched; `to_source` walks the
elementary intervals between segment boundaries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ATTACH_MODES = ("route", "photo", "orbit")
CRUSH_METHODS = ("burst", "rise", "drop")
DEFAULT_DURATION = 24.0


@dataclass
class AttachSeg:
    t0: float
    t1: float
    mode: str = "orbit"
    # for mode "photo": which photo to watch — a frame file name, or
    # "" for whichever photo landed most recently
    target: str = ""

    def __post_init__(self):
        if self.mode not in ATTACH_MODES:
            raise ValueError(f"unknown attach mode {self.mode!r}")
        if not self.t1 > self.t0:
            raise ValueError("attach segment must have t1 > t0")


@dataclass
class SlowSeg:
    t0: float
    t1: float
    factor: float = 2.0

    def __post_init__(self):
        if not self.t1 > self.t0:
            raise ValueError("slow segment must have t1 > t0")
        if not self.factor > 0:
            raise ValueError("slow factor must be positive")


@dataclass
class StudioProject:
    duration: float = DEFAULT_DURATION          # source seconds
    attach: list = field(default_factory=list)  # list[AttachSeg]
    slow: list = field(default_factory=list)    # list[SlowSeg]
    elements: dict = field(default_factory=lambda: {
        "grid": True, "route": True, "photos": True})
    # keyframe track: (time, amplitude, easing) — sampled by sample_track
    crush_keys: list = field(default_factory=lambda: [
        (0.0, 1.0, "linear"), (DEFAULT_DURATION, 1.0, "linear")])
    crush_method: str = "burst"     # burst | rise | drop
    # keyframe track for how hard the camera follows its attachment
    # subject (1 = locked on, 0 = free orbit), sampled the same way
    follow_keys: list = field(default_factory=lambda: [
        (0.0, 1.0, "linear"), (DEFAULT_DURATION, 1.0, "linear")])

    # -- queries ------------------------------------------------------
    def attach_at(self, s: float) -> str:
        seg = self.attach_seg_at(s)
        return seg.mode if seg else "orbit"

    def attach_seg_at(self, s: float):
        for seg in self.attach:
            if seg.t0 <= s < seg.t1:
                return seg
        return None

    @staticmethod
    def _sample(keys, s: float) -> float:
        from echo_super.ds_batch4 import sample_track
        keys = [(float(t), float(v), str(e)) for t, v, e in keys]
        return float(sample_track(keys, float(s)))

    def crush_at(self, s: float) -> float:
        return self._sample(self.crush_keys, s)

    def follow_at(self, s: float) -> float:
        return min(max(self._sample(self.follow_keys, s), 0.0), 1.0)

    # -- time mapping ---------------------------------------------------
    def _intervals(self):
        """Elementary [a, b, factor] slices of the source timeline."""
        cuts = {0.0, float(self.duration)}
        for seg in self.slow:
            cuts.add(min(max(seg.t0, 0.0), self.duration))
            cuts.add(min(max(seg.t1, 0.0), self.duration))
        pts = sorted(cuts)
        out = []
        for a, b in zip(pts, pts[1:]):
            if b - a <= 0:
                continue
            mid = 0.5 * (a + b)
            factor = 1.0
            for seg in self.slow:
                if seg.t0 <= mid < seg.t1:
                    factor = seg.factor
                    break
            out.append((a, b, factor))
        return out

    def output_duration(self) -> float:
        return sum((b - a) * f for a, b, f in self._intervals())

    def to_source(self, t_out: float) -> float:
        """Output seconds -> source seconds (slow segments stretched)."""
        if t_out <= 0:
            return 0.0
        acc = 0.0
        for a, b, f in self._intervals():
            span_out = (b - a) * f
            if t_out <= acc + span_out:
                return a + (t_out - acc) / f
            acc += span_out
        return float(self.duration)

    def to_output(self, s: float) -> float:
        """Source seconds -> output seconds."""
        if s <= 0:
            return 0.0
        acc = 0.0
        for a, b, f in self._intervals():
            if s <= b:
                return acc + (s - a) * f
            acc += (b - a) * f
        return acc

    # -- editing ---------------------------------------------------------
    def set_attach(self, t0: float, t1: float, mode: str,
                   target: str = "") -> None:
        """Set the attach mode over [t0, t1), cutting existing
        segments (their own targets survive the cut)."""
        t0, t1 = float(t0), float(t1)
        if not t1 > t0:
            raise ValueError("empty attach range")
        kept = []
        for seg in self.attach:
            if seg.t1 <= t0 or seg.t0 >= t1:
                kept.append(seg)
                continue
            if seg.t0 < t0:
                kept.append(AttachSeg(seg.t0, t0, seg.mode, seg.target))
            if seg.t1 > t1:
                kept.append(AttachSeg(t1, seg.t1, seg.mode, seg.target))
        kept.append(AttachSeg(t0, t1, mode, target))
        self.attach = sorted(kept, key=lambda x: x.t0)

    def set_slow(self, t0: float, t1: float, factor: float | None) -> None:
        """Set (or clear, factor=None) the slow factor over [t0, t1)."""
        t0, t1 = float(t0), float(t1)
        if not t1 > t0:
            raise ValueError("empty slow range")
        kept = []
        for seg in self.slow:
            if seg.t1 <= t0 or seg.t0 >= t1:
                kept.append(seg)
                continue
            if seg.t0 < t0:
                kept.append(SlowSeg(seg.t0, t0, seg.factor))
            if seg.t1 > t1:
                kept.append(SlowSeg(t1, seg.t1, seg.factor))
        if factor is not None:
            kept.append(SlowSeg(t0, t1, float(factor)))
        self.slow = sorted(kept, key=lambda x: x.t0)

    @staticmethod
    def _set_key(keys, t: float, value: float):
        keys = [(float(a), float(b), str(e)) for a, b, e in keys
                if abs(a - t) > 1e-9]
        keys.append((float(t), float(value), "ease_in_out"))
        return sorted(keys)

    def set_crush(self, t: float, amp: float) -> None:
        """Drop a crush keyframe at time t (replaces one at the same t)."""
        self.crush_keys = self._set_key(self.crush_keys, t, amp)

    def set_follow(self, t: float, strength: float) -> None:
        """Drop a follow-strength keyframe at time t."""
        self.follow_keys = self._set_key(self.follow_keys, t, strength)

    def set_crush_method(self, method: str) -> None:
        if method not in CRUSH_METHODS:
            raise ValueError(f"unknown crush method {method!r}")
        self.crush_method = method

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "attach": [{"t0": s.t0, "t1": s.t1, "mode": s.mode,
                        "target": s.target} for s in self.attach],
            "slow": [{"t0": s.t0, "t1": s.t1, "factor": s.factor}
                     for s in self.slow],
            "elements": dict(self.elements),
            "crush_keys": [list(k) for k in self.crush_keys],
            "crush_method": self.crush_method,
            "follow_keys": [list(k) for k in self.follow_keys],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StudioProject":
        p = cls(duration=float(d.get("duration", DEFAULT_DURATION)))
        p.attach = [AttachSeg(float(s["t0"]), float(s["t1"]), s["mode"],
                              s.get("target", ""))
                    for s in d.get("attach", [])]
        p.slow = [SlowSeg(float(s["t0"]), float(s["t1"]),
                          float(s["factor"])) for s in d.get("slow", [])]
        p.elements = {"grid": True, "route": True, "photos": True,
                      **d.get("elements", {})}
        if d.get("crush_keys"):
            p.crush_keys = [(float(a), float(b), str(e))
                            for a, b, e in d["crush_keys"]]
        if d.get("crush_method") in CRUSH_METHODS:
            p.crush_method = d["crush_method"]
        if d.get("follow_keys"):
            p.follow_keys = [(float(a), float(b), str(e))
                             for a, b, e in d["follow_keys"]]
        return p

    def save(self, job_dir: Path) -> Path:
        path = Path(job_dir) / "studio.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False,
                                   indent=2))
        return path

    @classmethod
    def load(cls, job_dir: Path) -> "StudioProject":
        path = Path(job_dir) / "studio.json"
        if path.exists():
            return cls.from_dict(json.loads(path.read_text()))
        return cls()
