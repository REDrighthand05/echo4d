"""Photo discovery kernel — catalogue entries recognised in a job's
frames, lifted into 3D with the job's own depth + poses, solidified
against the job cloud, delivered as a named object list.

    from echo_super.kernels.photo_discovery import (
        register_photo_discovery_kernels)
    reg = register_photo_discovery_kernels()          # fresh Registry
    reg = register_photo_discovery_kernels(other)     # or into existing

Contract:
  consumes : "job" — data is a job directory path, or a dict
             {"job_dir": path,
              "catalog": <catalogue source, optional — defaults to the
                          bundled catalog.json data file>,
              "detections": <converged OWLv2 records, optional —
                          injecting them skips the OWLv2 stage>,
              "masks": <mask cache, optional — injecting it skips the
                          SAM stage; forms in pipeline.py>}.
             The job directory must carry frames/, depth/*.npy,
             poses.json, alignment.json and scene_points.ply (the
             fused job layout). Parameters ride in meta: "config"
             (dict of DiscoveryConfig field overrides) and "plan"
             (per-class treatment overrides).
  produces : "objects" — data {"objects": [record, ...],
             "point_sets": {label: (N,3) float64},
             "room": {...}}. Each record: label, zh, bbox_min,
             bbox_max, extent, points (solidified count),
             n_detect_frames / detect_frames, top_score, the
             stage-by-stage point counts, typical_m and the
             catalogue's pitfalls note for that entry.

Pipeline (experiments/discovery_photo, three pioneer rounds, final
form per REPORT3): OWLv2 detection with per-class per-frame box
convergence -> SAM masks (depth main-cluster crop) -> lift through
the job's depth+poses (fuse-identical alignment) -> room/surface
gates + 2-frame corroboration -> per-class treatment (monitor:
rank-1 masks only; monitor+chair: largest-component instance
isolation of the corroborated set).

Verified record: on the real 2026-11-03 20:41 capture job, 4 of the
6 catalogue entries pass (desk / wardrobe / monitor / chair — names,
solidified counts and bbox magnitudes reproduce REPORT3 under cached
detection/mask injection; tests/test_photo_discovery.py). Known
boundaries, also printed on the affected output rows via the
catalogue's pitfalls field: shelf is one word for many objects
(needs instance separation, unsolved) and ac is unreliable in dark
light (detections are false alarms; needs a prior or one manual
entry). Evidence grade 估计 (the naming is model inference; the
geometry it lands on is the job's measured cloud), status 已验.
"""
from __future__ import annotations

from pathlib import Path

from ...core import EchoObject
from ...registry import Kernel, Registry
from .catalog import load_catalog
from .config import DiscoveryConfig
from .pipeline import run_photo_discovery

KERNEL_META = {
    "photo_discovery": {"evidence": "估计", "status": "已验"},
}


def _photo_discovery_kernel(job_obj: EchoObject):
    data = job_obj.data
    if isinstance(data, (str, Path)):
        spec = {"job_dir": data}
    elif isinstance(data, dict):
        spec = dict(data)
    else:
        raise ValueError(
            "photo_discovery needs a job directory path or a spec "
            f"dict, got {type(data).__name__}")
    if "job_dir" not in spec:
        raise ValueError("photo_discovery spec dict needs 'job_dir'")
    cfg = DiscoveryConfig()
    overrides = dict(job_obj.meta.get("config", {}))
    plan = overrides.pop("plan", None)
    if plan:
        cfg.plan = {**cfg.plan, **plan}
    if overrides:
        cfg.override(**overrides)
    catalog = load_catalog(spec.get(
        "catalog", job_obj.meta.get("catalog")))
    result = run_photo_discovery(
        spec["job_dir"], catalog, cfg,
        detections=spec.get("detections"),
        masks=spec.get("masks"),
        log=lambda *a: None)
    return job_obj.derive(
        "objects", result, source="kernel:photo_discovery",
        n_objects=len(result["objects"]),
        **KERNEL_META["photo_discovery"])


def register_photo_discovery_kernels(
        registry: Registry | None = None) -> Registry:
    """Register the photo discovery kernel; returns the registry."""
    reg = registry if registry is not None else Registry()
    reg.add(Kernel("photo_discovery", ("job",), "objects",
                   _photo_discovery_kernel, cost=25.0, version="0.3"))
    return reg
