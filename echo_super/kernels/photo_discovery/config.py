"""photo_discovery configuration — every tunable of the pipeline in one
place (the experiments had them scattered as module constants across
detect2 / segment2 / lift / lift2 / lift3_cache / finalize3).

Defaults are the round-3 final form, value for value:
  detect2.py  THRESH / IOU_CLUSTER / CONTAIN_CLUSTER / MAX_PER_CLASS /
              gamma 0.45
  segment2.py MAX_BOXES_PER_FRAME
  lift.py     STRIDE / VOXEL / CORR_VOX 0.1 / NN gate 0.15 / room 1-99%
              + 5% pad / degenerate 0.40
  lift2.py    depth_crop (40 bins, 2-98 pct, max(2 bins, 15% peak))
  finalize3.py isolation voxel 0.3 + per-class PLAN
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiscoveryConfig:
    # --- stage 1: open-vocabulary detection (OWLv2) ---
    owl_model: str = "google/owlv2-base-patch16-ensemble"
    detect_threshold: float = 0.10
    gamma: float = 0.45                # dark-frame lift before detection
    converge_iou: float = 0.3          # box-convergence clustering
    converge_containment: float = 0.5
    max_clusters_per_class: int = 2    # converged clusters kept per class/frame

    # --- stage 2: segmentation (SAM) ---
    sam_model: str = "facebook/sam-vit-base"
    sam_max_boxes_per_frame: int = 8   # round-robin budget across classes

    # --- stage 3: lift + solidify ---
    stride: int = 3                    # pixel subsample before lifting
    degenerate_frac: float = 0.40      # mask > frac of frame = SAM failure
    crop_bins: int = 40                # depth main-cluster histogram
    crop_pct_lo: float = 2.0
    crop_pct_hi: float = 98.0
    crop_window_bins: float = 2.0      # keep peak +- max(N bins, frac*peak)
    crop_window_frac: float = 0.15
    crop_min_px: int = 30
    dedup_voxel: float = 0.02
    corr_voxel: float = 0.1            # 2-frame corroboration voxel
    min_frames: int = 2                # distinct frames per corroborated voxel
    surface_nn_max: float = 0.15       # NN distance to the job cloud
    room_pct_lo: float = 1.0           # room envelope percentiles of the cloud
    room_pct_hi: float = 99.0
    room_pad_frac: float = 0.05

    # --- stage 4: instance isolation (round 3) ---
    isolation_voxel: float = 0.3

    # Per-class treatment plan (round-3 finalize3.PLAN): keep_rank=None
    # keeps every converged cluster's masks; isolate=True delivers only
    # the largest isolation_voxel-connected component of the
    # corroborated set. Classes not listed get the default treatment.
    plan: dict = field(default_factory=lambda: {
        "monitor": {"keep_rank": 1, "isolate": True},
        "chair": {"keep_rank": None, "isolate": True},
    })
    default_treatment: dict = field(
        default_factory=lambda: {"keep_rank": None, "isolate": False})

    def treatment_for(self, label: str) -> dict:
        t = dict(self.default_treatment)
        t.update(self.plan.get(label, {}))
        return t

    def override(self, **kw) -> "DiscoveryConfig":
        for k, v in kw.items():
            if not hasattr(self, k):
                raise ValueError(f"unknown DiscoveryConfig field {k!r}")
            setattr(self, k, v)
        return self
