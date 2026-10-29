"""Real TUM room 96-pose RGB witness preflight; never opens measured Z."""
import argparse
import json
from pathlib import Path
import numpy as np

from tum_room_sparse_quality import select_training
from tum_room_knownpose_multiview import choose_neighbors
from tum_room_pose_overlap_neighbors import choose_overlap_neighbors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.source / "replay-inputs/source/recording.json").read_text())
    _, selected, receipt = select_training(manifest)
    poses = [np.asarray(f["cameraToWorld"], dtype=float).reshape(4, 4, order="F")
             for f in selected["frames"]]
    old_missing = [i for i in range(96) if len(choose_neighbors(poses, i)) < 2]
    new_missing = [i for i in range(96) if len(choose_overlap_neighbors(poses, i)) < 2]
    rescued = sorted(set(old_missing) - set(new_missing))
    result = {"scene": "real TUM fr1_room", "source": "96 known training poses only",
              "source_indices": receipt["expanded_indices"],
              "old_missing_two_witnesses": old_missing,
              "new_missing_two_witnesses": new_missing,
              "pose_only_rescued": rescued,
              "note": "Pose overlap is not photometric depth or Mesh evidence."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"old_missing": len(old_missing), "new_missing": len(new_missing),
                      "pose_only_rescued": len(rescued)}), flush=True)
    if len(old_missing) != 22 or len(new_missing) > 4:
        raise RuntimeError("Frozen real 96-pose cohort or witness policy drift")


if __name__ == "__main__":
    main()
