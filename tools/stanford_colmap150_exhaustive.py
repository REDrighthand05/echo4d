"""Independent 150-frame Stanford Lounge RGB-D reconstruction with COLMAP exhaustive RGB matching.

Changes only the RGB image pair matching topology relative to the sequential
150-frame baseline. No prior pose, existing mesh, predicted depth or extra RGB-D
frames are used. This deliberately targets missing loop/non-neighbor matches.
"""
import argparse
import json
from pathlib import Path

import stanford_colmap_metric300 as base

_original_command = base.command


def exhaustive_command(argv, log, env):
    argv = list(argv)
    if len(argv) >= 2 and argv[0] == 'colmap' and argv[1] == 'sequential_matcher':
        db = argv[argv.index('--database_path') + 1]
        argv = ['colmap', 'exhaustive_matcher', '--database_path', db,
                '--SiftMatching.use_gpu', '0',
                '--ExhaustiveMatching.block_size', '25']
    return _original_command(argv, log, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    base.command = exhaustive_command
    base.run(args.output, 150)
    receipt = args.output / 'receipt.json'
    record = json.loads(receipt.read_text())
    record['rgb_matching'] = 'COLMAP exhaustive SIFT pairs among only the selected 150 images'
    record['baseline'] = 'sequential COLMAP 150, artifact 12920054039, 93 registered frames'
    record['visual_acceptance'] = 'NOT REVIEWED: inspect actual three recorded plus two free views'
    receipt.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
