"""CLI: progress and errors to stderr, successful final JSON to stdout."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
from . import __version__
from .demo import create_demo
from .evaluation import compare_points
from .fusion import FusionConfig, TSDFVolume
from .model import DataError, file_sha256, load_recording
from .output import mesh_summary, output_transaction, preview_recording, write_json, write_ply, write_scene


def reconstruct_command(args: argparse.Namespace) -> dict:
    recording = load_recording(args.recording)
    cfg = FusionConfig(tuple(args.bounds), args.voxel,
        args.voxel*3 if args.truncation is None else args.truncation,
        max_depth=args.max_depth, max_voxels=args.max_voxels, min_observations=args.min_observations)
    start = time.perf_counter()
    volume = TSDFVolume(cfg)
    evidence, integration = [], []
    with output_transaction(args.output) as output:
        for index, frame, item in recording.iter_frames(args.stride, args.limit):
            touched = volume.integrate(frame, recording.camera, index)
            evidence.append(item)
            integration.append({"frame":index, "t":frame.t, "updated_voxels":touched})
            if not args.quiet:
                print(f"frame {index}: {touched:,} voxel updates", file=sys.stderr)
        mesh = volume.extract_mesh()
        write_ply(output / "mesh.ply", mesh)
        np.save(output / "vertices.npy", mesh.vertices, allow_pickle=False)
        np.savez_compressed(output / "evidence.npz", observations=mesh.observations,
                            first_frame=mesh.first_frame, last_frame=mesh.last_frame)
        report = {"version":"echo-report-1", "software_version":__version__, "algorithm":"projective-tsdf-v1",
            "source":recording.source, "manifest_sha256":recording.manifest_sha256, "config":asdict(cfg),
            "stride":args.stride, "frames_integrated":volume.integrated, "volume_bytes":volume.allocated_bytes,
            "mesh":mesh_summary(mesh), "integration":integration, "input_evidence":evidence,
            "limitations":["Known calibrated camera poses required", "Static scene or supplied exclusion masks",
                           "Bounded CPU volume; no SLAM or loop closure", "No automatic person detection"]}
        if args.reference:
            reference = np.load(args.reference, allow_pickle=False, mmap_mode="r")
            report["independent_reference"] = {"sha256":file_sha256(args.reference),
                                              "metrics":compare_points(mesh.vertices, reference, args.threshold)}
        if not args.no_scene:
            write_scene(output / "scene.echo.json", recording, mesh, report)
        if args.save_volume:
            np.savez_compressed(output / "volume.npz", tsdf=volume.tsdf, weight=volume.weight,
                color=volume.color, color_weight=volume.color_weight, first=volume.first, last=volume.last,
                origin=volume.origin, config=np.array(json.dumps(asdict(cfg))))
        report["elapsed_seconds"] = time.perf_counter()-start
        report["outputs"] = {p.name:file_sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
        write_json(output / "report.json", report)
    return {"output":str(args.output.resolve()), "frames":volume.integrated, "vertices":len(mesh.vertices),
            "triangles":len(mesh.faces), "elapsed_seconds":report["elapsed_seconds"],
            "mesh_sha256":report["outputs"]["mesh.ply"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="echo4d", description="Calibrated RGB-D to observed surfaces. Known poses required.")
    parser.add_argument("--version", action="version", version=f"echo4d {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate analytic RGB-D fixture, not real sensor data")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--frames", type=int, default=16)
    demo.add_argument("--noise", type=float, default=0)
    validate = sub.add_parser("validate", help="Read and validate all frames")
    validate.add_argument("recording", type=Path)
    rec = sub.add_parser("reconstruct", help="Fuse known-pose depth frames in a bounded TSDF")
    rec.add_argument("recording", type=Path)
    rec.add_argument("--output", required=True, type=Path)
    rec.add_argument("--bounds", required=True, nargs=6, type=float, metavar=("XMIN","YMIN","ZMIN","XMAX","YMAX","ZMAX"))
    rec.add_argument("--voxel", type=float, default=.04)
    rec.add_argument("--truncation", type=float)
    rec.add_argument("--max-depth", type=float, default=8)
    rec.add_argument("--max-voxels", type=int, default=4_000_000)
    rec.add_argument("--min-observations", type=int, default=1)
    rec.add_argument("--stride", type=int, default=1)
    rec.add_argument("--limit", type=int)
    rec.add_argument("--reference", type=Path, help="Independent aligned Nx3 .npy reference")
    rec.add_argument("--threshold", type=float, default=.05)
    rec.add_argument("--no-scene", action="store_true")
    rec.add_argument("--save-volume", action="store_true")
    rec.add_argument("--quiet", action="store_true")
    preview = sub.add_parser("preview", help="Export compact echo-1 browser recording")
    preview.add_argument("recording", type=Path)
    preview.add_argument("--output", required=True, type=Path)
    tum = sub.add_parser("import-tum", help="Convert TUM RGB-D and known camera trajectory")
    tum.add_argument("dataset", type=Path)
    tum.add_argument("--output", required=True, type=Path)
    tum.add_argument("--intrinsics", nargs=4, type=float, required=True, metavar=("FX","FY","CX","CY"))
    tum.add_argument("--size", nargs=2, type=int, default=[640,480])
    tum.add_argument("--poses", default="groundtruth.txt")
    tum.add_argument("--depth-scale", type=float, default=5000)
    tum.add_argument("--stride", type=int, default=5)
    tum.add_argument("--limit", type=int, default=120)
    tum.add_argument("--downsample", type=int, default=2)
    tum.add_argument("--max-pose-gap", type=float, default=.1)
    tum.add_argument("--max-color-gap", type=float, default=.03)
    tum.add_argument("--world-up", choices=["y","z"], default="y", help="y preserves axes; z rotates input Z-up to viewer Y-up")
    colmap = sub.add_parser("import-colmap", help="One strict COLMAP + independently measured scale + metric Z-depth input")
    colmap.add_argument("--model", required=True, type=Path, help="COLMAP TEXT model (undistorted PINHOLE)")
    colmap.add_argument("--images", required=True, type=Path, help="RGB image directory")
    colmap.add_argument("--depth", required=True, type=Path, help="Pre-registered camera-Z depth NPY directory (metres)")
    colmap.add_argument("--timestamps", required=True, type=Path, help="Exact name,t CSV matching registered images")
    colmap.add_argument("--meters-per-colmap-unit", required=True, type=float,
                        help="Measured metres for one COLMAP world unit; GT post-hoc alignment is forbidden")
    colmap.add_argument("--scale-source", required=True,
                        help="Traceable independent real-world measurement source, not evaluation GT")
    colmap.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            create_demo(args.output, args.frames, args.noise)
            result = {"output":str(args.output.resolve()), "source":"synthetic"}
        elif args.command == "validate":
            recording = load_recording(args.recording)
            count = valid = 0
            for _, frame, _ in recording.iter_frames():
                count += 1
                valid += int(np.count_nonzero(frame.depth))
            result = {"valid":True, "frames":count, "depth_pixels":valid,
                      "source":recording.source, "intrinsics":recording.camera.to_dict()}
        elif args.command == "reconstruct":
            result = reconstruct_command(args)
        elif args.command == "preview":
            recording = load_recording(args.recording)
            with output_transaction(args.output) as output:
                write_json(output / "recording.echo.json", preview_recording(recording))
            result = {"output":str(args.output.resolve())}
        elif args.command == "import-colmap":
            # One strict adapter. Uses #6 + #24 hardening, NOT the incompatible
            # second converter implementation from draft PR #19.
            from .colmap import import_colmap
            result = import_colmap(args)
        else:
            from .tum import import_tum
            result = import_tum(args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (DataError, OSError, ValueError, MemoryError) as error:
        print(json.dumps({"error":str(error), "type":type(error).__name__}, ensure_ascii=False), file=sys.stderr)
        return 2
    except (KeyboardInterrupt, InterruptedError):
        print('{"error":"Cancelled; no completed output published"}', file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
