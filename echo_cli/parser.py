"""Command-line argument parser definitions for echo_cli."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser for echo_cli."""
    parser = argparse.ArgumentParser(
        prog="echo-cli",
        description="ECHO 4D Spatial Reconstruction - Headless CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # --- Subcommand: run ---
    run_parser = subparsers.add_parser(
        "run",
        help="Execute the 12-stage 3D reconstruction pipeline on a video",
    )
    run_parser.add_argument("video", type=str, help="Path to input video file")
    run_parser.add_argument(
        "--jobs-root",
        type=str,
        default=None,
        help="Custom directory to store job outputs",
    )
    run_parser.add_argument(
        "--export",
        action="store_true",
        help="Automatically export solid GLB model on completion",
    )
    run_parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Export destination folder (if --export is specified)",
    )

    # --- Subcommand: depth ---
    depth_parser = subparsers.add_parser(
        "depth",
        help="Generate depth estimation video from an input video",
    )
    depth_parser.add_argument("video", type=str, help="Path to input video file")

    # --- Subcommand: serve ---
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the local ECHO REST API service",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    serve_parser.add_argument(
        "--jobs-root",
        type=str,
        default=None,
        help="Custom directory to store/query job outputs",
    )

    # --- Subcommand: export ---
    export_parser = subparsers.add_parser(
        "export",
        help="Export reconstructed models or camera trajectory from a job",
    )
    export_parser.add_argument("job", type=str, help="Path to reconstructed job folder")
    export_parser.add_argument(
        "--format",
        choices=["glb", "ply", "route"],
        default="glb",
        help="Export format (glb, ply, route)",
    )
    export_parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Target folder to write exported files",
    )

    # --- Subcommand: status ---
    status_parser = subparsers.add_parser(
        "status",
        help="Display past reconstruction jobs and current system status",
    )
    status_parser.add_argument(
        "--jobs-root",
        type=str,
        default=None,
        help="Directory where jobs are stored",
    )
    status_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of recent jobs to show (default: 10)",
    )

    return parser
