"""M2 experimental typed-graph executor: real adapter invocation, no quality claims.

Only explicitly bound providers may run. A planner's installed flag is never
accepted as evidence of a working implementation on its own.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Callable, Mapping

from .composable import ArtifactType, ModuleSpec, Route, compose
from .model import file_sha256, load_recording, safe_path
from .output import output_transaction, write_json


class GraphBlocked(ValueError):
    """Execution is blocked before a trusted adapter can run."""


class GraphFailed(RuntimeError):
    """An adapter failed; no completed graph output was published."""


@dataclass(frozen=True)
class InputFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class Adapter:
    """Provider binding; source_file must hash to its ModuleSpec revision."""

    module_id: str
    source_file: Path
    invoke: Callable[[Mapping[ArtifactType, Path], Path], Mapping[ArtifactType, Path]]
    config: dict = field(default_factory=dict)


def _json_bytes(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _file(path: Path, root: Path | None = None) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise GraphBlocked(f"Missing, non-regular or symlink artifact: {path}")
    resolved = path.resolve()
    if root is not None and not resolved.is_relative_to(root.resolve()):
        raise GraphBlocked(f"Adapter returned an artifact outside its workspace: {path}")
    return resolved


def _checkout_sha(expected: str) -> str:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise GraphBlocked("A 40-hex immutable Git checkout SHA is required")
    root = Path(__file__).resolve().parents[2]
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5, text=True).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise GraphBlocked("Cannot verify actual Git checkout SHA") from error
    if actual != expected:
        raise GraphBlocked(f"Git checkout mismatch: expected {expected}, actual {actual}")
    return actual


def execute_graph(
    route: Route,
    modules: tuple[ModuleSpec, ...],
    adapters: Mapping[str, Adapter],
    inputs: Mapping[ArtifactType, InputFile],
    target: ArtifactType,
    destination: Path,
    *,
    checkout_sha: str,
) -> dict:
    """Execute an M1 route atomically, or leave a separate failure receipt.

    A completed file-level run is NOT a geometric or human-visual PASS.
    """
    destination = Path(destination)
    if destination.exists():
        raise GraphBlocked("Destination already exists; never overwrite experiments")
    failure = destination.with_name(destination.name + ".failed.json")
    if failure.exists():
        raise GraphBlocked("Previous failure receipt exists; use a new experiment destination")
    attempted = ""
    input_hashes: dict[str, str] = {}
    dependency_hashes: dict[str, dict[str, str]] = {}
    dependency_paths: list[tuple[Path, str]] = []
    try:
        actual_checkout = _checkout_sha(checkout_sha)
        if not isinstance(route, Route) or not isinstance(target, ArtifactType):
            raise GraphBlocked("A selected typed route and target are required")
        if not isinstance(inputs, Mapping) or not inputs:
            raise GraphBlocked("At least one pinned input artifact is required")
        if len(modules) > 32:
            raise GraphBlocked("Module registry exceeds experimental limit")
        by_id = {module.module_id: module for module in modules}
        if len(by_id) != len(modules):
            raise GraphBlocked("Duplicate module IDs")
        plan = compose(tuple(inputs), modules, target, max_steps=16,
                       max_states=2048, max_routes=512, budget_units=10000)
        if not any(r.path_id == route.path_id and r.modules == route.modules
                   for r in plan.routes):
            raise GraphBlocked("Selected graph is not reachable under the pinned registry")
        if not route.modules:
            raise GraphBlocked("Zero-step identity route is not an executed reconstruction")
        for port, item in inputs.items():
            if not isinstance(port, ArtifactType) or not isinstance(item, InputFile):
                raise GraphBlocked("Inputs must be typed, hash-pinned InputFile entries")
            if not re.fullmatch(r"[0-9a-f]{64}", item.sha256):
                raise GraphBlocked("Input SHA-256 must be a lowercase 64-hex digest")
            source = _file(item.path)
            digest = file_sha256(source)
            if digest != item.sha256:
                raise GraphBlocked(f"Input hash mismatch for {port.label()}")
            input_hashes[port.label()] = digest
            # The manifest hash alone does not pin referenced RGB/depth bytes.
            # Hash the entire actual source closure and include it in the run ID.
            if port.kind == "echo_recording":
                recording = load_recording(source)
                dependencies: dict[str, str] = {}
                if not recording.inline:
                    for index, entry in enumerate(recording.entries):
                        for key in ("depth", "color", "mask"):
                            if key not in entry:
                                continue
                            if len(dependencies) >= 20000:
                                raise GraphBlocked("Too many source dependencies to audit")
                            image = safe_path(recording.root, entry[key])
                            sha = file_sha256(image)
                            dependencies[f"{index}:{key}:{entry[key]}"] = sha
                            dependency_paths.append((image, sha))
                dependency_hashes[port.label()] = dependencies

        provider_hashes: dict[str, str] = {}
        configs: dict[str, str] = {}
        for module_id in route.modules:
            module = by_id[module_id]
            adapter = adapters.get(module_id)
            if not module.installed or adapter is None or adapter.module_id != module_id:
                raise GraphBlocked(f"No verified installed provider for {module_id}")
            source = _file(adapter.source_file)
            digest = file_sha256(source)
            if digest != module.revision or len(module.revision) != 64:
                raise GraphBlocked(f"Provider source revision mismatch for {module_id}")
            if not callable(adapter.invoke):
                raise GraphBlocked(f"Provider is not callable: {module_id}")
            provider_hashes[module_id] = digest
            configs[module_id] = hashlib.sha256(_json_bytes(adapter.config)).hexdigest()

        artifacts: dict[ArtifactType, Path] = {
            port: _file(item.path) for port, item in inputs.items()
        }
        produced: dict[str, dict] = {}
        with output_transaction(destination) as workspace:
            for step, module_id in enumerate(route.modules):
                attempted = module_id
                module = by_id[module_id]
                adapter = adapters[module_id]
                if any(port not in artifacts for port in module.inputs):
                    raise GraphBlocked(f"Missing typed input at execution: {module_id}")
                step_dir = workspace / f"{step:02d}_{module_id}"
                step_dir.mkdir()
                try:
                    output = adapter.invoke(
                        {port: artifacts[port] for port in module.inputs}, step_dir)
                except Exception as error:
                    raise GraphFailed(f"Provider {module_id} failed: {type(error).__name__}: {error}") from error
                if not isinstance(output, Mapping) or set(output) != set(module.outputs):
                    raise GraphFailed(f"Provider {module_id} returned incorrect output ports")
                # Adapters may read earlier artifacts, but must not rewrite them.
                for port in module.inputs:
                    path = _file(artifacts[port])
                    expected = (produced[port.label()]["sha256"]
                                if port.label() in produced else input_hashes[port.label()])
                    if file_sha256(path) != expected:
                        raise GraphFailed(f"Typed input artifact changed during execution: {module_id}")
                for port, raw_path in output.items():
                    try:
                        path = _file(Path(raw_path), step_dir)
                    except (TypeError, ValueError, GraphBlocked) as error:
                        raise GraphFailed(f"Invalid output artifact for {module_id}: {error}") from error
                    # Match the planner: the first producer of a port wins.
                    # Multi-output providers cannot silently replace an existing edge.
                    if port in artifacts:
                        continue
                    artifacts[port] = path
                    produced[port.label()] = {
                        "provider": module_id,
                        "path": str(path.relative_to(workspace)),
                        "sha256": file_sha256(path),
                        "bytes": path.stat().st_size,
                    }
                if file_sha256(_file(adapter.source_file)) != provider_hashes[module_id]:
                    raise GraphFailed(f"Provider source changed during execution: {module_id}")
                if hashlib.sha256(_json_bytes(adapter.config)).hexdigest() != configs[module_id]:
                    raise GraphFailed(f"Provider config changed during execution: {module_id}")
            for port, item in inputs.items():
                if file_sha256(_file(item.path)) != input_hashes[port.label()]:
                    raise GraphFailed(f"Input changed during execution: {port.label()}")
            for source_path, expected_sha in dependency_paths:
                if file_sha256(_file(source_path)) != expected_sha:
                    raise GraphFailed(f"Referenced source image changed during execution: {source_path.name}")
            if target not in artifacts or target.label() not in produced:
                raise GraphFailed("Target Mesh was not actually produced by a provider")
            receipt = {
                "version": "echo-graph-execution-1",
                "status": "executed_unscored",
                "quality": "not_evaluated",
                "path_id": route.path_id,
                "modules": list(route.modules),
                "checkout_sha": actual_checkout,
                "input_sha256": input_hashes,
                "input_dependencies_sha256": dependency_hashes,
                "provider_sha256": provider_hashes,
                "config_sha256": configs,
                "outputs": produced,
                "target": target.label(),
            }
            receipt["experiment_id"] = hashlib.sha256(_json_bytes({
                "path_id": route.path_id, "checkout": actual_checkout,
                "inputs": input_hashes, "source_dependencies": dependency_hashes,
                "providers": provider_hashes, "configs": configs,
            })).hexdigest()
            write_json(workspace / "execution.json", receipt)
        return receipt
    except Exception as error:
        status = "blocked" if isinstance(error, GraphBlocked) else "failed"
        if not failure.exists():
            write_json(failure, {
                "version": "echo-graph-failure-1",
                "status": status,
                "path_id": getattr(route, "path_id", None),
                "checkout_sha_claim": checkout_sha,
                "attempted_module": attempted or None,
                "input_sha256": input_hashes,
                "input_dependencies_sha256": dependency_hashes,
                "error_type": type(error).__name__,
                "error": str(error),
            })
        raise


def tsdf_recording_adapter(
    module_id: str,
    source_file: Path,
    recording_port: ArtifactType,
    mesh_port: ArtifactType,
    *,
    bounds: tuple[float, float, float, float, float, float],
    voxel: float = 0.08,
    max_voxels: int = 4_000_000,
) -> Adapter:
    """Bind existing known-pose RGB-D CPU TSDF, not a SLAM/MVS provider."""
    from .fusion import FusionConfig
    cfg = FusionConfig(tuple(bounds), voxel, voxel * 3, max_voxels=max_voxels)
    if recording_port.kind != "echo_recording" or mesh_port.kind != "mesh":
        raise GraphBlocked("TSDF binding requires echo_recording -> mesh")
    if (recording_port.scale != "metric" or mesh_port.scale != "metric"
            or recording_port.length_unit != "m" or mesh_port.length_unit != "m"
            or recording_port.scene_id != mesh_port.scene_id
            or recording_port.coordinate_id != mesh_port.coordinate_id
            or recording_port.handedness != "right" or mesh_port.handedness != "right"
            or recording_port.partition != "train" or mesh_port.partition != "train"):
        raise GraphBlocked("TSDF binding requires aligned right-handed metric train data")
    config = {"bounds": list(cfg.bounds), "voxel": cfg.voxel,
              "max_voxels": cfg.max_voxels, "algorithm": "projective-tsdf-v1"}

    def invoke(data: Mapping[ArtifactType, Path], work: Path) -> Mapping[ArtifactType, Path]:
        from .cli import reconstruct_command
        args = argparse.Namespace(
            recording=data[recording_port], output=work / "result",
            bounds=list(cfg.bounds), voxel=cfg.voxel, truncation=None,
            max_depth=cfg.max_depth, max_voxels=cfg.max_voxels,
            min_observations=cfg.min_observations, stride=1, limit=None,
            reference=None, threshold=.05, no_scene=False, save_volume=False,
            quiet=True,
        )
        result = reconstruct_command(args)
        return {mesh_port: Path(result["output"]) / "mesh.ply"}

    return Adapter(module_id, Path(source_file), invoke, config)



def projective_recording_adapter(
    module_id: str,
    source_file: Path,
    recording_port: ArtifactType,
    mesh_port: ArtifactType,
    *,
    pixel_stride: int = 4,
    frame_stride: int = 1,
    max_depth: float = 6.0,
    max_depth_jump: float = 0.12,
    max_edge_length: float = 0.30,
    max_vertices: int = 300_000,
    max_faces: int = 500_000,
) -> Adapter:
    """Bind independent per-frame depth triangles; never claim TSDF fusion.

    The adapter source must be pinned to projective_meshing.py, not this
    orchestration module. Runtime source files remain pinned by execute_graph.
    """
    from . import projective_meshing
    from .model import DataError
    from .output import mesh_summary, write_ply
    if Path(source_file).resolve() != Path(projective_meshing.__file__).resolve():
        raise GraphBlocked("Projective provider must pin the actual algorithm source")
    if recording_port.kind != "echo_recording" or mesh_port.kind != "mesh":
        raise GraphBlocked("Projective binding requires echo_recording -> mesh")
    if (recording_port.scale != "metric" or mesh_port.scale != "metric"
            or recording_port.length_unit != "m" or mesh_port.length_unit != "m"
            or recording_port.scene_id != mesh_port.scene_id
            or recording_port.coordinate_id != mesh_port.coordinate_id
            or recording_port.handedness != "right" or mesh_port.handedness != "right"
            or recording_port.partition != "train" or mesh_port.partition != "train"):
        raise GraphBlocked("Projective binding requires aligned right-handed metric train data")
    config = {
        "algorithm": "per-frame-projective-triangles-v1",
        "pixel_stride": pixel_stride, "frame_stride": frame_stride,
        "max_depth": max_depth, "max_depth_jump": max_depth_jump,
        "max_edge_length": max_edge_length, "max_vertices": max_vertices,
        "max_faces": max_faces,
    }
    # Reject invalid configurations at binding time; do not defer to a 450MB run.
    for key in ("pixel_stride", "frame_stride", "max_vertices", "max_faces"):
        if type(config[key]) is not int or config[key] < 1:
            raise GraphBlocked(f"Invalid projective {key}")
    if pixel_stride > 16:
        raise GraphBlocked("Projective pixel_stride exceeds limit")
    import math
    for key in ("max_depth", "max_depth_jump", "max_edge_length"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise GraphBlocked(f"Invalid projective {key}")
    if max_depth > 100:
        raise GraphBlocked("Invalid projective max_depth")

    def invoke(data: Mapping[ArtifactType, Path], work: Path) -> Mapping[ArtifactType, Path]:
        recording = load_recording(data[recording_port])
        mesh, dependencies = projective_meshing.projective_mesh(
            recording, pixel_stride=pixel_stride, frame_stride=frame_stride,
            max_depth=max_depth, max_depth_jump=max_depth_jump,
            max_edge_length=max_edge_length, max_vertices=max_vertices,
            max_faces=max_faces)
        mesh_path = work / "mesh.ply"
        write_ply(mesh_path, mesh,
                  comment="ECHO per-frame projective depth triangles; metres; known poses")
        write_json(work / "provider-report.json", {
            "schema": "echo-projective-graph-provider-1",
            "status": "executed_unscored", "quality": "not_evaluated",
            "algorithm": "per-frame-projective-triangles-v1",
            "fusion": False, "source_claim": recording.source,
            "source_dependencies_sha256": dependencies,
            "geometry": mesh_summary(mesh),
            "mesh_sha256": file_sha256(mesh_path),
            "parameters": config,
            "independent_surface_truth": False,
            "human_visual_review": False,
        })
        return {mesh_port: mesh_path}

    return Adapter(module_id, Path(source_file), invoke, config)
