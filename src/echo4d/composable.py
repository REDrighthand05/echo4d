"""M1: fail-closed, typed, bounded reconstruction strategy composition.

Planning is *not* execution.  Declaring an adapter installed is a caller claim;
no data bytes, licenses, inputs, algorithms or geometry are verified here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal

Frame = Literal["none", "camera", "world", "local"]
Scale = Literal["none", "relative", "metric"]
Clock = Literal["none", "frame_pts", "sensor_synced"]
Evidence = Literal["measured", "estimated", "generated", "unknown"]
Partition = Literal["train", "holdout", "unassigned"]
Phase = Literal["build", "evaluate"]
LengthUnit = Literal["none", "m", "cm", "mm"]
TimeUnit = Literal["none", "s", "ms", "us", "ns"]
Handedness = Literal["none", "right", "left"]
PoseDirection = Literal["none", "camera_to_world", "world_to_camera"]

_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True, order=True)
class ArtifactType:
    """A port type. Equality checks ALL units, clocks, frames and evidence.

    Evidence is a provenance *class*, not proof that a source is authentic.
    """

    kind: str
    frame: Frame
    scale: Scale
    clock: Clock
    evidence: Evidence
    partition: Partition
    scene_id: str
    coordinate_id: str
    time_id: str
    length_unit: LengthUnit
    time_unit: TimeUnit
    handedness: Handedness
    pose_direction: PoseDirection

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not _ID.fullmatch(self.kind):
            raise ValueError("artifact kind must be a safe nonempty identifier")
        fields = {
            "frame": {"none", "camera", "world", "local"},
            "scale": {"none", "relative", "metric"},
            "clock": {"none", "frame_pts", "sensor_synced"},
            "evidence": {"measured", "estimated", "generated", "unknown"},
            "partition": {"train", "holdout", "unassigned"},
            "length_unit": {"none", "m", "cm", "mm"},
            "time_unit": {"none", "s", "ms", "us", "ns"},
            "handedness": {"none", "right", "left"},
            "pose_direction": {"none", "camera_to_world", "world_to_camera"},
        }
        for key, allowed in fields.items():
            value = getattr(self, key)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"unsupported artifact {key}: {value!r}")
        for key in ("scene_id", "coordinate_id", "time_id"):
            value = getattr(self, key)
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise ValueError(f"invalid {key}: {value!r}")
        if self.scene_id in {"none", "unknown", "unassigned"}:
            raise ValueError("scene_id must identify a declared scene cohort")
        if (self.frame == "none") != (self.coordinate_id == "none"):
            raise ValueError("coordinate_id must be explicit for spatial ports, none otherwise")
        if (self.clock == "none") != (self.time_id == "none"):
            raise ValueError("time_id must be explicit for timed ports, none otherwise")
        if (self.scale == "metric") != (self.length_unit != "none"):
            raise ValueError("metric scale requires an explicit length unit; nonmetric ports cannot claim units")
        if (self.clock == "none") != (self.time_unit == "none"):
            raise ValueError("timed ports require an explicit timestamp unit")
        if (self.frame == "none") != (self.handedness == "none"):
            raise ValueError("spatial ports require explicit coordinate handedness")
        if (self.kind == "poses") != (self.pose_direction != "none"):
            raise ValueError("pose ports require explicit camera/world transform direction")
        if self.kind in {"depth_z", "depth_ray"} and self.frame != "camera":
            raise ValueError("depth ports must explicitly use camera frame")
        if self.kind == "depth_z" and self.scale == "none":
            raise ValueError("depth_z must declare metric or relative scale")

    def label(self) -> str:
        return "/".join((self.kind, self.frame, self.scale,
                         self.clock, self.evidence, self.partition,
                         self.scene_id, self.coordinate_id, self.time_id,
                         self.length_unit, self.time_unit, self.handedness,
                         self.pose_direction))


@dataclass(frozen=True)
class ModuleSpec:
    """A declared typed transformation; registered!=tested/executed.

    installed=True must only be set by a caller after separately checking an
    installed adapter. It is NOT evidence of a completed experiment.
    """

    module_id: str
    inputs: tuple[ArtifactType, ...]
    outputs: tuple[ArtifactType, ...]
    estimated_units: int
    installed: bool
    revision: str
    phase: Phase = "build"

    def __post_init__(self) -> None:
        if not isinstance(self.module_id, str) or not _ID.fullmatch(self.module_id):
            raise ValueError("module_id must be a safe nonempty identifier")
        for field in ("inputs", "outputs"):
            ports = getattr(self, field)
            if not isinstance(ports, tuple) or not ports or not all(
                    isinstance(a, ArtifactType) for a in ports):
                raise ValueError(f"{field} must be a nonempty tuple of port types")
            if len(ports) != len(set(ports)):
                raise ValueError(f"duplicate {field} port")
        if type(self.estimated_units) is not int or self.estimated_units < 1:
            raise ValueError("estimated_units must be a positive integer")
        if type(self.installed) is not bool:
            raise ValueError("installed must be an explicit boolean")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.revision):
            raise ValueError("module revision must be a pinned 40/64-character hexadecimal digest")
        if self.phase not in ("build", "evaluate"):
            raise ValueError("phase must be build or evaluate")
        if self.phase == "build" and any(a.partition != "train" for a in self.inputs + self.outputs):
            raise ValueError("build modules must reject heldout or unassigned ports")


@dataclass(frozen=True)
class Route:
    modules: tuple[str, ...]
    estimated_units: int
    path_id: str
    status: Literal["not_executed"] = "not_executed"


@dataclass(frozen=True)
class SearchResult:
    routes: tuple[Route, ...]
    visited_states: int
    truncated: bool
    not_executed: bool = True


def compose(
    available: tuple[ArtifactType, ...],
    modules: tuple[ModuleSpec, ...],
    target: ArtifactType,
    *,
    max_steps: int = 5,
    max_states: int = 128,
    max_routes: int = 12,
    budget_units: int = 30,
) -> SearchResult:
    """Enumerate distinct dependency DAGs with bounded search and cost.

    Two legal schedules of the same producer/dependency graph are ONE route.
    Providers, pinned revisions and typed input edges remain part of identity.
    Planning is not execution and never supplies reconstruction quality scores.
    """
    if not isinstance(available, tuple) or not all(isinstance(x, ArtifactType) for x in available):
        raise ValueError("available must be a tuple of ArtifactType")
    if not isinstance(modules, tuple) or not all(isinstance(x, ModuleSpec) for x in modules):
        raise ValueError("modules must be a tuple of ModuleSpec")
    if not isinstance(target, ArtifactType):
        raise ValueError("target must be an ArtifactType")
    for name, value in (("max_steps", max_steps), ("max_states", max_states),
                        ("max_routes", max_routes), ("budget_units", budget_units)):
        if type(value) is not int or value < 1 or value > 10000:
            raise ValueError(f"{name} must be a bounded positive integer")
    ids = [m.module_id for m in modules]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate module identifiers")
    if target.partition == "holdout":
        raise ValueError("holdout cannot be a reconstruction goal")

    from collections import deque
    candidates = tuple(sorted((m for m in modules if m.installed and m.phase == "build"),
                              key=lambda m: m.module_id))
    module_by_id = {m.module_id: m for m in candidates}
    origin = frozenset(available)

    def canonical_route(producers: dict[ArtifactType, str]) -> Route:
        # Only modules on the actual dependency ancestry of the target count.
        # Unrelated side branches are not additional reconstruction strategies.
        used: set[str] = set()
        order: list[str] = []

        def visit(module_id: str) -> None:
            if module_id in used:
                return
            used.add(module_id)
            module = module_by_id[module_id]
            for port in sorted(module.inputs, key=lambda p: p.label()):
                parent = producers.get(port)
                if parent is not None:
                    visit(parent)
            order.append(module_id)

        root = producers.get(target)
        if root is not None:
            visit(root)
        nodes = sorted(used)
        edges = sorted(
            (parent, module_id, port.label())
            for module_id in nodes
            for port in module_by_id[module_id].inputs
            for parent in (producers.get(port),)
            if parent is not None
        )
        descriptor = json.dumps(
            {
                "starting_ports": sorted(port.label() for port in origin),
                "target": target.label(),
                "nodes": [
                    {
                        "id": module_id,
                        "revision": module_by_id[module_id].revision,
                        "inputs": sorted(p.label() for p in module_by_id[module_id].inputs),
                        "outputs": sorted(p.label() for p in module_by_id[module_id].outputs),
                        "estimated_units": module_by_id[module_id].estimated_units,
                    }
                    for module_id in nodes
                ],
                "edges": edges,
            },
            ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        )
        return Route(
            tuple(order),
            sum(module_by_id[module_id].estimated_units for module_id in nodes),
            hashlib.sha256(descriptor.encode()).hexdigest(),
        )

    # Producer provenance is tracked per artifact, rather than assuming that
    # equal output port types imply the same algorithm or dependency graph.
    queue = deque([(origin, (), 0, {})])
    seen_states = {(origin, frozenset(), frozenset())}
    seen_routes: set[str] = set()
    routes: list[Route] = []
    visited = 0
    truncated = False
    while queue:
        if visited >= max_states or len(routes) >= max_routes:
            truncated = True
            break
        types, steps, spent, producers = queue.popleft()
        visited += 1
        if target in types:
            route = canonical_route(producers)
            if route.path_id not in seen_routes:
                seen_routes.add(route.path_id)
                routes.append(route)
            continue
        if len(steps) >= max_steps:
            continue
        for module in candidates:
            if module.module_id in steps or spent + module.estimated_units > budget_units:
                continue
            if not set(module.inputs).issubset(types):
                continue
            newly_produced = frozenset(module.outputs) - types
            if not newly_produced:
                continue
            if any(port.partition == "holdout" for port in newly_produced):
                continue
            new_types = types | newly_produced
            new_steps = steps + (module.module_id,)
            new_producers = dict(producers)
            new_producers.update({port: module.module_id for port in newly_produced})
            # Different schedules with the same selected producers are one
            # intermediate state; different providers remain distinguishable.
            state_key = (
                new_types, frozenset(new_steps), frozenset(new_producers.items()),
            )
            if state_key in seen_states:
                continue
            if len(seen_states) >= max_states:
                truncated = True
                continue
            seen_states.add(state_key)
            queue.append((
                new_types, new_steps, spent + module.estimated_units, new_producers,
            ))
    return SearchResult(tuple(routes), visited, truncated)
