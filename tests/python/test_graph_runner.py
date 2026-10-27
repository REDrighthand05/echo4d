"""M2 graph execution contracts; synthetic smoke is NOT real scene acceptance."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from echo4d.composable import ArtifactType, ModuleSpec, compose
from echo4d.demo import create_demo
from echo4d.graph_runner import (
    Adapter, GraphBlocked, GraphFailed, InputFile, execute_graph, tsdf_recording_adapter,
)
from echo4d.model import file_sha256


def ports():
    common = dict(scene_id="fixture", coordinate_id="fixture_space",
                  length_unit="m", handedness="right", partition="train")
    recording = ArtifactType("echo_recording", "world", "metric", "frame_pts",
                             "unknown", time_id="fixture_clock", time_unit="s",
                             pose_direction="none", **common)
    mesh = ArtifactType("mesh", "world", "metric", "none",
                        "estimated", time_id="none", time_unit="none",
                        pose_direction="none", **common)
    return recording, mesh


def checkout():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def spec(name, inp, out, source, installed=True):
    return ModuleSpec(name, (inp,), (out,), 1, installed,
                      file_sha256(source))


def synthetic_input(tmp_path):
    data = tmp_path / "data"
    create_demo(data, frames=4)
    return data / "recording.json"


def plan(input_port, output_port, module):
    routes = compose((input_port,), (module,), output_port)
    assert len(routes.routes) == 1
    return routes.routes[0]


def test_actual_tsdf_adapter_executes_synthetic_mesh_and_pins_receipt(tmp_path):
    manifest = synthetic_input(tmp_path)
    source = Path(__import__("echo4d.graph_runner", fromlist=["x"]).__file__)
    input_port, mesh_port = ports()
    module = spec("known_pose_tsdf", input_port, mesh_port, source)
    route = plan(input_port, mesh_port, module)
    adapter = tsdf_recording_adapter(module.module_id, source, input_port, mesh_port,
                                     bounds=(-1.8, -.2, -1.9, 1.8, 2.2, 1.8))
    output = tmp_path / "run"
    receipt = execute_graph(
        route, (module,), {module.module_id: adapter},
        {input_port: InputFile(manifest, file_sha256(manifest))},
        mesh_port, output, checkout_sha=checkout())
    assert receipt["status"] == "executed_unscored"
    assert receipt["quality"] == "not_evaluated"
    assert receipt["checkout_sha"] == checkout()
    assert receipt["path_id"] == route.path_id
    produced = receipt["outputs"][mesh_port.label()]
    assert produced["bytes"] > 100
    assert file_sha256(output / produced["path"]) == produced["sha256"]
    assert (output / "execution.json").is_file()
    assert not (tmp_path / "run.failed.json").exists()
    report = json.loads((output / "00_known_pose_tsdf" / "result" / "report.json").read_text())
    assert report["source"] == "synthetic" and report["mesh"]["triangles"] > 100


def test_missing_provider_is_blocked_and_never_publishes_mesh(tmp_path):
    manifest = synthetic_input(tmp_path)
    source = Path(__import__("echo4d.graph_runner", fromlist=["x"]).__file__)
    inp, out = ports()
    module = spec("not_bound", inp, out, source)
    with pytest.raises(GraphBlocked, match="No verified installed provider"):
        execute_graph(plan(inp, out, module), (module,), {},
                      {inp: InputFile(manifest, file_sha256(manifest))},
                      out, tmp_path / "run", checkout_sha=checkout())
    assert not (tmp_path / "run").exists()
    failure = json.loads((tmp_path / "run.failed.json").read_text())
    assert failure["status"] == "blocked"


def test_tampered_input_sha_blocks_before_execution(tmp_path):
    manifest = synthetic_input(tmp_path)
    source = Path(__import__("echo4d.graph_runner", fromlist=["x"]).__file__)
    inp, out = ports()
    module = spec("tsdf", inp, out, source)
    adapter = tsdf_recording_adapter("tsdf", source, inp, out,
                                     bounds=(-1.8, -.2, -1.9, 1.8, 2.2, 1.8))
    with pytest.raises(GraphBlocked, match="Input hash mismatch"):
        execute_graph(plan(inp, out, module), (module,), {"tsdf": adapter},
                      {inp: InputFile(manifest, "0" * 64)},
                      out, tmp_path / "run", checkout_sha=checkout())
    assert not (tmp_path / "run").exists()


def test_unpinned_provider_file_is_blocked(tmp_path):
    manifest = synthetic_input(tmp_path)
    inp, out = ports()
    pinned = tmp_path / "provider.py"
    pinned.write_text("original")
    module = spec("tsdf", inp, out, pinned)
    pinned.write_text("modified")
    adapter = Adapter("tsdf", pinned, lambda inputs, work: {})
    with pytest.raises(GraphBlocked, match="source revision mismatch"):
        execute_graph(plan(inp, out, module), (module,), {"tsdf": adapter},
                      {inp: InputFile(manifest, file_sha256(manifest))},
                      out, tmp_path / "run", checkout_sha=checkout())


def test_adapter_cannot_publish_artifact_outside_workspace(tmp_path):
    manifest = synthetic_input(tmp_path)
    inp, out = ports()
    pinned = tmp_path / "provider.py"
    pinned.write_text("provider")
    outside = tmp_path / "outside.ply"
    outside.write_text("untrusted")
    module = spec("escape", inp, out, pinned)
    adapter = Adapter("escape", pinned, lambda inputs, work: {out: outside})
    with pytest.raises(GraphFailed, match="Invalid output artifact"):
        execute_graph(plan(inp, out, module), (module,), {"escape": adapter},
                      {inp: InputFile(manifest, file_sha256(manifest))},
                      out, tmp_path / "run", checkout_sha=checkout())
    assert not (tmp_path / "run").exists()
    failure = json.loads((tmp_path / "run.failed.json").read_text())
    assert failure["status"] == "failed" and failure["attempted_module"] == "escape"


def test_wrong_checkout_sha_is_blocked(tmp_path):
    manifest = synthetic_input(tmp_path)
    inp, out = ports()
    pinned = tmp_path / "provider.py"
    pinned.write_text("provider")
    module = spec("tsdf", inp, out, pinned)
    with pytest.raises(GraphBlocked, match="Git checkout mismatch"):
        execute_graph(plan(inp, out, module), (module,), {},
                      {inp: InputFile(manifest, file_sha256(manifest))},
                      out, tmp_path / "run", checkout_sha="0" * 40)


def test_not_installed_module_never_becomes_a_real_route(tmp_path):
    inp, out = ports()
    pinned = tmp_path / "provider.py"
    pinned.write_text("provider")
    module = spec("future", inp, out, pinned, installed=False)
    assert not compose((inp,), (module,), out).routes


def test_second_provider_is_distinct_route_but_unbound_is_not_executable(tmp_path):
    manifest = synthetic_input(tmp_path)
    inp, out = ports()
    pinned = tmp_path / "provider.py"
    pinned.write_text("provider")
    a = spec("a", inp, out, pinned)
    b = spec("b", inp, out, pinned)
    candidates = compose((inp,), (a, b), out).routes
    assert len(candidates) == 2 and candidates[0].path_id != candidates[1].path_id
    with pytest.raises(GraphBlocked, match="No verified installed provider"):
        execute_graph(candidates[1], (a, b), {"a": Adapter("a", pinned, lambda x, w: {})},
                      {inp: InputFile(manifest, file_sha256(manifest))},
                      out, tmp_path / "run", checkout_sha=checkout())


def test_source_image_changed_during_run_cannot_publish_mesh(tmp_path):
    manifest = synthetic_input(tmp_path)
    inp, out = ports()
    pinned = tmp_path / "provider.py"
    pinned.write_text("provider")
    module = spec("mutating_provider", inp, out, pinned)

    def mutate_source(inputs, work):
        depth = manifest.parent / "depth" / "0000.png"
        depth.write_bytes(depth.read_bytes() + b"changed-after-preflight")
        mesh = work / "mesh.ply"
        mesh.write_bytes(b"ply\n")
        return {out: mesh}

    adapter = Adapter("mutating_provider", pinned, mutate_source)
    with pytest.raises(GraphFailed, match="Referenced source image changed"):
        execute_graph(plan(inp, out, module), (module,),
                      {"mutating_provider": adapter},
                      {inp: InputFile(manifest, file_sha256(manifest))},
                      out, tmp_path / "run", checkout_sha=checkout())
    assert not (tmp_path / "run").exists()
    failure = json.loads((tmp_path / "run.failed.json").read_text())
    assert failure["status"] == "failed"
    assert failure["input_dependencies_sha256"][inp.label()]


def artifact(kind):
    return ArtifactType(kind, "none", "none", "none", "estimated", "train",
                        "fixture", "none", "none", "none", "none", "none", "none")


def test_multi_output_provider_preserves_planned_first_producer(tmp_path):
    seed, x, y, target = map(artifact, ("seed", "x", "y", "mesh"))
    source = tmp_path / "provider.py"
    source.write_text("# synthetic provider")
    item = tmp_path / "input"
    item.write_text("seed")
    def module(name, ins, outs):
        return ModuleSpec(name, tuple(ins), tuple(outs), 1, True, file_sha256(source))
    modules = (module("a", [seed], [x]), module("b", [seed], [x, y]),
               module("join", [x, y], [target]))
    route = next(r for r in compose((seed,), modules, target).routes if "a" in r.modules)
    def first(inputs, work):
        path = work / "x"
        path.write_text("A")
        return {x: path}
    def second(inputs, work):
        px, py = work / "x", work / "y"
        px.write_text("B")
        py.write_text("Y")
        return {x: px, y: py}
    def join(inputs, work):
        path = work / "mesh"
        path.write_text(inputs[x].read_text() + "|" + inputs[y].read_text())
        return {target: path}
    adapters = {name: Adapter(name, source, fn)
                for name, fn in (("a", first), ("b", second), ("join", join))}
    receipt = execute_graph(route, modules, adapters,
        {seed: InputFile(item, file_sha256(item))}, target,
        tmp_path / "run", checkout_sha=checkout())
    output = receipt["outputs"][target.label()]
    assert (tmp_path / "run" / output["path"]).read_text() == "A|Y"
    assert receipt["outputs"][x.label()]["provider"] == "a"


def test_adapter_cannot_mutate_prior_intermediate_and_publish(tmp_path):
    seed, intermediate, target = map(artifact, ("seed", "intermediate", "mesh"))
    source = tmp_path / "provider.py"
    source.write_text("# synthetic provider")
    item = tmp_path / "input"
    item.write_text("seed")
    first = ModuleSpec("first", (seed,), (intermediate,), 1, True, file_sha256(source))
    last = ModuleSpec("last", (intermediate,), (target,), 1, True, file_sha256(source))
    def produce(inputs, work):
        path = work / "intermediate"
        path.write_text("frozen")
        return {intermediate: path}
    def mutate(inputs, work):
        inputs[intermediate].write_text("changed")
        path = work / "mesh"
        path.write_text("derived from changed input")
        return {target: path}
    modules = (first, last)
    route = compose((seed,), modules, target).routes[0]
    with pytest.raises(GraphFailed, match="Typed input artifact changed"):
        execute_graph(route, modules,
            {"first": Adapter("first", source, produce),
             "last": Adapter("last", source, mutate)},
            {seed: InputFile(item, file_sha256(item))}, target,
            tmp_path / "run", checkout_sha=checkout())
    assert not (tmp_path / "run").exists()
    failure = json.loads((tmp_path / "run.failed.json").read_text())
    assert failure["status"] == "failed" and failure["attempted_module"] == "last"

def test_projective_and_tsdf_are_distinct_executed_mesh_providers(tmp_path):
    """Actual synthetic meshes, not a claim of a real-scene A/B result."""
    from echo4d import projective_meshing
    from echo4d.graph_runner import projective_recording_adapter
    manifest = synthetic_input(tmp_path)
    inp, out = ports()
    tsdf_source = Path(__import__("echo4d.graph_runner", fromlist=["x"]).__file__)
    projective_source = Path(projective_meshing.__file__)
    a = spec("known_pose_tsdf", inp, out, tsdf_source)
    b = spec("observed_projective", inp, out, projective_source)
    routes = compose((inp,), (a, b), out).routes
    assert len(routes) == 2 and len({r.path_id for r in routes}) == 2
    adapters = {
        a.module_id: tsdf_recording_adapter(a.module_id, tsdf_source, inp, out,
            bounds=(-1.8, -.2, -1.9, 1.8, 2.2, 1.8)),
        b.module_id: projective_recording_adapter(b.module_id, projective_source, inp, out,
            pixel_stride=2, max_vertices=300_000, max_faces=500_000),
    }
    receipts = {}
    for route in routes:
        name = route.modules[0]
        receipt = execute_graph(route, (a, b), adapters,
            {inp: InputFile(manifest, file_sha256(manifest))}, out,
            tmp_path / name, checkout_sha=checkout())
        mesh = receipt["outputs"][out.label()]
        assert mesh["bytes"] > 200
        assert file_sha256(tmp_path / name / mesh["path"]) == mesh["sha256"]
        assert receipt["quality"] == "not_evaluated"
        receipts[name] = receipt
    assert receipts[a.module_id]["experiment_id"] != receipts[b.module_id]["experiment_id"]
    assert receipts[a.module_id]["outputs"][out.label()]["sha256"] != receipts[b.module_id]["outputs"][out.label()]["sha256"]
    provider_receipt = json.loads((tmp_path / b.module_id /
        "00_observed_projective" / "provider-report.json").read_text())
    assert provider_receipt["fusion"] is False
    assert provider_receipt["algorithm"] == "per-frame-projective-triangles-v1"


def test_projective_provider_rejects_wrong_source_and_relative_depth(tmp_path):
    from echo4d import projective_meshing
    from echo4d.graph_runner import projective_recording_adapter
    inp, out = ports()
    with pytest.raises(GraphBlocked, match="actual algorithm source"):
        projective_recording_adapter("p", tmp_path / "pretend.py", inp, out)
    from dataclasses import replace
    with pytest.raises(GraphBlocked, match="metric"):
        projective_recording_adapter("p", Path(projective_meshing.__file__),
            replace(inp, scale="relative", length_unit="none"), out)
    with pytest.raises(GraphBlocked, match="pixel_stride"):
        projective_recording_adapter("p", Path(projective_meshing.__file__),
            inp, out, pixel_stride=17)
