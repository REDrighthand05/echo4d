"""Prune disconnected TSDF islands from real TUM Freiburg1 Desk RGB-D GLB.

Source: measured native-640 RGB-D fusion, Actions run 41068611251.
No new measured geometry, no prior scan, no inferred surface.
Five rendered viewpoints are VIRTUAL, not original camera positions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh

MIN_FACES = 100
VIEW_OFFSETS = (
    (4.5, 1.5, 0.0), (3.0, 3.0, 3.0), (-3.0, 3.0, -3.0),
    (0.0, 1.5, 4.5), (-4.5, 1.5, 0.0),
)


def remove_small_islands(mesh: trimesh.Trimesh, min_faces: int = MIN_FACES):
    """Remove small disconnected components while preserving vertex RGB."""
    if min_faces < 1:
        raise ValueError("min_faces must be >= 1")
    if mesh.visual.kind != "vertex":
        raise ValueError("Input must contain measured vertex RGB")
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
    keep = np.zeros(len(mesh.faces), dtype=bool)
    removed_components = 0
    for component in components:
        if len(component) >= min_faces:
            keep[component] = True
        else:
            removed_components += 1
    if not keep.any():
        raise ValueError("No component survived; refusing empty export")
    cleaned = mesh.copy()
    cleaned.update_faces(keep)
    cleaned.remove_unreferenced_vertices()
    stats = {
        "input_triangles": len(mesh.faces),
        "output_triangles": len(cleaned.faces),
        "removed_triangles": int((~keep).sum()),
        "input_components": len(components),
        "removed_components": removed_components,
        "kept_components": len(components) - removed_components,
        "min_component_triangles": min_faces,
    }
    return cleaned, stats


def render_virtual(mesh, position, focal, width=440, height=330):
    """Same offscreen colored mesh camera before/after; no photo compositing."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from PIL import Image

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(
        np.ascontiguousarray(mesh.vertices, np.float32), deep=True))
    faces = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces))
    cells = vtk.vtkCellArray()
    cells.SetCells(len(mesh.faces), numpy_to_vtkIdTypeArray(
        np.ascontiguousarray(faces.reshape(-1), np.int64), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    rgb = np.ascontiguousarray(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
    colors = numpy_to_vtk(rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    colors.SetNumberOfComponents(3)
    colors.SetName("measured_rgb")
    poly.GetPointData().SetScalars(colors)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarModeToUsePointData()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToFlat()
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(.07, .07, .07)
    renderer.AddActor(actor)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(*position)
    camera.SetFocalPoint(*focal)
    camera.SetViewUp(0., 1., 0.)
    camera.SetViewAngle(58.)
    camera.SetClippingRange(.01, 15.)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(width, height)
    window.SetMultiSamples(0)
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    capture.Update()
    pixels = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
    return Image.fromarray(pixels.reshape(height, width, 3)[::-1].copy())


def make_comparison(original, cleaned, output: Path):
    from PIL import Image, ImageDraw
    focal = original.bounds.mean(axis=0)
    sheet = Image.new("RGB", (880, 5 * 355), (12, 12, 12))
    pen = ImageDraw.Draw(sheet)
    for i, offset in enumerate(VIEW_OFFSETS):
        position = focal + np.asarray(offset)
        old = render_virtual(original, position, focal)
        new = render_virtual(cleaned, position, focal)
        sheet.paste(old, (0, i * 355 + 25))
        sheet.paste(new, (440, i * 355 + 25))
        pen.text((8, i * 355 + 4), f"VIRTUAL {i+1} / BEFORE", fill="white")
        pen.text((448, i * 355 + 4), f"VIRTUAL {i+1} / ISLANDS REMOVED", fill="white")
        old.save(output / f"virtual-{i+1}-before.png")
        new.save(output / f"virtual-{i+1}-after.png")
    sheet.save(output / "five-virtual-view-island-prune-ab.png")


def run(input_path: Path, output: Path, min_faces=MIN_FACES, photos=None):
    output.mkdir(parents=True, exist_ok=True)
    original = trimesh.load(input_path, force="mesh")
    cleaned, stats = remove_small_islands(original, min_faces)
    cleaned.export(output / "TUM-DESK-MEASURED-640-islands-pruned.glb")
    make_comparison(original, cleaned, output)
    if photos is not None:
        for index in (11, 31, 53):
            photo = Path(photos) / f"real-desk-{index:02d}-640x480.png"
            if photo.is_file():
                shutil.copy2(photo, output / photo.name)
    receipt = {
        "scene": "TUM Freiburg1 Desk, real RGB-D, known poses",
        "source_run": "https://github.com/REDrighthand05/echo4d/actions/runs/41068611251",
        "source_geometry": "Measured RGB-D TSDF only, no pre-existing scan mesh",
        "operation": "Remove disconnected triangle islands; NO invented surfaces",
        "view_provenance": "Five fixed VIRTUAL views; original camera poses NOT recovered",
        "geometry_stats": stats,
        "human_visual_verdict": "PENDING independent visual review",
        "free_orbit": "FAIL until visually accepted",
        "issue_74": "FAIL unchanged",
        "historical_desk": "FAIL 16.7779 cm > 15 cm unchanged",
    }
    (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-faces", type=int, default=MIN_FACES)
    parser.add_argument("--photos", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.min_faces, args.photos), indent=2))


if __name__ == "__main__":
    main()
