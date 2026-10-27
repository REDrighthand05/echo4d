"""Single maintained COLMAP import CLI must use the strict #6/#24 adapter."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

from echo4d.cli import build_parser, main
from echo4d.model import load_recording


def _args(tmp_path: Path, scale: str = ".05") -> list[str]:
    return ["import-colmap", "--model", str(tmp_path / "model"),
            "--images", str(tmp_path / "images"),
            "--depth", str(tmp_path / "depth"),
            "--timestamps", str(tmp_path / "times.csv"),
            "--meters-per-colmap-unit", scale,
            "--scale-source", "3 independently measured depth anchor frames",
            "--output", str(tmp_path / "output")]


def test_cli_routes_to_existing_undistorted_camera_z_adapter(tmp_path):
    a=build_parser().parse_args(_args(tmp_path))
    assert a.command=="import-colmap"
    assert a.meters_per_colmap_unit == .05
    assert a.scale_source.startswith("3 independently measured")
    assert a.depth_index is None  # opt-in index, ordinary metric NPY remains the default
    assert a.depth_scale is None  # never implicitly convert already-metric camera Z


def test_refuse_unknown_or_zero_scale_even_before_file_import(tmp_path,capsys):
    code=main(_args(tmp_path,scale="0"))
    assert code==2
    err=json.loads(capsys.readouterr().err)
    assert "scale" in err["error"].lower()
    assert not (tmp_path/"output").exists()


def test_refuse_missing_colmap_files_with_clean_structured_error(tmp_path,capsys):
    code=main(_args(tmp_path))
    assert code==2
    err=json.loads(capsys.readouterr().err)
    assert err["type"] in {"DataError","OSError"}
    assert not (tmp_path/"output").exists()


def test_standard_cli_scales_camera_center_without_rescaling_metric_depth(tmp_path, capsys):
    for name in ('model', 'images', 'depth'):
        (tmp_path / name).mkdir()
    (tmp_path / 'model' / 'cameras.txt').write_text('1 PINHOLE 8 6 12 13 3.5 2.5\n')
    (tmp_path / 'model' / 'images.txt').write_text('1 1 0 0 0 -1 -2 -3 1 a.png\n\n')
    (tmp_path / 'times.csv').write_text('name,t\na.png,9\n')
    Image.fromarray(np.full((6, 8, 3), 128, np.uint8)).save(tmp_path / 'images' / 'a.png')
    np.save(tmp_path / 'depth' / 'a.npy', np.full((6, 8), 1.5, np.float32))
    assert main(_args(tmp_path, scale='2')) == 0
    assert json.loads(capsys.readouterr().out)['frames'] == 1
    frames = list(load_recording(tmp_path / 'output' / 'recording.json').iter_frames())
    assert np.allclose(frames[0][1].camera_to_world[:3, 3], [2, 4, 6])
    assert np.allclose(frames[0][1].depth, 1.5)
