import argparse
import json

import numpy as np
import pytest
from PIL import Image

from echo4d.colmap import import_colmap
from echo4d.model import DataError, load_recording


def _write_colmap_text(root, *, image_rows: list[str], camera_row: str) -> None:
    text = root / "colmap"
    text.mkdir()
    (text / "cameras.txt").write_text(camera_row + "\n", encoding="utf-8")
    # Include empty feature lines to match COLMAP's two-line-per-image format.
    lines = []
    for row in image_rows:
        lines.extend([row, ""])
    (text / "images.txt").write_text("\n".join(lines), encoding="utf-8")


def test_colmap_import_converts_world_to_camera_pose(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    for i in range(2):
        Image.fromarray(np.full((3, 4, 3), 30 + i, dtype=np.uint8)).save(root / f"img{i}.png")
        np.save(root / f"depth{i}.npy", np.full((3, 4), 4 + i, dtype=np.uint16), allow_pickle=False)

    # R_wc = Ry(90deg), C_world=[1,2,3] -> t_wc = -R_wc @ C_world = [-3,-2,1]
    q = np.sqrt(0.5)
    row0 = f"1 {q} 0 {q} 0 -3 -2 1 1 img0.png"
    row1 = "2 1 0 0 0 -2 -2 -3 1 img1.png"
    _write_colmap_text(root, image_rows=[row0, row1], camera_row="1 PINHOLE 4 3 50 50 1.5 1")

    (root / "timestamps.txt").write_text("img0.png 10.00\nimg1.png 10.05\n", encoding="utf-8")
    (root / "depth-index.txt").write_text("img0.png depth0.npy\nimg1.png depth1.npy\n", encoding="utf-8")

    args = argparse.Namespace(
        dataset=root,
        output=tmp_path / "out",
        colmap_text=root / "colmap",
        depth_index="depth-index.txt",
        timestamps="timestamps.txt",
        depth_scale=2.0,
        meters_per_colmap_unit=2.0,
        scale_evidence="3 anchor frames from sensor depth",
    )
    result = import_colmap(args)
    assert result["frames"] == 2

    recording = load_recording(args.output / "recording.json")
    frames = list(recording.iter_frames())
    first = frames[0][1]
    assert np.allclose(first.camera_to_world[:3, 3], [2, 4, 6], atol=1e-6)
    assert first.depth[0, 0] == pytest.approx(2.0)

    provenance = json.loads((args.output / "import-provenance.json").read_text(encoding="utf-8"))
    assert provenance["scale_evidence"] == "3 anchor frames from sensor depth"
    assert provenance["meters_per_colmap_unit"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "camera_row,match",
    [
        ("1 SIMPLE_RADIAL 4 3 50 1.5 1 0.01", "undistort"),
        ("1 PINHOLE 4 3 50 50 1.5", "expects 4 parameters"),
    ],
)
def test_colmap_import_rejects_invalid_camera_contract(tmp_path, camera_row, match):
    root = tmp_path / "dataset"
    root.mkdir()
    Image.fromarray(np.full((3, 4, 3), 30, dtype=np.uint8)).save(root / "img0.png")
    np.save(root / "depth0.npy", np.full((3, 4), 4, dtype=np.uint16), allow_pickle=False)
    _write_colmap_text(root, image_rows=["1 1 0 0 0 0 0 0 1 img0.png"], camera_row=camera_row)
    (root / "timestamps.txt").write_text("img0.png 10\n", encoding="utf-8")
    (root / "depth-index.txt").write_text("img0.png depth0.npy\n", encoding="utf-8")

    args = argparse.Namespace(
        dataset=root,
        output=tmp_path / "out",
        colmap_text=root / "colmap",
        depth_index="depth-index.txt",
        timestamps="timestamps.txt",
        depth_scale=1.0,
        meters_per_colmap_unit=1.0,
        scale_evidence="sensor",
    )
    with pytest.raises(DataError, match=match):
        import_colmap(args)


def test_colmap_import_requires_scale_evidence_and_safe_depth_paths(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    Image.fromarray(np.full((3, 4, 3), 30, dtype=np.uint8)).save(root / "img0.png")
    _write_colmap_text(root, image_rows=["1 1 0 0 0 0 0 0 1 img0.png"], camera_row="1 PINHOLE 4 3 50 50 1.5 1")
    (root / "timestamps.txt").write_text("img0.png 10\n", encoding="utf-8")
    (root / "depth-index.txt").write_text("img0.png ../outside.npy\n", encoding="utf-8")

    args = argparse.Namespace(
        dataset=root,
        output=tmp_path / "out",
        colmap_text=root / "colmap",
        depth_index="depth-index.txt",
        timestamps="timestamps.txt",
        depth_scale=1.0,
        meters_per_colmap_unit=1.0,
        scale_evidence="",
    )
    with pytest.raises(DataError, match="scale-evidence"):
        import_colmap(args)

    args.scale_evidence = "sensor"
    with pytest.raises(DataError, match="outside dataset"):
        import_colmap(args)


def test_colmap_import_requires_timestamps_for_all_images(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    Image.fromarray(np.full((3, 4, 3), 30, dtype=np.uint8)).save(root / "img0.png")
    np.save(root / "depth0.npy", np.full((3, 4), 4, dtype=np.uint16), allow_pickle=False)
    _write_colmap_text(root, image_rows=["1 1 0 0 0 0 0 0 1 img0.png"], camera_row="1 PINHOLE 4 3 50 50 1.5 1")
    (root / "timestamps.txt").write_text("other.png 10\n", encoding="utf-8")
    (root / "depth-index.txt").write_text("img0.png depth0.npy\n", encoding="utf-8")

    args = argparse.Namespace(
        dataset=root,
        output=tmp_path / "out",
        colmap_text=root / "colmap",
        depth_index="depth-index.txt",
        timestamps="timestamps.txt",
        depth_scale=1.0,
        meters_per_colmap_unit=1.0,
        scale_evidence="sensor",
    )
    with pytest.raises(DataError, match="Missing timestamp"):
        import_colmap(args)


def test_colmap_import_rejects_nonpositive_metric_scale(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    Image.fromarray(np.full((3, 4, 3), 30, dtype=np.uint8)).save(root / "img0.png")
    np.save(root / "depth0.npy", np.full((3, 4), 4, dtype=np.uint16), allow_pickle=False)
    _write_colmap_text(root, image_rows=["1 1 0 0 0 0 0 0 1 img0.png"], camera_row="1 PINHOLE 4 3 50 50 1.5 1")
    (root / "timestamps.txt").write_text("img0.png 10\n", encoding="utf-8")
    (root / "depth-index.txt").write_text("img0.png depth0.npy\n", encoding="utf-8")
    args = argparse.Namespace(
        dataset=root,
        output=tmp_path / "out",
        colmap_text=root / "colmap",
        depth_index="depth-index.txt",
        timestamps="timestamps.txt",
        depth_scale=1.0,
        meters_per_colmap_unit=0.0,
        scale_evidence="sensor",
    )
    with pytest.raises(DataError, match="meters-per-colmap-unit"):
        import_colmap(args)
