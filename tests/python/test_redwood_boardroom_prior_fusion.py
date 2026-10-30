"""Real Boardroom source and pose guards; visual success is assessed separately."""
import io
import zipfile
import numpy as np
import pytest
from redwood_boardroom_prior_fusion import official_assets, read_log, index_zip, choose, DataError

def test_boardroom_table_links_are_selected_by_column():
    html = """<table><tr><td>Bedroom</td><td>21</td><td><a href="wrong">scan</a></td>
    <td><a href="wrong">RGBD</a></td><td></td><td><a href="wrong">mesh</a></td><td><a href="wrong">pose</a></td></tr>
    <tr><td>Boardroom</td><td>60</td><td><a href="lidar.zip">LiDAR</a></td>
    <td><a href="boardroom-rgbd.zip">RGBD</a></td><td></td>
    <td><a href="boardroom.ply">mesh</a></td>
    <td><a href="boardroom.json">JSON</a><a href="boardroom.log">LOG</a></td></tr></table>"""
    links=official_assets(html)
    assert links["rgbd"].endswith("boardroom-rgbd.zip")
    assert links["prior"].endswith("boardroom.ply")
    assert links["poses"].endswith("boardroom.log")
    assert len(set(links.values()))==3

def test_missing_boardroom_is_not_replaced_with_another_scene():
    with pytest.raises(DataError):
        official_assets("<table><tr><td>Bedroom</td></tr></table>")

def test_pose_log_and_evenly_spread_source_selection():
    lines=[]
    for i in range(120):
        lines.append(f"{i} {i} 0")
        lines.extend(" ".join(map(str,row)) for row in np.eye(4))
    poses=read_log("\n".join(lines))
    assert len(poses)==120
    ids=choose(list(poses),40)
    assert len(ids)==40 and ids[0]==0 and ids[-1]==119

def test_only_complete_rgb_depth_zip_pairs_are_used():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w") as z:
        z.writestr("rgbd/00000-12345.jpg",b"color")
        z.writestr("rgbd/00000-12345.png",b"depth")
        z.writestr("rgbd/00001-12345.jpg",b"color")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as z:
        assert set(index_zip(z))=={0}
