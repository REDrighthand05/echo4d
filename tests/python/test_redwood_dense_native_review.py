import numpy as np
import pytest
from PIL import Image
from redwood_dense_native_review import barycentric_rgb,old_panel


def test_vertex_color_interpolation():
    colors=np.eye(3)
    faces=np.array([[0,1,2]])
    rgb=barycentric_rgb(np.array([0,0,0]),np.array([[0,0],[1,0],[0,1]]),colors,faces)
    assert rgb.tolist()==[[255,0,0],[0,255,0],[0,0,255]]
    assert barycentric_rgb(np.array([0]),np.array([[.25,.25]]),colors,faces).tolist()==[[128,64,64]]


@pytest.mark.parametrize("uv",[[[-.1,.5]],[[.7,.7]]])
def test_invalid_barycentric_is_rejected(uv):
    with pytest.raises(ValueError):
        barycentric_rgb(np.array([0]),np.array(uv),np.eye(3),np.array([[0,1,2]]))


def test_crop_is_old_mesh_not_original_photo(tmp_path):
    im=Image.new("RGB",(640,268))
    im.paste((255,0,0),(0,28,320,268))
    im.paste((0,255,0),(320,28,640,268))
    p=tmp_path/"baseline.png"
    im.save(p)
    assert old_panel(p,False).getpixel((100,100))==(0,255,0)
    with pytest.raises(ValueError):
        old_panel(p,True)
