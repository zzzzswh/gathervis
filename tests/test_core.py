import numpy as np
import pytest

import gathervis as gv
from gathervis.core import Geometry


def test_default_axes():
    assert gv.from_array(np.zeros((10, 50), "f4")).axes == ("trace", "time")
    assert gv.from_array(np.zeros((3, 10, 50), "f4")).axes == ("shot", "rec", "time")
    assert gv.from_array(np.zeros((3, 4, 5, 50), "f4")).axes == \
        ("shot", "recy", "recx", "time")


def test_axes_validation():
    d = np.zeros((3, 10, 50), "f4")
    with pytest.raises(ValueError):
        gv.from_array(d, axes=("shot", "time", "rec"))     # time must be last
    with pytest.raises(ValueError):
        gv.from_array(d, axes=("rec", "shot", "time"))     # shot must be first
    with pytest.raises(ValueError):
        gv.from_array(d, axes=("foo", "rec", "time"))      # unknown name
    g = gv.from_array(d, axes=("recy", "recx", "time"))    # a 3-D shot gather
    assert "shot" not in g.axes


def test_shot_and_sample():
    d = np.arange(3 * 4 * 5, dtype="f4").reshape(3, 4, 5)
    g = gv.from_array(d, dt=0.004)
    assert g.nshot == 3 and g.nt == 5
    assert np.allclose(g.shot(1), d[1])
    assert g.times[-1] == pytest.approx(0.004 * 4)
    assert g.sample(10).ndim == 1


def test_geometry_shapes():
    src = np.zeros((3, 2))
    rec = np.zeros((7, 3))
    geo = Geometry(src, rec)
    assert geo.ns == 3 and geo.nr == 7 and geo.shared
    assert geo.src.shape == (3, 3)          # z padded
    per_shot = Geometry(src, np.zeros((3, 7, 2)))
    assert not per_shot.shared and per_shot.rec_for(2).shape == (7, 3)
    with pytest.raises(ValueError):
        Geometry(src, np.zeros((4, 7, 2)))  # ns mismatch


def test_geometry_data_consistency():
    d = np.zeros((3, 7, 50), "f4")
    g = gv.from_array(d, src=np.zeros((3, 2)), rec=np.zeros((7, 2)))
    assert g.geometry.nr == 7
    with pytest.raises(ValueError):
        gv.from_array(d, src=np.zeros((3, 2)), rec=np.zeros((6, 2)))
    d4 = np.zeros((3, 2, 5, 50), "f4")     # 10 traces per shot, flattened
    g4 = gv.from_array(d4, src=np.zeros((3, 2)), rec=np.zeros((10, 2)))
    assert g4.geometry.nr == 10


def test_from_file_memmap(tmp_path):
    d = np.random.default_rng(0).standard_normal((4, 8, 16)).astype("f4")
    p = tmp_path / "line.bin"
    d.tofile(p)
    g = gv.from_file(p, shape=(4, 8, 16), dt=0.002)
    assert isinstance(g.data, np.memmap)
    assert np.allclose(g.shot(2), d[2])
