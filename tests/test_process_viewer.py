import numpy as np
import panel as pn
import pytest

import gathervis as gv
from gathervis.demo import synthetic_line
from gathervis.process import decimate, quantize, robust_clim
from gathervis.viewer import ShotBrowser, SliceView, acquisition_app, view_gather


# ---------------- process ----------------
def test_robust_clim_symmetric():
    a = np.random.default_rng(0).standard_normal((100, 100)).astype("f4")
    lo, hi = robust_clim(a, 98)
    assert lo == -hi and hi > 0


def test_robust_clim_degenerate():
    assert robust_clim(np.zeros((10, 10))) == (-1.0, 1.0)


def test_decimate():
    a = np.zeros((5000, 300))
    d = decimate(a, (1600, 1600))
    assert d.shape[0] <= 1600 and d.shape[1] == 300
    assert decimate(a, (10000, 10000)).shape == a.shape  # no-op


def test_quantize_range():
    a = np.array([[-2.0, 0.0, 2.0]])
    q = quantize(a, (-1.0, 1.0))
    assert q.dtype == np.uint8
    assert q[0, 0] == 0 and q[0, 1] == 127 and q[0, 2] == 255


# ---------------- viewer construction ----------------
@pytest.fixture(scope="module")
def line():
    return synthetic_line(ns=6, nr=24, nt=101)


def test_view_gather(line):
    app = view_gather(line.shot(0), dt=line.dt)
    assert isinstance(app, pn.viewable.Viewable)


def test_shot_browser_2d_per_shot(line):
    b = ShotBrowser(line)
    b.set_shot(3)
    assert b.ishot == 3
    assert isinstance(b.panel(), pn.viewable.Viewable)


def test_shot_browser_4d():
    d = np.random.default_rng(1).standard_normal((3, 6, 8, 40)).astype("f4")
    b = ShotBrowser(gv.from_array(d))
    b.set_shot(2)  # swaps the per-shot SliceView volume
    assert isinstance(b.panel(), pn.viewable.Viewable)


def test_slice_view_sliders():
    d = np.random.default_rng(2).standard_normal((10, 12, 40)).astype("f4")
    sv = SliceView(d, names=("recy", "recx"))
    sv.w0.value = 5
    sv.w2.value = 10
    assert isinstance(sv.panel(), pn.viewable.Viewable)


def test_acquisition_app(line):
    assert isinstance(acquisition_app(line), pn.viewable.Viewable)


def test_show_dispatch(line):
    assert gv.show(line.shot(0), dt=line.dt) is not None            # 2-D
    assert gv.show(line.data) is not None                           # 3-D browse
    assert gv.show(line.data, view="slices") is not None            # 3-D slices
    assert gv.show(line) is not None                                # geometry app
    vol = np.zeros((4, 5, 30), "f4")
    assert gv.show(vol, axes=("recy", "recx", "time")) is not None  # slice default
    with pytest.raises(ValueError):
        gv.show(line.data, view="nope")


def test_show_from_path(tmp_path, line):
    p = tmp_path / "d.bin"
    np.asarray(line.data).tofile(p)
    app = gv.show(str(p), shape=line.shape, dt=line.dt)
    assert app is not None
    with pytest.raises(ValueError):
        gv.show(str(p))  # shape required
