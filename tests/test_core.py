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


class _Lazy:
    """The whole contract for a lazy dataset: shape, ndim, dtype, and
    indexing along axis 0 -- e.g. several memmaps concatenated on demand."""

    def __init__(self, parts):
        self.parts = parts
        self.edges = np.cumsum([0] + [p.shape[0] for p in parts])
        self.shape = (int(self.edges[-1]),) + parts[0].shape[1:]
        self.ndim, self.dtype = len(self.shape), parts[0].dtype

    def __getitem__(self, key):
        first, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        if isinstance(first, (int, np.integer)):
            k = int(np.searchsorted(self.edges, first, side="right")) - 1
            return self.parts[k][(int(first) - int(self.edges[k]),) + rest]
        return np.stack([self[(int(i),) + rest]
                         for i in np.arange(self.shape[0])[first]])


def test_sample_works_on_an_index_only_lazy_array():
    from gathervis.demo import synthetic_line
    g = synthetic_line(ns=6, nr=40, nt=200)
    lazy = _Lazy([np.asarray(g.data[:4]), np.asarray(g.data[4:])])
    s = gv.from_array(lazy, src=g.geometry.src, rec=g.geometry.rec,
                      dt=g.dt).sample(max_samples=4000, chunks=4)
    assert s.size > 0 and np.isfinite(s).all() and np.abs(s).max() > 0


def test_every_tab_works_on_an_index_only_lazy_array():
    from gathervis.demo import synthetic_line
    from gathervis.viewer import Workspace
    g = synthetic_line(ns=6, nr=40, nt=200)
    lazy = _Lazy([np.asarray(g.data[:4]), np.asarray(g.data[4:])])
    ws = Workspace(gv.from_array(lazy, src=g.geometry.src,
                                 rec=g.geometry.rec, dt=g.dt))
    ws.browser.set_shot(5)                        # a shot in the second part
    for tab in range(len(ws.tabs)):
        ws.tabs.active = tab
        ws._w_ftype.value = "band-pass"           # volume tab re-processes
        ws._w_gain.value = "AGC"
        ws._w_ftype.value, ws._w_gain.value = "off", "off"
    assert ws.slices.vol.shape == (6, 40, 200)


def test_sample_never_flattens_a_non_contiguous_array():
    """reshape(-1) copies a non-contiguous array: 8 GB here, so the test
    would fail by running out of memory rather than by an assert."""
    trace = np.linspace(-1, 1, 1000, dtype=np.float32)
    big = np.broadcast_to(trace, (2000, 1000, 1000))          # 8 GB, 4 KB real
    s = gv.from_array(big, dt=0.002).sample(max_samples=5000, chunks=8)
    assert 0 < s.size <= 5000 + 8 * 1000 and np.isfinite(s).all()
    view = np.zeros((40, 30, 20), "f4").transpose(1, 0, 2)    # strided view
    assert gv.from_array(np.ascontiguousarray(view)).sample().size == 24000
    assert gv.from_array(view).sample().size > 0
