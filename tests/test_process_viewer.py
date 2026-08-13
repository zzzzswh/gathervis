import numpy as np
import panel as pn
import pytest

import gathervis as gv
from gathervis.demo import synthetic_line
from gathervis.process import decimate, quantize, robust_clim
from gathervis.viewer import LayoutMap, ShotBrowser, SliceView, Workspace, view_gather


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


def _state(g):
    from gathervis.process import robust_clim
    s = {"sample": g.sample()}
    s["clim"] = robust_clim(s["sample"])
    return s


def test_shot_browser_2d_per_shot(line):
    b = ShotBrowser(line, _state(line))
    b.set_shot(3)
    assert b.ishot == 3
    assert isinstance(b.panel(), pn.viewable.Viewable)


def test_shot_browser_4d():
    d = np.random.default_rng(1).standard_normal((3, 6, 8, 40)).astype("f4")
    g = gv.from_array(d)
    b = ShotBrowser(g, _state(g))
    b.set_shot(2)  # swaps the per-shot SliceView volume
    assert isinstance(b.panel(), pn.viewable.Viewable)


def test_slice_view_sliders():
    d = np.random.default_rng(2).standard_normal((10, 12, 40)).astype("f4")
    sv = SliceView(d, names=("recy", "recx"))
    sv.w0.value = 5
    sv.w2.value = 10
    assert isinstance(sv.panel(), pn.viewable.Viewable)


def test_workspace_tabs_and_info(line):
    ws = Workspace(line)                       # 3-D line with geometry
    labels = [t[0] for t in ws.tabs._names] if hasattr(ws.tabs, "_names") else \
             list(ws.tabs._names)
    assert len(ws.tabs) == 2                   # shot gathers + volume slices
    assert ws.map is not None
    from gathervis.viewer import _info_md, _survey_kind
    md = _info_md(line)
    assert "2-D seismic line" in md and "sources" in md
    assert isinstance(ws.panel(), pn.viewable.Viewable)
    ws2 = Workspace(line, view="slices")
    assert ws2.tabs.active == 1                # slices tab initially active
    vol = gv.from_array(np.zeros((4, 5, 30), "f4"),
                        axes=("recy", "recx", "time"))
    assert _survey_kind(vol) == "volume"
    assert len(Workspace(vol).tabs) == 1       # slices only, no shot tab


def test_layout_map(line):
    m = LayoutMap(line.geometry)
    picked = []
    m.on_pick = picked.append
    m._on_tap("indices", [], [3])
    m.set_active(3)
    assert picked == [3]


def test_show_dispatch(line):
    assert gv.show(line.shot(0), dt=line.dt) is not None            # 2-D
    assert gv.show(line.data) is not None                           # 3-D browse
    assert gv.show(line.data, view="slices") is not None            # 3-D slices
    assert gv.show(line, name="2-D acoustic, 2-D velocity model") is not None
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


# ---------------- serving helpers ----------------
def test_resolve_port_fallback():
    import socket
    from gathervis.viewer import _resolve_port
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    busy = blocker.getsockname()[1]
    try:
        p = _resolve_port(busy, "127.0.0.1")   # busy -> auto fallback
        assert p != busy and p > 0
        assert _resolve_port(0, "127.0.0.1") > 0  # 0 -> always auto
    finally:
        blocker.close()


# ---------------- CLI ----------------
def test_cli_load(tmp_path, line):
    from gathervis.__main__ import _load, _parser
    npy = tmp_path / "d.npy"
    np.save(npy, np.asarray(line.data))
    geom = tmp_path / "g.npz"
    np.savez(geom, src=line.geometry.src, rec=line.geometry.rec, dt=line.dt)
    g = _load(_parser().parse_args([str(npy), "--geom", str(geom)]))
    assert g.geometry is not None and g.dt == line.dt   # dt picked up from npz
    raw = tmp_path / "d.bin"
    np.asarray(line.data).tofile(raw)
    g2 = _load(_parser().parse_args(
        [str(raw), "--shape", *map(str, line.shape), "--dt", "0.004"]))
    assert g2.dt == 0.004 and isinstance(g2.data, np.memmap)


def test_banner_modes(monkeypatch, capsys):
    from gathervis.viewer import _banner
    for k in ("SSH_CONNECTION", "SSH_TTY", "TERM_PROGRAM"):
        monkeypatch.delenv(k, raising=False)
    for k in list(__import__("os").environ):
        if k.startswith("VSCODE_"):
            monkeypatch.delenv(k, raising=False)
    _banner(1234)                                   # local: URL only
    out = capsys.readouterr().out
    assert "http://localhost:1234" in out and "ssh -L" not in out
    monkeypatch.setenv("SSH_TTY", "/dev/pts/0")     # bare SSH: full steps
    _banner(1234)
    out = capsys.readouterr().out
    assert "ssh -L 1234:127.0.0.1:1234 <user>@<this-server>" in out
    assert "LocalForward" in out
    monkeypatch.setenv("TERM_PROGRAM", "vscode")    # VS Code: auto-forward note
    _banner(1234)
    out = capsys.readouterr().out
    assert "auto-forwarded" in out and "ssh -L" not in out
