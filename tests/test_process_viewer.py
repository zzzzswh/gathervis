import numpy as np
import panel as pn
import pytest

import gathervis as gv
from gathervis.demo import synthetic_line
from gathervis.process import decimate, quantize, robust_clim
from bokeh.models import TextInput

from gathervis.viewer import (LayoutMap, ShotBrowser, VolumeView3D, Workspace,
                              _set_throttled, view_gather)


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
    b.set_shot(2)  # swaps the per-shot cuboid's volume
    assert isinstance(b.panel(), pn.viewable.Viewable)


def test_volume3d_geometry_and_sliders():
    d = np.random.default_rng(2).standard_normal((10, 12, 40)).astype("f4")
    sv = VolumeView3D(d, names=("recy", "recx"), dt=0.002)
    assert isinstance(sv.fig, dict)     # plain dict: Panel must never install
    #                                     restyle hooks (plotly.js gl3d #2866)
    assert len(sv.fig["data"]) == 4                    # 3 slice planes + box
    s0, s1, s2 = sv.fig["data"][:3]
    assert all(t["type"] == "surface" for t in (s0, s1, s2))
    assert sv.fig["data"][3]["type"] == "scatter3d"
    # each plane is constant along its own axis, spans the other two
    assert float(s0["x"].max()) == 9 and s0["x"].min() == s0["x"].max()
    assert s1["y"].min() == s1["y"].max() == 11
    assert s2["z"].min() == s2["z"].max() == 0.0       # time face at t0
    assert s0["surfacecolor"].dtype == np.uint8        # quantized wire format
    assert s0["surfacecolor"].shape == s0["z"].shape
    # time coordinates are seconds
    assert np.isclose(float(s0["z"].max()), 0.002 * 39)
    sv.w0.value = 5                       # live update while dragging
    assert sv.fig["data"][0]["x"].max() == 5
    sv.w2.value = 10
    assert np.isclose(float(sv.fig["data"][2]["z"].min()), 0.002 * 10)
    sv.set_cmap("gray")                                # recolors all 3 planes
    assert sv.fig["data"][0]["colorscale"] == sv.fig["data"][2]["colorscale"]
    big = np.zeros((8, 6, 30), "f4")
    sv.set_volume(big)                                 # swap + clamp sliders
    assert sv.w0.end == 7 and sv.w0.value <= 7
    assert isinstance(sv.panel(), pn.viewable.Viewable)


def test_volume3d_decimation_budget():
    from gathervis.viewer import MAX_PX_3D
    d = np.zeros((3, 5, 5000), "f4")                   # nt far over budget
    sv = VolumeView3D(d)
    s0 = sv.fig["data"][0]
    assert s0["z"].shape[1] <= MAX_PX_3D[1]            # time axis decimated
    assert s0["z"].shape == s0["surfacecolor"].shape == s0["y"].shape


def test_workspace_tabs_and_info(line):
    ws = Workspace(line)                # 3-D line with geometry -> 3 tabs
    assert list(ws.tabs._names) == ["Shot gathers", "Geometry", "Volume slices"]
    assert ws.map is not None
    from gathervis.viewer import _info_md, _survey_kind
    md = _info_md(line)
    assert "2-D seismic line" in md and "sources" in md
    assert isinstance(ws.panel(), pn.viewable.Viewable)
    ws2 = Workspace(line, view="slices")
    assert ws2.tabs.active == 2                # slices tab initially active
    vol = gv.from_array(np.zeros((4, 5, 30), "f4"),
                        axes=("recy", "recx", "time"))
    assert _survey_kind(vol) == "volume"
    assert len(Workspace(vol).tabs) == 1       # slices only, no shot/map tab
    bare = gv.from_array(np.zeros((4, 5, 30), "f4"))   # no geometry
    assert len(Workspace(bare).tabs) == 2      # shot + volume, no map tab


def test_layout_tab_pick_jumps_to_shots(line):
    ws = Workspace(line, view="slices")
    assert ws.tabs.active == 2
    ws.map._on_tap("indices", [], [4])  # tap a source on the layout tab
    assert ws.browser.ishot == 4        # shot selected...
    assert ws.tabs.active == 0          # ...and view jumped to shot gathers
    assert ws.map._cds_asrc.data["x"]   # active shot highlighted on the map


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


# ---------------- v0.4: display modes, colormaps, shot jump ----------------
def test_palettes():
    from gathervis.viewer import _palette, _CMAPS
    assert {"seismic", "gray", "petrel", "rainbow"} <= set(_CMAPS)
    for name in _CMAPS:                      # every advertised cmap renders
        pal = _palette(name)
        assert len(pal) == 256 and pal[0].startswith("#")
    assert _palette("gray")[0] == "#000000" and _palette("gray_r")[0] == "#ffffff"
    with pytest.raises(ValueError):
        _palette("nope")


def test_wiggle_mode(line):
    from gathervis.viewer import ImagePane, MAX_WIGGLE
    from gathervis.process import robust_clim
    a = np.asarray(line.shot(0))             # (96, 751) traces x time
    clim = robust_clim(a)
    p = ImagePane()
    p.update(a, clim, y0=0.0, dy=line.dt)
    assert p.cds.data["image"]                # density populated
    p.set_display("wiggle")
    ntr = len(p._cds_wig.data["xs"])
    assert 0 < ntr <= MAX_WIGGLE[0]           # trace budget respected
    assert len(p._cds_wig.data["xs"][0]) <= MAX_WIGGLE[1]
    assert len(p._cds_fill.data["xs"]) == ntr # one fill polygon per trace
    assert not p.cds.data["image"]            # hidden mode's payload dropped
    assert p._r_wig.visible and not p._r_img.visible
    p.set_display("density")
    assert p.cds.data["image"] and not p._cds_wig.data["xs"]
    with pytest.raises(ValueError):
        p.set_display("hologram")


def test_volume3d_display_noop():
    d = np.zeros((4, 5, 30), "f4")
    sv = VolumeView3D(d)
    sv.set_display("wiggle")                  # must not raise or alter traces
    assert sv.fig["data"][0]["type"] == "surface"


def test_shot_jump_input(line):
    b = ShotBrowser(line, _state(line))
    b.w_jump.value = 4                        # typing a shot number jumps
    assert b.ishot == 4
    b.set_shot(2)                             # programmatic jump mirrors input
    assert b.w_jump.value == 2
    b.set_shot(999)                           # clamped to valid range
    assert b.ishot == line.nshot - 1


def test_geometry_tab_label(line):
    ws = Workspace(line)
    assert "Geometry" in list(ws.tabs._names)


def test_live_slider_semantics(line):
    b = ShotBrowser(line, _state(line))
    seen = []
    b.on_shot_change = seen.append
    b.w_shot.value = 3                        # dragging updates live
    assert seen == [3] and b.w_jump.value == 3


# ---------------- v0.5: depth-axis property volumes ----------------
def test_asymmetric_clim():
    v = np.random.default_rng(0).uniform(1500, 3200, (50, 50)).astype("f4")
    lo, hi = robust_clim(v, 98, symmetric=False)
    assert 1500 <= lo < hi <= 3200          # percentile bounds, not +-|v|
    s_lo, s_hi = robust_clim(v, 98)         # default stays symmetric
    assert s_lo == -s_hi
    assert robust_clim(np.zeros((5, 5)), symmetric=False) == (-1.0, 1.0)


def test_depth_volume_workspace():
    vel = np.random.default_rng(1).uniform(1600, 3300, (20, 16, 12)).astype("f4")
    g = gv.from_array(vel, axes=("x", "y", "depth"), dt=10.0)
    from gathervis.viewer import _info_md, _survey_kind
    assert _survey_kind(g) == "property volume (depth)"
    assert "dz" in _info_md(g) and "110 m" in _info_md(g)   # 10*(12-1)
    ws = Workspace(g)
    assert ws.state["symmetric"] is False
    assert ws.state["clim"][0] > 0
    sv = ws.slices
    assert sv.fig["layout"]["scene"]["zaxis"]["title"] == "depth (m)"
    assert sv.w2.name == "depth sample"
    with pytest.raises(ValueError):
        gv.from_array(vel, axes=("x", "y", "rec"))          # bad last axis


# ---------------- v0.6: camera persistence + axis scaling ----------------
def test_camera_persists_across_updates():
    d = np.random.default_rng(3).standard_normal((8, 10, 40)).astype("f4")
    sv = VolumeView3D(d)
    cam = {"eye": {"x": 0.3, "y": -2.0, "z": 1.4}}
    # simulate a user rotation: plotly relayout reports scene.camera back
    sv.pane.relayout_data = {"scene.camera": cam}
    assert sv.fig["layout"]["scene"]["camera"] == cam
    sv.w0.value = 2                            # slice move keeps the camera
    assert sv.fig["layout"]["scene"]["camera"] == cam
    sv.set_volume(np.zeros((8, 10, 40), "f4"))  # same shape: keep camera
    assert sv.fig["layout"]["scene"]["camera"] == cam
    sv.set_volume(np.zeros((5, 6, 20), "f4"))   # new shape: view resets
    assert sv.fig["layout"]["scene"]["camera"] != cam


def test_per_axis_stretch_and_tools(line):
    d = np.zeros((8, 10, 40), "f4")
    sv = VolumeView3D(d)
    ar = sv.fig["layout"]["scene"]["aspectratio"]
    base = dict(ar)
    for w in sv.w_stretch:                        # 1.0 sits at the midpoint
        assert w.value == 1.0
        assert w.options[len(w.options) // 2] == 1.0
        assert w.options[0] == 0.125 and w.options[-1] == 8.0
    assert [w.name.split()[0] for w in sv.w_stretch] == ["recy", "recx", "time"]
    sv.w_stretch[2].value = 2.0                   # vertical exaggeration
    assert np.isclose(ar["z"], 2 * base["z"]) and np.isclose(ar["x"], base["x"])
    sv.w_stretch[0].value = 0.5                   # squeeze axis 0
    assert np.isclose(sv.fig["layout"]["scene"]["aspectratio"]["x"],
                      0.5 * base["x"])
    sv.set_volume(np.zeros((8, 10, 40), "f4"))    # stretch survives volume swap
    ar2 = sv.fig["layout"]["scene"]["aspectratio"]
    assert np.isclose(ar2["z"], 2 * base["z"]) and np.isclose(ar2["x"],
                                                              0.5 * base["x"])
    from gathervis.viewer import ImagePane
    from bokeh.models import WheelZoomTool
    dims = {t.dimensions for t in ImagePane().figure.select(WheelZoomTool)}
    assert {"both", "width", "height"} <= dims  # per-axis zooming available


# ---------------- v0.7: trapezoid (Ormsby) filters ----------------
def test_bandpass_response():
    from gathervis.process import bandpass
    dt, nt = 0.002, 1000
    t = np.arange(nt) * dt

    def peak(fsig, **kw):
        x = np.sin(2 * np.pi * fsig * t)[None].astype("f4")
        return float(np.abs(bandpass(x, dt, **kw)).max())

    bp = dict(f1=10, f2=20, f3=60, f4=80)
    assert peak(40, **bp) > 0.99                 # passband untouched
    assert peak(5, **bp) < 1e-3                  # below low cut: gone
    assert peak(120, **bp) < 1e-3                # above high cut: gone
    assert abs(peak(15, f1=10, f2=20) - 0.5) < 0.02   # linear ramp midpoint
    assert peak(40, f3=60, f4=80) > 0.99         # low-pass leaves 40 Hz
    # zero phase: a spike stays centered
    x = np.zeros((1, nt), "f4"); x[0, 500] = 1.0
    y = bandpass(x, dt, f1=5, f2=10, f3=60, f4=80)
    assert int(np.abs(y[0]).argmax()) == 500
    # invalid ramp (f2 <= f1) is ignored, not crashed
    assert peak(40, f1=20, f2=20) > 0.99


def test_workspace_filter_controls(line):
    ws = Workspace(line)
    raw = dict(ws.browser.pane.cds.data)
    ws._w_ftype.value = "band-pass"
    assert ws.state["filter"] == dict(f1=5.0, f2=10.0, f3=60.0, f4=80.0)
    assert ws._row_lo.visible and ws._row_hi.visible
    assert not np.array_equal(ws.browser.pane.cds.data["image"][0],
                              raw["image"][0])   # display actually filtered
    ws._w_ftype.value = "low-pass"
    assert ws.state["filter"] == dict(f3=60.0, f4=80.0)
    assert not ws._row_lo.visible and ws._row_hi.visible
    ws._w_f[3].value = 90.0                      # editing a corner re-applies
    assert ws.state["filter"] == dict(f3=60.0, f4=90.0)
    ws._w_ftype.value = "off"
    assert ws.state["filter"] is None
    assert np.array_equal(ws.browser.pane.cds.data["image"][0],
                          raw["image"][0])       # back to raw display


def test_stretch_range_extended():
    d = np.zeros((8, 10, 40), "f4")
    sv = VolumeView3D(d)
    for w in sv.w_stretch:
        assert w.options[0] == 0.125 and w.options[-1] == 8.0
        assert w.options[len(w.options) // 2] == 1.0    # still centered


# ---------------- v0.8: analysis windows + spectra + export ----------------
def _draw_rect(wt, x, y, w, h):
    d = dict(wt.cds_rect.data)
    for k, v in zip(("x", "y", "width", "height", "color"),
                    (x, y, w, h, "#e6741e")):
        d[k] = list(d[k]) + [v]
    wt.cds_rect.data = d


def test_window_parsing_and_mask():
    from gathervis.viewer import ImagePane, WindowTool
    p = ImagePane()
    wt = WindowTool(p)
    _draw_rect(wt, x=50, y=0.5, w=20, h=0.2)          # rect via center/size
    wt.cds_poly.data = dict(xs=[[0, 10, 0]], ys=[[0.0, 0.0, 0.1]],
                            color=["#2ca02c"])         # triangle
    wins = wt.windows()
    assert [w["kind"] for w in wins] == ["rect", "polygon"]
    assert wins[0]["trace"] == [40.0, 60.0]
    assert wins[0]["time"] == [0.4, 0.6]
    m = WindowTool._mask(wins[0], nx=100, nt=1000, x0=0, dx=1, y0=0, dy=0.002)
    assert m[50, 250] and not m[30, 250] and not m[50, 100]
    assert m.sum() == 21 * 101                        # inclusive rect bounds
    mt = WindowTool._mask(wins[1], nx=20, nt=100, x0=0, dx=1, y0=0, dy=0.002)
    assert 0.3 < mt.sum() / (10 * 50) < 0.7           # ~half the bounding box


def test_window_spectrum_and_export():
    import json
    from gathervis.demo import synthetic_line
    big = synthetic_line(ns=4, nr=96, nt=751)         # 0.5 s window -> 2 Hz resolution
    b = ShotBrowser(big, _state(big))
    assert b.wt is not None
    # inject a 30 Hz sine into traces 20..60, 0.35..0.85 s: peak at 30 Hz
    arr = np.array(b.pane._last[0], copy=True)
    t = np.arange(arr.shape[1]) * big.dt
    arr[20:60] = np.sin(2 * np.pi * 30 * t)[None]
    b.pane.update(arr, b.state["clim"], y0=0, dy=big.dt)
    _draw_rect(b.wt, x=40, y=0.6, w=30, h=0.5)
    b.wt.compute()
    d = b.wt.cds_spec.data
    assert b.wt.spec_fig.visible
    assert len(d["xs"]) > 1                           # one curve per trace
    fpk = d["xs"][0][int(np.argmax(d["ys"][0]))]
    assert abs(fpk - 30.0) < 2.5                      # peak at the sine
    # the legend carries what was plotted, not just a name
    assert d["label"][0].startswith("rect 1 · ")
    assert "tr" in d["label"][0] and "df" in d["label"][0]
    assert len(set(d["label"])) == 1                  # ... as one legend entry
    assert list(b.wt.cds_rect.data["color"]) == [d["color"][0]]  # color-matched
    # after a shot change the visible spectrum recomputes; windows persist
    b.set_shot(2)
    assert len(b.wt.windows()) == 1
    # JSON export
    payload = json.loads(b.wt._export().getvalue())
    assert payload["windows"][0]["kind"] == "rect"
    assert payload["dt"] == big.dt
    assert payload["windows"][0]["trace"] == [25.0, 55.0]


def test_window_tool_in_workspace_2d(line):
    a = np.asarray(line.shot(0))
    ws = Workspace(gv.from_array(a, axes=("rec", "time"), dt=line.dt))
    assert hasattr(ws, "_wt2d")
    _draw_rect(ws._wt2d, x=12, y=0.1, w=10, h=0.12)   # window within the (24-trace, 0.2 s) extents
    ws._wt2d.compute()
    assert ws._wt2d.spec_fig.visible


# ---------------- v0.8.2: cursor readout ----------------
def test_cursor_readout(line):
    from gathervis.viewer import ImagePane
    from gathervis.process import quantize
    b = ShotBrowser(line, _state(line))
    p = b.pane
    # readout Div exists with client-side mousemove/mouseleave callbacks
    assert p.readout is not None
    assert "mousemove" in p.figure.js_event_callbacks
    assert "mouseleave" in p.figure.js_event_callbacks
    # lo/hi ship with the image so JS de-quantizes uint8 back to amplitude
    d = p.cds.data
    assert (d["lo"][0], d["hi"][0]) == b.state["clim"]
    # de-quantization error <= one quantization step (JS uses this formula)
    lo, hi = b.state["clim"]
    arr = np.asarray(p._last[0], dtype="f4")
    q = quantize(arr, (lo, hi)).astype("f4")
    back = lo + q / 255.0 * (hi - lo)
    clipped = np.clip(arr, lo, hi)
    assert np.abs(back - clipped).max() <= (hi - lo) / 255.0 + 1e-6
    # wiggle mode empties the image column -> JS degrades to coords only
    p.set_display("wiggle")
    assert not p.cds.data["image"]
    p.set_display("density")
    assert p.cds.data["lo"][0] == lo


# ---------------- v0.8.3: spectrum readout + crosshair ----------------
def test_spectrum_readout_and_crosshair(line):
    from bokeh.models import CrosshairTool
    b = ShotBrowser(line, _state(line))
    wt = b.wt
    # spectrum figure: readout Div + mousemove callback + crosshair (off)
    assert "mousemove" in wt.spec_fig.js_event_callbacks
    assert wt.spec_fig.select(CrosshairTool)
    assert wt.spec_fig.toolbar.active_inspect == []
    assert wt.spec_readout.visible is False        # shows/hides with the spectrum figure
    _draw_rect(wt, x=12, y=0.1, w=10, h=0.12)
    wt.compute()
    assert wt.spec_readout.visible is True
    # gather panel has the toggleable crosshair too
    assert b.pane.figure.select(CrosshairTool)
    assert b.pane.figure.toolbar.active_inspect == []


# ---------------- v0.9: AGC / trace balance (M2) ----------------
def test_agc_and_trace_balance():
    from gathervis.process import agc, trace_balance
    t = np.arange(2000) * 0.002
    decaying = (np.sin(2 * np.pi * 25 * t) * np.exp(-1.5 * t))[None].astype("f4")
    g = agc(decaying, 0.002, window=0.4)
    early = np.sqrt((g[0, 100:400] ** 2).mean())
    late = np.sqrt((g[0, 1400:1700] ** 2).mean())
    assert 0.8 < early / late < 1.25          # decay flattened
    assert np.isfinite(agc(np.zeros((1, 300), "f4"), 0.002)).all()
    a = np.random.default_rng(0).standard_normal((5, 500)).astype("f4")
    a *= np.array([1, 10, 0.1, 5, 2], "f4")[:, None]
    r = np.sqrt((trace_balance(a) ** 2).mean(axis=1))
    assert r.max() / r.min() < 1.05           # per-trace RMS equalized
    v3 = np.random.default_rng(1).standard_normal((4, 6, 400)).astype("f4")
    assert agc(v3, 0.002).shape == v3.shape   # works on 3-D per-shot volumes too


def test_gain_controls_and_autoclim(line):
    ws = Workspace(line)
    raw_clim = ws.state["clim"]
    raw_img = np.array(ws.browser.pane.cds.data["image"][0], copy=True)
    ws._w_gain.value = "AGC"
    assert ws.state["gain"] == ("agc", 0.5)
    assert ws._w_agcwin.visible
    # display clim recomputed from the processed gather
    from gathervis.viewer import _process_gather
    lo, hi = ws.browser.pane.cds.data["lo"][0], ws.browser.pane.cds.data["hi"][0]
    assert (lo, hi) != raw_clim
    _, expect = _process_gather(ws.g.shot(ws.browser.ishot), ws.g.dt, ws.state)
    assert np.allclose((lo, hi), expect)
    assert not np.array_equal(ws.browser.pane.cds.data["image"][0], raw_img)
    ws._w_agcwin.value = 0.2                  # changing the window re-applies
    assert ws.state["gain"] == ("agc", 0.2)
    ws._w_gain.value = "trace balance"
    assert ws.state["gain"] == ("balance",) and not ws._w_agcwin.visible
    ws._w_gain.value = "off"
    assert ws.state["gain"] is None
    assert np.array_equal(ws.browser.pane.cds.data["image"][0], raw_img)
    # filter + gain chain together
    ws._w_ftype.value = "band-pass"
    ws._w_gain.value = "AGC"
    assert ws.state["filter"] and ws.state["gain"]


# ---------------- v0.10: SEG-Y import, f-k, window import ----------------
def _make_segy(path, ns=3, nr=8, nt=200, ragged=False):
    import segyio
    spec = segyio.spec()
    spec.format = 5
    spec.samples = np.arange(nt)
    spec.tracecount = ns * nr - (2 if ragged else 0)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((ns, nr, nt)).astype("f4")
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = 2000
        k = 0
        for i in range(ns):
            nj = nr - 2 if (ragged and i == 1) else nr
            for j in range(nj):
                f.header[k] = {
                    segyio.TraceField.FieldRecord: 100 + i,
                    segyio.TraceField.SourceX: int((500 + 50 * i) * 100),
                    segyio.TraceField.SourceY: 100000,
                    segyio.TraceField.GroupX: int((300 + 25 * j) * 100),
                    segyio.TraceField.GroupY: 100000,
                    segyio.TraceField.SourceGroupScalar: -100,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: 2000}
                f.trace[k] = data[i, j]
                k += 1
    return data


def test_from_segy(tmp_path):
    pytest.importorskip("segyio")
    p = tmp_path / "t.sgy"
    data = _make_segy(p)
    g = gv.from_segy(p)
    assert g.shape == (3, 8, 200) and g.dt == 0.002
    assert np.allclose(np.asarray(g.data), data, atol=1e-6)
    assert np.allclose(g.geometry.src[:, 0], [500, 550, 600])   # coordinate scalar applied
    assert np.allclose(g.geometry.rec[0, :, 0], 300 + 25 * np.arange(8))
    ws = Workspace(g)                                  # full app incl geometry
    assert "Geometry" in list(ws.tabs._names)
    from gathervis.__main__ import _load, _parser
    g2 = _load(_parser().parse_args([str(p)]))         # CLI autodetect
    assert g2.dt == 0.002 and g2.geometry is not None


def test_from_segy_ragged_and_toobig(tmp_path):
    pytest.importorskip("segyio")
    p = tmp_path / "r.sgy"
    _make_segy(p, ragged=True)
    with pytest.warns(UserWarning, match="ragged"):
        g = gv.from_segy(p)
    assert g.shape == (3, 8, 200)
    assert np.allclose(np.asarray(g.data)[1, 6:], 0)   # zero-padded
    with pytest.raises(MemoryError, match="max_gb"):
        gv.from_segy(p, max_gb=1e-9)


def test_fk_spectrum_and_window_import():
    from gathervis.viewer import ImagePane, WindowTool
    nx, nt, dt = 96, 800, 0.002
    f0, k0 = 30.0, 0.15
    x = np.arange(nx)[:, None]
    t = np.arange(nt)[None] * dt
    a = np.sin(2 * np.pi * (f0 * t - k0 * x)).astype("f4")
    p = ImagePane()
    p.update(a, robust_clim(a), y0=0, dy=dt)
    wt = WindowTool(p)
    wt.cds_rect.data = dict(x=[48.], y=[0.8], width=[80.], height=[1.2],
                            color=[""])
    wt.compute_fk()
    d = wt.cds_fk.data
    img = d["image"][0]
    ks = d["x"][0] + np.arange(img.shape[1]) / (img.shape[1] - 1) * d["dw"][0]
    fs = d["y"][0] + np.arange(img.shape[0]) / (img.shape[0] - 1) * d["dh"][0]
    i, j = np.unravel_index(np.argmax(img), img.shape)
    assert abs(ks[j] - k0) < 0.02          # +x-travelling wave lands at +k (geophysics convention)
    assert abs(fs[i] - f0) < 2.0
    assert wt.fk_fig.visible and wt.fk_readout.visible
    # export -> clear -> import round-trip
    raw = wt._export().getvalue().encode()
    wt.cds_rect.data = dict(x=[], y=[], width=[], height=[], color=[])

    class _E:
        new = raw

    wt._import(_E())
    assert wt.windows()[0]["trace"] == [8.0, 88.0]


# ---------------- v0.11: event picking ----------------
def test_pick_tool_per_shot_and_csv(line):
    b = ShotBrowser(line, _state(line))
    pt = b.pt
    # pick 3 first breaks on shot 0
    pt.cds.data = dict(x=[5.0, 1.0, 3.0], y=[0.05, 0.01, 0.03])
    # connecting line is sorted by trace
    assert list(pt._cds_line.data["x"]) == [1.0, 3.0, 5.0]
    # change shot -> picks follow (shot 0 stored, shot 2 empty)
    b.set_shot(2)
    assert pt.cds.data["x"] == []
    pt.cds.data = dict(x=[2.0], y=[0.02])
    b.set_shot(0)                              # back to shot 0 -> restored
    assert sorted(pt.cds.data["x"]) == [1.0, 3.0, 5.0]
    assert pt.picks() == {0: ([5.0, 1.0, 3.0], [0.05, 0.01, 0.03]),
                          2: ([2.0], [0.02])}
    # CSV export: header + sorted by shot then trace
    csv = pt._export().getvalue()
    lines = csv.strip().splitlines()
    assert lines[0] == "shot,trace,time_s"
    assert lines[1].startswith("0,1.00,") and lines[4].startswith("2,2.00,")
    # import round-trip (header tolerated)
    pt2 = b.pt

    class _E:
        new = csv.encode()

    pt.cds.data = dict(x=[], y=[])
    pt._store = {}
    pt._import(_E())
    assert sorted(pt.cds.data["x"]) == [1.0, 3.0, 5.0]   # current shot (0) restored
    b.set_shot(2)
    assert pt.cds.data["x"] == [2.0]
    # PointDrawTool attached to the figure
    from bokeh.models import PointDrawTool
    assert b.pane.figure.select(PointDrawTool)


# ---------------- v0.12: snap + STA/LTA auto pick ----------------
def test_sta_lta_picker():
    from gathervis.process import pick_first_breaks
    rng = np.random.default_rng(0)
    dt, nt = 0.002, 1000
    t = np.arange(nt) * dt
    truth = 0.1 + 0.005 * np.arange(48)          # known wavelet center per trace
    def ricker(tc):
        u = (t - tc) * 25 * np.pi
        return (1 - 2 * u * u) * np.exp(-u * u)
    a = np.stack([ricker(tc) for tc in truth]).astype("f4")
    a += 0.01 * rng.standard_normal(a.shape)
    picks = pick_first_breaks(a, dt, sta=0.02, lta=0.15, thresh=4)
    err = picks - truth
    assert np.isfinite(picks).all()
    assert np.nanstd(err) < 0.01                 # consistency: spread < 10 ms
    assert -0.04 < np.nanmean(err) < 0.0         # triggers at onset (before center)
    dead = np.zeros((2, nt), "f4")
    assert np.isnan(pick_first_breaks(dead, dt)).all()


def test_snap_and_auto_pick_ui():
    from gathervis.viewer import ImagePane, PickTool
    dt, nt, nx = 0.002, 600, 32
    t = np.arange(nt) * dt
    a = np.zeros((nx, nt), "f4")
    for j in range(nx):                          # peak at 0.30 s, trough at 0.34 s
        a[j] += np.exp(-((t - 0.30) / 0.008) ** 2)
        a[j] -= np.exp(-((t - 0.34) / 0.008) ** 2)
    p = ImagePane()
    p.update(a, robust_clim(a), y0=0, dy=dt)
    pt = PickTool(p)
    pt.w_snap.value = "peak"
    pt.cds.data = dict(x=[10.4], y=[0.312])      # slightly off -> snaps to peak + integer trace
    assert pt.cds.data["x"] == [10.0]
    assert abs(pt.cds.data["y"][0] - 0.30) <= dt
    pt.w_snap.value = "trough"
    pt.cds.data = dict(x=[10.0], y=[0.325])
    assert abs(pt.cds.data["y"][0] - 0.34) <= dt
    pt.w_snap.value = "|max|"
    pt.cds.data = dict(x=[10.0], y=[0.31])
    assert abs(pt.cds.data["y"][0] - 0.30) <= dt  # |max| = the positive peak here
    # auto pick: known sloping onset times
    truth = 0.1 + 0.004 * np.arange(nx)
    b = np.stack([np.exp(-((t - tc) / 0.01) ** 2) for tc in truth]).astype("f4")
    b += 0.005 * np.random.default_rng(1).standard_normal(b.shape)
    p.update(b, robust_clim(b), y0=0, dy=dt)
    pt.w_snap.value = "off"
    pt.auto_pick()
    assert len(pt.cds.data["x"]) == nx
    got = np.asarray(pt.cds.data["y"])
    assert np.std(got - truth) < 0.012            # tracks the moveout slope consistently
    csv = pt._export().getvalue()                 # exports directly
    assert csv.count("\n") == nx + 1


# ---------------- v0.13.1: clear picks ----------------
def test_clear_picks(line):
    b = ShotBrowser(line, _state(line))
    pt = b.pt
    pt.cds.data = dict(x=[1.0, 2.0], y=[0.01, 0.02])
    b.set_shot(2)
    pt.cds.data = dict(x=[3.0], y=[0.03])
    pt.clear_shot()                     # only shot 2 cleared
    assert pt.cds.data["x"] == [] and pt.picks() == {
        0: ([1.0, 2.0], [0.01, 0.02])}
    b.set_shot(0)
    assert pt.cds.data["x"] == [1.0, 2.0]
    pt.clear_all()
    assert pt.picks() == {} and pt.cds.data["x"] == []
    b.set_shot(2)                       # nothing resurrects
    assert pt.cds.data["x"] == []


# ---------------- v0.13.2: clear windows / clear picks in FB card ----------
def test_clear_windows(line):
    b = ShotBrowser(line, _state(line))
    wt = b.wt
    _draw_rect(wt, x=12, y=0.1, w=10, h=0.12)
    wt.cds_poly.data = dict(xs=[[0, 5, 5]], ys=[[0.0, 0.0, 0.1]], color=[""])
    wt.compute()
    wt.compute_fk()
    assert wt.spec_fig.visible
    wt.clear_windows()
    assert not wt.windows()
    assert not wt.cds_spec.data["xs"] and not wt.cds_fk.data["image"]
    assert not wt.spec_fig.visible and not wt.fk_fig.visible
    assert not wt.spec_readout.visible and not wt.fk_readout.visible
    b.set_shot(1)                       # refresh path stays quiet afterwards
    assert not wt.spec_fig.visible


def test_fb_clear_button(line):
    b = ShotBrowser(line, _state(line))
    b.pt.cds.data = dict(x=[3.0], y=[0.05])
    b.pt.w_clear_fb.clicks += 1         # FB-card button -> clear_shot
    assert b.pt.cds.data["x"] == []


# ---------------- v0.13.3: gesture cheat-sheet ----------------
def _find(obj, cls, out=None):
    out = [] if out is None else out
    if isinstance(obj, cls):
        out.append(obj)
    for c in getattr(obj, "objects", []):
        _find(c, cls, out)
    return out


@pytest.mark.parametrize("tools", ["sidebar", "below"])
def test_gesture_help_toggle(line, tools):
    from gathervis.viewer import Workspace
    ws = Workspace(line, tools=tools)
    # the help button follows the cards: sidebar by default, else under the plot
    body = pn.Column(*ws.sidebar()) if tools == "sidebar" else ws.browser.panel()

    btns = [b for b in _find(body, pn.widgets.Button)
            if b.name == "? gestures"]
    assert len(btns) == 1
    helps = [h for h in _find(body, pn.pane.HTML)
             if "Toolbar gestures" in str(h.object)]
    assert len(helps) == 1 and helps[0].visible is False
    btns[0].clicks += 1                    # toggle open
    assert helps[0].visible is True
    btns[0].clicks += 1                    # toggle closed
    assert helps[0].visible is False
    assert "SHIFT+drag" in str(helps[0].object)
    assert "BACKSPACE" in str(helps[0].object)


# ---------------- v0.13.4: spectrum position + size ----------------
def test_spectrum_position_and_size(line):
    b = ShotBrowser(line, _state(line), tools="below")
    body = b.panel()
    col = body[1]                          # figure column of the 2-D branch
    kinds = [type(o).__name__ for o in col.objects]
    # size controls, gather figure, readout, spectra column, cards, help
    assert kinds[:3] == ["Row", "Bokeh", "Bokeh"]
    assert "FlexBox" in kinds              # cards still under the figure
    spec_col = col.objects[kinds.index("Column")]
    wrapped = [getattr(o, "object", None) for o in spec_col.objects]
    assert b.wt.spec_fig in wrapped        # spectra column holds the figure
    # default M size: smaller + width-capped
    assert b.wt.spec_fig.height == 240 and b.wt.spec_fig.max_width == 900
    b.wt.w_size.value = "S"
    assert b.wt.spec_fig.height == 180 and b.wt.fk_fig.max_width == 650
    b.wt.w_size.value = "L"
    assert b.wt.spec_fig.height == 320 and b.wt.spec_fig.max_width is None


# ---------------- tool cards in the sidebar ----------------
_CARD_TITLES = {"Window", "Spectrum", "Event picking", "FB picking"}


def _card_titles(obj):
    return {c.title for c in _find(obj, pn.Card)} & _CARD_TITLES


def test_tool_cards_default_to_sidebar(line):
    ws = Workspace(line)
    assert _card_titles(pn.Column(*ws.sidebar())) == _CARD_TITLES
    # and no longer under the figure
    assert _card_titles(ws.browser.panel()) == set()


def test_tool_cards_below_is_still_available(line):
    ws = Workspace(line, tools="below")
    assert _card_titles(ws.browser.panel()) == _CARD_TITLES
    assert _card_titles(pn.Column(*ws.sidebar())) == set()


def test_sidebar_cards_start_collapsed_and_stretch(line):
    ws = Workspace(line)
    cards = [c for c in _find(pn.Column(*ws.sidebar()), pn.Card)
             if c.title in _CARD_TITLES]
    assert all(c.collapsed for c in cards)
    assert all(c.sizing_mode == "stretch_width" for c in cards)


def test_sidebar_cards_hidden_on_non_gather_tabs(line):
    ws = Workspace(line)                   # shot gathers + Geometry tabs
    other = [i for i in range(len(ws.tabs)) if i not in ws._tool_tabs]
    assert other, "need a tab the cards do not apply to"
    assert ws._tools_box.visible is True    # starts on the shot-gather tab
    ws.tabs.active = other[0]
    assert ws._tools_box.visible is False
    ws.tabs.active = ws._shot_tab
    assert ws._tools_box.visible is True


def test_tab_switch_preserves_card_state(line):
    """Visibility is toggled on the container, so a card the user opened is
    still open when they come back from another tab."""
    ws = Workspace(line)
    card = [c for c in _find(pn.Column(*ws.sidebar()), pn.Card)
            if c.title == "FB picking"][0]
    card.collapsed = False
    other = [i for i in range(len(ws.tabs)) if i not in ws._tool_tabs][0]
    ws.tabs.active = other
    ws.tabs.active = ws._shot_tab
    assert card.collapsed is False


def test_sidebar_cards_are_shared_not_duplicated(line):
    """sidebar() is called by both panel() and template(); widgets must not
    be rebuilt, or the two copies would drift apart."""
    ws = Workspace(line)
    assert ws.sidebar()[-1] is ws.sidebar()[-1]
    a = [c for c in _find(pn.Column(*ws.sidebar()), pn.Card)
         if c.title == "FB picking"][0]
    bcard = [c for c in _find(pn.Column(*ws.sidebar()), pn.Card)
             if c.title == "FB picking"][0]
    assert a is bcard


def test_tools_argument_is_validated(line):
    with pytest.raises(ValueError, match="tools must be"):
        gv.show(np.zeros((4, 8, 16), "f4"), tools="left")


# ---------------- panel sizing: height / fit / fullscreen ----------------
def test_height_slider_drives_the_figure(line):
    b = ShotBrowser(line, _state(line))
    assert b.pane.figure.height == b.pane.w_height.value
    b.pane.w_height.value = 900
    assert b.pane.figure.height == 900
    b.pane.w_height.value = 300
    assert b.pane.figure.height == 300


def test_size_controls_are_above_the_figure(line):
    b = ShotBrowser(line, _state(line))
    col = b.panel()[1]
    row = col.objects[0]
    assert isinstance(row, pn.Row)
    assert [b.pane.w_height, b.pane.w_fit, b.pane.w_fullscreen,
            b.pane.w_download] == list(row.objects)


def test_frame_carries_the_fullscreen_classes(line):
    b = ShotBrowser(line, _state(line))
    col = b.panel()[1]
    assert "gv-plot" in col.css_classes         # styled by _PLOT_CSS
    assert b.pane._cls in col.css_classes       # unique fullscreen target


def test_each_pane_gets_a_distinct_fullscreen_target(line):
    """Two panes on one page must not fullscreen each other."""
    from gathervis.viewer import ImagePane
    a, b = ImagePane(), ImagePane()
    assert a._cls != b._cls


def test_fit_and_fullscreen_have_js_bound(line):
    b = ShotBrowser(line, _state(line))
    for cb in (b.pane.js_fit, b.pane.js_fullscreen):
        # both drive the same slider, so Python state stays in sync
        assert cb.args["sl"] is b.pane.w_height
    fs = b.pane.js_fullscreen
    assert fs.args["cls"] == b.pane._cls
    code = "".join(fs.code.values())
    assert "requestFullscreen" in code and "exitFullscreen" in code
    assert "shadowRoot" in code            # Panel may render into shadow DOM


def test_fit_clamps_to_the_slider_range(line):
    """The JS writes through the slider, so its bounds are the clamp."""
    b = ShotBrowser(line, _state(line))
    code = "".join(b.pane.js_fit.code.values())
    assert "Math.min(sl.end" in code and "Math.max(sl.start" in code
    assert b.pane.js_fit.args["chrome"] > 0


# ---------------- v0.13.5: polarity flip ----------------
def test_polarity_flip(line):
    ws = Workspace(line)
    flip = ws._controls[1]
    assert flip.name == "flip polarity"
    raw = np.array(ws.browser.pane.cds.data["image"][0], dtype=np.int16)
    flip.value = True
    assert ws.state["flip"] is True
    flipped = np.array(ws.browser.pane.cds.data["image"][0], dtype=np.int16)
    # negating the data mirrors the quantized image: q(-x) = 255 - q(x) +-1
    assert np.abs((flipped + raw) - 255).max() <= 1
    # wiggle fills flip too (through the same processing chain)
    ws._controls[0].value = "wiggle"
    fills_flipped = [np.array(x) for x in ws.browser.pane._cds_fill.data["xs"]]
    flip.value = False
    fills_raw = [np.array(x) for x in ws.browser.pane._cds_fill.data["xs"]]
    assert any(not np.array_equal(a, b)
               for a, b in zip(fills_flipped, fills_raw))
    ws._controls[0].value = "density"
    back = np.array(ws.browser.pane.cds.data["image"][0], dtype=np.int16)
    assert np.array_equal(back, raw)       # off restores exactly


# ---------------- v0.14: processing chain on the volume tab ----------------
def test_volume_tab_follows_chain(line):
    from gathervis.process import bandpass
    ws = Workspace(line)
    raw = ws.g.data
    assert ws.slices.vol is raw
    ws._w_ftype.value = "band-pass"        # chain on -> processed volume
    assert ws.slices.vol is not raw
    expect = bandpass(np.asarray(raw, dtype=np.float32), line.dt,
                      f1=5.0, f2=10.0, f3=60.0, f4=80.0)
    assert np.allclose(ws.slices.vol, expect, atol=1e-5)
    # the three planes now come from the same processed cube (consistent),
    # including the time slice
    k = ws.slices.w2.value
    assert np.allclose(np.asarray(ws.slices.vol[:, :, k]), expect[:, :, k])
    ws._controls[1].value = True           # + flip
    assert np.allclose(ws.slices.vol, -expect, atol=1e-5)
    ws._controls[1].value = False
    ws._w_gain.value = "AGC"               # + gain -> clim rescaled
    assert ws.slices.clim != ws.state["clim"]
    ws._w_gain.value = "off"
    ws._w_ftype.value = "off"              # chain off -> raw memmap identity
    assert ws.slices.vol is raw
    assert not ws._proc_note.visible


def test_volume_too_big_falls_back(line, monkeypatch):
    import gathervis.viewer as V
    ws = Workspace(line)
    monkeypatch.setattr(V, "PROC_MAX_BYTES", 10)   # force the fallback
    ws._w_ftype.value = "band-pass"
    assert ws.slices.vol is ws.g.data              # stays raw
    assert ws._proc_note.visible                   # and says so
    ws._w_ftype.value = "off"
    assert not ws._proc_note.visible


def test_depth_volume_untouched_by_chain():
    vel = np.random.default_rng(0).uniform(1600, 3300, (20, 16, 12)) \
        .astype("f4")
    ws = Workspace(gv.from_array(vel, axes=("x", "y", "depth"), dt=10.0))
    ws._w_ftype.value = "band-pass"        # meaningless along depth: raw
    assert ws.slices.vol is ws.g.data
    assert not ws._proc_note.visible       # and no scary note either


# ---------------- v0.15: full-resolution image export ----------------
def _decode_png(blob):
    """Minimal stdlib PNG reader (8-bit RGB, 'Up' filter) for the tests."""
    import struct
    import zlib
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    i, idat, hdr = 8, b"", None
    while i < len(blob):
        n = struct.unpack(">I", blob[i:i + 4])[0]
        tag, data = blob[i + 4:i + 8], blob[i + 8:i + 8 + n]
        crc = struct.unpack(">I", blob[i + 8 + n:i + 12 + n])[0]
        assert crc == zlib.crc32(tag + data) & 0xFFFFFFFF
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", data)
        elif tag == b"IDAT":
            idat += data
        i += 12 + n
    w, h, depth, color, comp, filt, inter = hdr
    assert (depth, color, comp, filt, inter) == (8, 2, 0, 0, 0)
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * 3 + 1)
    assert (raw[:, 0] == 2).all()                  # every row uses filter "Up"
    up = np.cumsum(raw[:, 1:].astype(np.uint16), axis=0, dtype=np.uint16) % 256
    return up.astype(np.uint8).reshape(h, w, 3)


def test_png_roundtrip_is_lossless():
    from gathervis.render import png_bytes
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    assert np.array_equal(_decode_png(png_bytes(rgb)), rgb)
    with pytest.raises(ValueError):
        png_bytes(np.zeros((4, 4), np.uint8))      # needs (h, w, 3)


def test_auto_px_and_raster_size():
    from gathervis.render import auto_px, raster_size
    assert auto_px("density") == (1, 1)            # 0 = auto = original res
    assert auto_px("wiggle") == (8, 1)             # wiggle needs room to swing
    assert auto_px("density", 3, 2) == (3, 2)      # explicit wins
    assert raster_size(40, 300, "density") == (40, 300)
    assert raster_size(40, 300, "density", 2, 3) == (80, 900)
    w, _ = raster_size(40, 300, "wiggle")          # + clipped-excursion margin
    assert w == 40 * 8 + 2 * 16


def test_render_density_is_the_data_itself():
    from gathervis.process import quantize
    from gathervis.render import palette_rgb, render_density
    a = np.linspace(-1, 1, 24 * 40, dtype="f4").reshape(24, 40)
    img = render_density(a, (-1.0, 1.0), cmap="gray")
    assert img.shape == (40, 24, 3)                # (samples, traces) -> rows
    assert np.array_equal(img, palette_rgb("gray")[quantize(a, (-1, 1)).T])
    assert np.array_equal(render_density(a, (-1, 1), cmap="gray",
                                         flip_y=False), img[::-1])
    big = render_density(a, (-1, 1), cmap="gray", px_trace=3, px_sample=2)
    assert big.shape == (80, 72, 3)


def test_render_wiggle_draws_every_trace():
    from gathervis.render import render_wiggle
    nx, nt, px = 30, 120, 8
    a = np.zeros((nx, nt), "f4")
    a[:, 60] = 1.0                                 # one positive spike each
    img = render_wiggle(a, (-1.0, 1.0), px_trace=px)
    assert img.shape == (nt, nx * px + 2 * 2 * px, 3)
    ink = img[..., 0] < 128
    assert ink.any() and not ink.all()
    # every trace is drawn (no MAX_WIGGLE decimation): each baseline has ink
    for i in range(nx):
        assert ink[:, int(round(2 * px + px * (i + 0.5)))].any()
    # the spike row is filled to the right of its baseline (variable area)
    assert ink[60].sum() > ink[0].sum()


def test_export_budget_is_enforced():
    import gathervis.render as R
    assert R.MAX_EXPORT_PIXELS > 10_000_000        # generous by default
    small = np.zeros((100, 100), "f4")             # checked before allocating
    for fn in (R.render_density, R.render_wiggle):
        with pytest.raises(MemoryError, match="budget"):
            fn(small, (-1.0, 1.0), px_trace=100, px_sample=100)


def test_download_button_sits_with_fit_and_fullscreen(line):
    ws = Workspace(line)
    pane = ws.browser.pane
    row = list(pane.size_controls())
    assert row[-1] is pane.w_download                  # right of fullscreen
    assert pane.w_download.description.startswith("download full image")
    assert not pane.w_download.disabled
    assert pane.w_download in pane.frame().select(pn.widgets.FileDownload)


def test_download_is_full_resolution_unlike_the_panel():
    """The panel ships a decimated image; the download must not be."""
    g = synthetic_line(ns=2, nr=40, nt=2000)       # nt > MAX_PX[1] = 1600
    pane = Workspace(g).browser.pane
    shown = np.asarray(pane.cds.data["image"][0])
    assert shown.shape[0] < 2000                   # screen copy is decimated
    png = _decode_png(pane.w_download.callback().getvalue())
    assert png.shape == (2000, 40, 3)              # download is not
    assert "40 × 2000 px" in pane.w_download.description


def test_download_follows_the_display_chain(line):
    from gathervis.process import quantize
    from gathervis.render import palette_rgb
    ws = Workspace(line)
    pane = ws.browser.pane
    w_disp, w_flip, w_cmap = ws._controls[0], ws._controls[1], ws._controls[2]

    raw = _decode_png(pane.w_download.callback().getvalue())
    arr, clim = pane._last[0], pane._last[1]
    assert np.array_equal(raw, palette_rgb("seismic")[quantize(arr, clim).T])

    w_cmap.value = "petrel"                        # colormap
    ws._w_ftype.value = "band-pass"                # filter
    ws._w_gain.value = "AGC"                       # gain (rescales the clim)
    w_flip.value = True                            # polarity
    proc, clim2 = pane._last[0], pane._last[1]
    assert not np.allclose(proc, arr)              # the chain really applied
    assert clim2 != clim                           # ... and the clim followed
    img = _decode_png(pane.w_download.callback().getvalue())
    assert np.array_equal(img, palette_rgb("petrel")[quantize(proc, clim2).T])

    w_disp.value = "wiggle"                        # display mode
    nx, nt = proc.shape
    wig = _decode_png(pane.w_download.callback().getvalue())
    assert wig.shape == (nt, nx * 8 + 2 * 16, 3)   # every trace, room to swing
    assert f"{nx * 8 + 2 * 16} × {nt} px" in pane.w_download.description


def test_download_is_guarded_by_the_budget(line, monkeypatch):
    import gathervis.render as R
    ws = Workspace(line)
    pane = ws.browser.pane
    nx, nt = pane._last[0].shape
    monkeypatch.setattr(R, "MAX_EXPORT_PIXELS", nx * nt - 1)
    ws.browser.redraw()                            # any redraw resyncs it
    assert pane.w_download.disabled
    assert "too large" in pane.w_download.description
    monkeypatch.setattr(R, "MAX_EXPORT_PIXELS", nx * nt)
    ws.browser.redraw()
    assert not pane.w_download.disabled


def test_download_filename_follows_the_shot():
    # deliberately NOT the module-scoped `line`: earlier tests rename it
    g = synthetic_line(ns=5, nr=12, nt=48)
    assert Workspace(g).browser.pane.w_download.filename == "gather_shot0000.png"
    g.name = "/data/line 07.npy"                   # named dataset -> slugged
    ws = Workspace(g)
    assert ws.browser.pane.w_download.filename == "line_07_shot0000.png"
    ws.browser.set_shot(4)
    assert ws.browser.pane.w_download.filename == "line_07_shot0004.png"


def test_download_offered_for_single_gathers_too():
    from gathervis.viewer import view_gather
    a = synthetic_line(ns=1, nr=20, nt=64).shot(0)
    ws = Workspace(gv.from_array(np.asarray(a), dt=0.002))
    dl = ws._pane2d.w_download
    assert dl.filename == "gather.png"
    assert _decode_png(dl.callback().getvalue()).shape == (64, 20, 3)
    app = view_gather(a, dt=0.002, title="shot 1")
    assert len(app.select(pn.widgets.FileDownload)) == 1


def test_no_export_card_anywhere(line):
    """The download is one button, not a card full of options."""
    ws = Workspace(line)
    titles = [c.title for c in ws.tabs[0].select(pn.Card)]
    titles += [c.title for c in pn.Column(*ws.sidebar()).select(pn.Card)]
    assert "Export image" not in titles
    assert not hasattr(ws.browser, "ex")


def test_per_shot_3d_has_no_download():
    g = gv.from_array(np.zeros((2, 4, 5, 20), "f4"), dt=0.004)
    ws = Workspace(g)                              # 4-D: no 2-D pane at all
    assert not hasattr(ws.browser, "pane")


def test_save_png_script_helper(tmp_path):
    from gathervis.render import save_png
    a = synthetic_line(ns=1, nr=16, nt=80).shot(0)
    out = save_png(tmp_path / "shot.png", a, cmap="gray")
    img = _decode_png(open(out, "rb").read())
    assert img.shape == (80, 16, 3)
    assert (img[..., 0] == img[..., 1]).all()      # gray cmap -> r == g == b
    wig = save_png(tmp_path / "w.png", a, display="wiggle", px_trace=6)
    assert _decode_png(open(wig, "rb").read()).shape[1] == 16 * 6 + 2 * 12


# ---------------- v0.15: consistent card styling, measured fit ----------------
def _tool_cards(ws):
    box = pn.Column(*ws.sidebar())
    return [c for c in box.select(pn.Card)]


def test_cards_share_one_flat_style(line):
    from gathervis.viewer import _CARD_CSS
    cards = _tool_cards(Workspace(line))
    assert len(cards) == 4                         # Window/Spectrum/picking/FB
    for c in cards:
        assert c.stylesheets == [_CARD_CSS]        # same sheet, not per-card
        assert "box-shadow: none" in _CARD_CSS     # ... and it kills the shadow
        assert c.sizing_mode == "stretch_width"
        assert c.margin == (4, 0)
        assert c.collapsed


def test_card_controls_are_uniformly_sized(line):
    """No hand-picked widths inside a card: everything stretches."""
    for card in _tool_cards(Workspace(line)):
        for w in card.select(pn.widgets.Widget):
            assert w.width is None, f"{card.title}/{w.name} pins a width"
            assert w.sizing_mode == "stretch_width", f"{card.title}/{w.name}"
            assert w.margin == (4, 0), f"{card.title}/{w.name}"


def test_card_buttons_use_one_colour_scheme(line):
    """One accent per card for its main action; everything else neutral."""
    accents = {}
    for card in _tool_cards(Workspace(line)):
        kinds = [b.button_type for b in card.select(pn.widgets.Button)]
        assert "light" not in kinds                # the flat/text odd-one-out
        assert set(kinds) <= {"default", "primary"}
        accents[card.title] = kinds.count("primary")
        assert accents[card.title] <= 1
    assert accents["Spectrum"] == 1 and accents["FB picking"] == 1
    assert accents["Window"] == 0 and accents["Event picking"] == 0


def test_file_inputs_are_restyled(line):
    from gathervis.viewer import _FILE_CSS
    inputs = [w for card in _tool_cards(Workspace(line))
              for w in card.select(pn.widgets.FileInput)]
    assert len(inputs) == 2                        # windows.json + picks.csv
    for w in inputs:
        assert w.stylesheets == [_FILE_CSS]
        assert "::file-selector-button" in _FILE_CSS


def test_fit_measures_the_layout_instead_of_guessing(line):
    b = ShotBrowser(line, _state(line))
    fit = b.pane.js_fit
    code = "".join(fit.code.values())
    # the container class is passed in, so the JS can measure the real box
    assert fit.args["cls"] == b.pane._cls
    assert "getBoundingClientRect" in code
    assert "r.height - sl.value" in code           # chrome inside the column
    assert "window.innerHeight - r.top" in code    # chrome above it
    # fullscreen re-uses the same measurement, not a screen-size guess
    fs = "".join(b.pane.js_fullscreen.code.values())
    assert "fitHeight(el, sl" in fs and "screen.height" not in fs


def test_fit_falls_back_when_the_container_is_missing(line):
    import gathervis.viewer as V
    code = "".join(ShotBrowser(line, _state(line)).pane.js_fit.code.values())
    assert "if (!el) return Math.round(window.innerHeight - chrome)" in code
    assert V._FIT_CHROME_PX == 215                 # fallback only


def test_download_button_is_text_not_a_bare_icon(line):
    dl = Workspace(line).browser.pane.w_download
    assert "full image" in dl.label                # reads, not just an arrow
    assert dl.button_type == "light"               # matches fit / fullscreen


# ---------------- v0.15: raw-DFT window spectra ----------------
def test_spectrum_is_the_plain_dft_of_the_gated_samples(line):
    """No taper, no normalization, no averaging: just |rfft| in dB."""
    b = ShotBrowser(line, _state(line))
    b.wt.w_traces.value = "middle trace"
    b.wt.w_scale.value = "dB"
    _draw_rect(b.wt, x=12, y=0.08, w=10, h=0.1)
    b.wt.compute()
    d = b.wt.cds_spec.data
    assert len(d["xs"]) == 1

    arr, _c, x0, dx, y0, dy = b.pane._last
    arr = np.asarray(arr, dtype="f4")
    m = b.wt._mask(b.wt.windows()[0], *arr.shape, x0, dx, y0, dy)
    cols, rows = np.where(m.any(1))[0], np.where(m.any(0))[0]
    k0, k1 = int(rows[0]), int(rows[-1]) + 1
    spec = np.abs(np.fft.rfft(arr[cols, k0:k1] * m[cols, k0:k1], axis=-1))
    mid = spec[cols.size // 2]
    want = 20 * np.log10(mid / (mid.max() + 1e-30) + 1e-6)
    assert np.allclose(d["ys"][0], want, atol=1e-4)
    assert np.allclose(d["xs"][0], np.fft.rfftfreq(k1 - k0, dy), atol=1e-6)


def test_single_trace_spectrum_keeps_its_scatter(line):
    """A single trace's periodogram scatter must survive to the plot; only
    `mean` is allowed to reduce it, and only for independent traces."""
    g = synthetic_line(ns=2, nr=120, nt=513, noise=0.0)
    b = ShotBrowser(g, _state(g))
    arr = np.asarray(b.pane._last[0]).copy()
    rng = np.random.default_rng(0)
    arr += rng.standard_normal(arr.shape).astype("f4")   # independent per trace
    b.pane.update(arr, b.state["clim"], y0=0, dy=g.dt)
    _draw_rect(b.wt, x=60, y=0.5, w=120, h=0.9)
    rough = {}
    b.wt.w_scale.value = "dB"
    for mode in ("middle trace", "mean"):
        b.wt.w_traces.value = mode
        b.wt.compute()
        y = np.asarray(b.wt.cds_spec.data["ys"][0], dtype=float)
        rough[mode] = float(np.abs(np.diff(y, 2)).mean())
    assert rough["middle trace"] > 4 * rough["mean"], rough


def test_mean_only_smooths_when_traces_are_independent(line):
    """On a coherent gather the traces are near-copies, so averaging them
    changes very little -- the smoothness in a real panel comes from many
    *independent* traces, not from the averaging alone."""
    g = synthetic_line(ns=2, nr=120, nt=513, noise=0.0)   # near-identical traces
    b = ShotBrowser(g, _state(g))
    _draw_rect(b.wt, x=60, y=0.5, w=120, h=0.9)
    rough = {}
    b.wt.w_scale.value = "dB"
    for mode in ("middle trace", "mean"):
        b.wt.w_traces.value = mode
        b.wt.compute()
        y = np.asarray(b.wt.cds_spec.data["ys"][0], dtype=float)
        rough[mode] = float(np.abs(np.diff(y, 2)).mean())
    assert rough["mean"] > 0.5 * rough["middle trace"], rough


def test_per_trace_mode_draws_every_trace_as_one_legend_entry(line):
    import gathervis.viewer as V
    b = ShotBrowser(line, _state(line))
    b.wt.w_traces.value = "per trace"
    _draw_rect(b.wt, x=12, y=0.1, w=20, h=0.16)
    b.wt.compute()
    d = b.wt.cds_spec.data
    ntr = len(np.where(b.wt._mask(b.wt.windows()[0], *b.pane._last[0].shape,
                                  b.pane._last[2], b.pane._last[3],
                                  b.pane._last[4], b.pane._last[5])
                       .any(axis=1))[0])
    assert len(d["xs"]) == min(ntr, V._MAX_SPEC_CURVES)
    assert len(set(d["label"])) == 1               # bokeh shows one row
    assert set(d["alpha"]) and max(d["alpha"]) <= 1.0
    assert b.wt.w_traces.value in ("per trace",)


def test_spectrum_curve_count_is_capped(line):
    import gathervis.viewer as V
    big = synthetic_line(ns=2, nr=400, nt=301)
    b = ShotBrowser(big, _state(big))
    _draw_rect(b.wt, x=200, y=0.3, w=400, h=0.3)
    b.wt.compute()
    n = len(b.wt.cds_spec.data["xs"])
    assert 40 <= n <= V._MAX_SPEC_CURVES           # subsampled, not 400
    assert "of 400 tr" in b.wt.cds_spec.data["label"][0]


def test_spectrum_card_controls(line):
    ws = Workspace(line)
    card = [c for c in pn.Column(*ws.sidebar()).select(pn.Card)
            if c.title == "Spectrum"][0]
    names = [w.name for w in card.select(pn.widgets.Widget)]
    assert names == ["compute spectrum", "traces", "y axis",
                     "f-k spectrum", "size"]


def test_y_axis_switches_between_amplitude_and_db(line):
    b = ShotBrowser(line, _state(line))
    assert b.wt.w_scale.value == "amplitude"        # linear by default
    _draw_rect(b.wt, x=12, y=0.1, w=20, h=0.16)
    for mode in ("mean", "middle trace", "per trace"):
        b.wt.w_traces.value = mode
        b.wt.w_scale.value = "amplitude"
        b.wt.compute()
        ys = [np.asarray(y, dtype=float) for y in b.wt.cds_spec.data["ys"]]
        assert b.wt.spec_fig.yaxis[0].axis_label == "amplitude"
        assert min(y.min() for y in ys) >= 0.0      # a ratio, never negative
        assert abs(max(y.max() for y in ys) - 1.0) < 1e-6   # peak is 1.0
        b.wt.w_scale.value = "dB"                   # same data, log scale
        b.wt.compute()
        dbs = [np.asarray(y, dtype=float) for y in b.wt.cds_spec.data["ys"]]
        assert b.wt.spec_fig.yaxis[0].axis_label == "amplitude (dB)"
        assert abs(max(d.max() for d in dbs)) < 1e-4    # loudest curve = 0 dB
        # every curve is exactly the log of its linear twin: no other change
        for lin, d in zip(ys, dbs):
            assert np.allclose(d, 20 * np.log10(lin + 1e-6), atol=1e-3)


def test_spectrum_readout_unit_follows_the_axis(line):
    """The cursor readout must not bake in 'dB' when the axis can change."""
    b = ShotBrowser(line, _state(line))
    cbs = b.wt.spec_fig.js_event_callbacks["mousemove"]
    code = "".join(cb.code for cb in cbs)
    assert "axis_label.match" in code               # unit read at runtime
    assert '" dB"' not in code                      # ... not hardcoded
    assert any(cb.args.get("ax") is b.wt.spec_fig.yaxis[0] for cb in cbs)


def test_fk_is_the_plain_2d_dft_too(line):
    b = ShotBrowser(line, _state(line))
    _draw_rect(b.wt, x=12, y=0.1, w=16, h=0.12)
    b.wt.compute_fk()
    img = np.asarray(b.wt.cds_fk.data["image"][0])
    assert np.isfinite(img).all() and b.wt.fk_fig.visible
    assert img.max() <= 0.01 and img.min() >= -121   # peak-normalized dB
    assert not hasattr(b.wt, "_taper")               # no window function


# ---------------- keyboard shortcut ----------------
def _press(ws, key):
    """Simulate one key press arriving from the browser."""
    ws._counter = getattr(ws, "_counter", 0) + 1
    ws._keychan.value = f"{key}|{ws._counter}"


def _claimed(ws):
    return ws._keys_js.args["keys"]


def test_only_the_arrow_keys_are_claimed(line):
    """Claiming a key means swallowing it, so claim only the one shortcut."""
    ws = Workspace(line, keys=True)
    assert sorted(_claimed(ws)) == ["ArrowLeft", "ArrowRight"]


def test_a_shotless_view_binds_nothing(line):
    """No shot axis, no shortcut -- and no listener eating the arrow keys."""
    flat = Workspace(gv.from_array(np.zeros((8, 40), "f4"), dt=0.002), keys=True)
    assert flat._keychan is None and flat._keys_js is None
    assert not any(isinstance(i, TextInput) for i in flat.sidebar())
    flat.bind_keys()                                     # no-op, no crash


def test_the_sheet_only_advertises_the_key_where_it_works(line):
    """The gesture sheet is shared, the shortcut is not."""
    ws = Workspace(line, keys=True)
    assert "next shot" in ws.browser._help[1].object

    flat = Workspace(gv.from_array(np.zeros((8, 40), "f4"), dt=0.002))
    assert "next shot" not in flat._help[1].object
    assert "Toolbar gestures" in flat._help[1].object      # the rest stays


def test_arrow_keys_step_shots_and_clamp(line):
    ws = Workspace(line, keys=True)
    _press(ws, "ArrowRight")
    assert ws.browser.ishot == 1
    for _ in range(line.nshot + 3):                      # walk off the end
        _press(ws, "ArrowRight")
    assert ws.browser.ishot == line.nshot - 1
    assert ws.browser.w_jump.value == line.nshot - 1     # mirror widget follows
    assert ws.map._cds_asrc.data["x"][0] == line.geometry.src[-1, 0]
    for _ in range(line.nshot + 3):                      # ... and off the start
        _press(ws, "ArrowLeft")
    assert ws.browser.ishot == 0


def test_the_display_state_is_left_to_the_widgets(line):
    """w / p / g / [ / ] / f / 1-9 / ? are gone: set-once controls keep
    their widgets, and the keys stay with the browser."""
    ws = Workspace(line, keys=True)
    snap = lambda: (ws._w_disp.value, ws.state.get("flip"),
                    ws.state.get("gain"), ws.state.get("perc"),
                    ws.tabs.active)
    before = snap()
    for k in ("w", "p", "g", "[", "]", "f", "1", "2", "?", "Home", "End",
              "shift+ArrowRight", "z", "Escape", "", "Backspace"):
        _press(ws, k)
    assert snap() == before
    assert ws.browser.ishot == 0


def test_key_listener_leaves_typing_alone(line):
    """Panel widgets live in shadow roots, where e.target is the host element:
    the listener has to look through composedPath or an arrow key in
    'go to shot' would move the caret and the shot at once."""
    ws = Workspace(line, keys=True)
    js = ws._keys_js.code
    assert "composedPath" in js
    assert "'INPUT'" in js and "'TEXTAREA'" in js and "isContentEditable" in js
    assert "e.ctrlKey || e.metaKey || e.altKey" in js    # browser shortcuts
    assert "e.repeat" in js                              # held keys throttled
    assert "keys.indexOf(e.key) < 0" in js               # unclaimed keys pass
    assert "shiftKey" not in js and "clickFullscreen" not in js


def test_keys_can_be_switched_off(line):
    ws = Workspace(line, keys=False)
    assert ws._keychan is None and ws._keys_js is None
    assert not any(isinstance(i, TextInput) for i in ws.sidebar())
    ws.bind_keys()                                       # no-op, no crash


def test_key_channel_is_a_plain_bokeh_model(line):
    """A custom model the browser cannot resolve takes the whole document
    down with it, so the shortcut rides on stock models only."""
    ws = Workspace(line, keys=True)
    assert isinstance(ws._keychan, TextInput) and not ws._keychan.visible
    assert any(i is ws._keychan for i in ws.sidebar())
    assert isinstance(ws.panel(), pn.viewable.Viewable)
    assert not any(type(m).__module__.startswith("panel.")
                   for m in (ws._keychan, ws._keys_js))


def test_keys_bind_to_the_session_document(line):
    """The listener is page-global, so it hangs off the document, once."""
    from bokeh.document import Document
    from bokeh.events import DocumentReady
    from panel.io.state import set_curdoc
    ws = Workspace(line, keys=True)
    doc = Document()
    with set_curdoc(doc):
        ws.bind_keys()
        ws.bind_keys()                                   # idempotent
    cbs = doc.callbacks.js_event_callbacks.get(DocumentReady.event_name, [])
    assert [cb for cb in cbs if cb is ws._keys_js] == [ws._keys_js]
    assert ws._keys_js.args["chan"] is ws._keychan


# ---------------- URL state ----------------
@pytest.fixture
def session():
    """A served-session stand-in: pn.state.location only exists inside one."""
    from bokeh.document import Document
    from panel.io.state import set_curdoc
    with set_curdoc(Document()):
        yield pn.state.location


def test_url_restores_the_view(line, session):
    ws = Workspace(line)
    session.search = ("?shot=4&cmap=gray&display=wiggle&clip=95.0&flip=True"
                      "&filter=band-pass&f3=40.0&f4=50.0&gain=AGC&tab=1")
    ws.sync_location()
    assert ws.browser.ishot == 4
    assert ws._w_cmap.value == "gray" and ws._w_disp.value == "wiggle"
    assert ws.state["perc"] == 95.0          # applied, not just shown
    assert ws.state["flip"] is True
    assert ws.state["filter"] == {"f1": 5.0, "f2": 10.0, "f3": 40.0, "f4": 50.0}
    assert ws.state["gain"][0] == "agc"
    assert ws.tabs.active == 1


def test_changes_write_themselves_back_into_the_url(line, session):
    ws = Workspace(line)
    ws.sync_location()
    ws.browser.set_shot(3)
    ws._w_cmap.value = "petrel"
    assert "shot=3" in session.search and "cmap=petrel" in session.search
    ws.sync_location()                       # idempotent: no double watchers
    ws.browser.set_shot(2)
    assert session.search.count("shot=") == 1


def test_a_stale_url_cannot_break_the_view(line, session, capsys):
    ws = Workspace(line)
    cmap, clim = ws._w_cmap.value, ws.state["clim"]
    session.search = "?cmap=nosuchmap&clip=abc"
    ws.sync_location()
    assert ws._w_cmap.value == cmap          # rolled back, not left dangling
    assert ws.state["clim"] == clim
    assert "nosuchmap" in capsys.readouterr().out


def test_no_url_sync_outside_a_session(line):
    """In a notebook pn.state.location is the notebook's own URL."""
    assert pn.state.location is None
    ws = Workspace(line)
    ws.sync_location()
    assert ws._synced is False


def test_keys_add_no_exotic_models(line):
    """This is the bug that shipped once: a model type the browser cannot
    resolve does not break the shortcut, it fails the whole document and
    leaves a blank page. Turning keys on may only add stock bokeh widgets."""
    from bokeh.document import Document

    def model_types(keys):
        root = Workspace(line, keys=keys).panel().get_root(Document())
        return {type(m).__name__ for m in root.references()}

    assert model_types(True) - model_types(False) <= {"TextInput"}
