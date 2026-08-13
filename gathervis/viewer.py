"""Panel/Bokeh viewers.

One rendering primitive (ImagePane: a quantized uint8 bokeh image) is reused by
every view:

  * view_gather      -- a single (trace, time) panel
  * SliceView        -- 3 orthogonal slices of a volume (cigvis-style)
  * ShotBrowser      -- slider over the shot axis (2-D panel or per-shot SliceView)
  * LayoutMap        -- acquisition map with tap-to-pick-a-shot
  * Workspace        -- sidebar (dataset info + display controls) + view tabs
  * show()           -- dispatcher; serves in a browser (SSH port-forward friendly)

All components are plain objects exposing bokeh figures / panel layouts, so
power users can also compose their own dashboards with pn.Row / pn.Column.
"""
from __future__ import annotations

import numpy as np
import panel as pn
from bokeh.models import ColumnDataSource, LinearColorMapper, TapTool
from bokeh.plotting import figure

from .core import Gathers, from_array, from_file
from .process import robust_clim, decimate, quantize

__all__ = ["show"]

MAX_PX = (1600, 1600)  # decimation budget per displayed panel
_CMAPS = ("seismic", "gray")

_ext_done = False


def _ensure_ext():
    global _ext_done
    if not _ext_done:
        pn.extension(design="material")
        _ext_done = True


# ---------------------------------------------------------------------------
# palettes (no matplotlib dependency)
# ---------------------------------------------------------------------------
def _palette(name: str):
    t = np.linspace(0.0, 1.0, 256)
    if name == "gray":
        rgb = np.stack([t, t, t], axis=1)
    elif name == "seismic":  # dark blue -> white -> dark red
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        anchors = np.array([[0.0, 0.0, 0.45],
                            [0.1, 0.3, 1.0],
                            [1.0, 1.0, 1.0],
                            [1.0, 0.25, 0.1],
                            [0.45, 0.0, 0.0]])
        rgb = np.stack([np.interp(t, pos, anchors[:, c]) for c in range(3)], axis=1)
    else:
        raise ValueError(f"unknown cmap {name!r}; available: {_CMAPS}")
    return ["#%02x%02x%02x" % tuple(c) for c in (rgb * 255 + 0.5).astype(int)]


# ---------------------------------------------------------------------------
# ImagePane: the single rendering primitive
# ---------------------------------------------------------------------------
class ImagePane:
    """A bokeh image of a 2-D array with rows on x and columns on y.

    ``flip_y=True`` gives the seismic convention (time increasing downward).
    """

    def __init__(self, xlabel="trace", ylabel="time (s)", flip_y=True,
                 cmap="seismic", height=520, match_aspect=False):
        self.mapper = LinearColorMapper(palette=_palette(cmap), low=0, high=255)
        self.cds = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
        fig = figure(height=height, sizing_mode="stretch_width",
                     x_axis_label=xlabel, y_axis_label=ylabel,
                     match_aspect=match_aspect,
                     tools="pan,wheel_zoom,box_zoom,reset,save",
                     active_scroll="wheel_zoom")
        fig.image(image="image", x="x", y="y", dw="dw", dh="dh",
                  source=self.cds, color_mapper=self.mapper)
        fig.y_range.flipped = bool(flip_y)
        fig.x_range.range_padding = fig.y_range.range_padding = 0
        fig.toolbar.logo = None
        self.figure = fig

    def set_cmap(self, name: str):
        self.mapper.palette = _palette(name)

    def update(self, arr2d, clim, x0=0.0, dx=1.0, y0=0.0, dy=1.0):
        """arr2d: (nx, ny) with axis 0 on x. Decimates, quantizes, ships uint8."""
        nx, ny = arr2d.shape
        img = quantize(decimate(arr2d, MAX_PX), clim)
        self.cds.data = dict(image=[img.T], x=[x0], y=[y0],
                             dw=[nx * dx], dh=[ny * dy])


def _display_controls(state, panes, redraw, cmap, perc):
    """Shared cmap + clim-percentile widgets. ``state['clim']`` is read by redraw()."""
    w_cmap = pn.widgets.Select(name="colormap", options=list(_CMAPS), value=cmap,
                               width=130)
    w_perc = pn.widgets.FloatSlider(name="clip percentile", start=80.0, end=100.0,
                                    step=0.5, value=perc, width=180)

    def _on_cmap(event):
        for p in panes:
            p.set_cmap(event.new)

    def _on_perc(event):
        state["clim"] = robust_clim(state["sample"], event.new)
        redraw()

    w_cmap.param.watch(_on_cmap, "value")
    w_perc.param.watch(_on_perc, "value")
    return pn.Row(w_cmap, w_perc)


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------
def view_gather(arr2d, dt=1.0, t0=0.0, cmap="seismic", perc=98.0, title=""):
    """Single (trace, time) panel."""
    _ensure_ext()
    state = {"sample": np.asarray(arr2d), "clim": None}
    state["clim"] = robust_clim(state["sample"], perc)
    pane = ImagePane(cmap=cmap)

    def redraw():
        pane.update(arr2d, state["clim"], y0=t0, dy=dt)

    redraw()
    header = pn.pane.Markdown(f"### {title or 'gather'}")
    return pn.Column(header, _display_controls(state, [pane], redraw, cmap, perc),
                     pane.figure, sizing_mode="stretch_width")


class SliceView:
    """Three orthogonal slices of a (n0, n1, time) volume with position sliders."""

    def __init__(self, vol, names=("recy", "recx"), dt=1.0, t0=0.0,
                 cmap="seismic", perc=98.0, clim=None):
        _ensure_ext()
        self.vol, self.names, self.dt, self.t0 = vol, names, dt, t0
        self.clim = clim or robust_clim(np.asarray(vol), perc)
        self.p0 = ImagePane(xlabel=names[1], cmap=cmap)              # vol[i]
        self.p1 = ImagePane(xlabel=names[0], cmap=cmap)              # vol[:, j]
        self.p2 = ImagePane(xlabel=names[0], ylabel=names[1],        # vol[:, :, k]
                            flip_y=False, cmap=cmap, match_aspect=True)
        self.panes = [self.p0, self.p1, self.p2]

        n0, n1, nt = vol.shape
        self.w0 = pn.widgets.IntSlider(name=names[0], start=0, end=n0 - 1,
                                       value=n0 // 2)
        self.w1 = pn.widgets.IntSlider(name=names[1], start=0, end=n1 - 1,
                                       value=n1 // 2)
        self.w2 = pn.widgets.IntSlider(name="time sample", start=0, end=nt - 1,
                                       value=nt // 2)
        for w, cb in ((self.w0, self._d0), (self.w1, self._d1), (self.w2, self._d2)):
            w.param.watch(lambda e, cb=cb: cb(), "value")
        self.redraw()

    # per-slice redraws: only touch the bytes of the requested slice
    def _d0(self):
        self.p0.update(self.vol[self.w0.value], self.clim, y0=self.t0, dy=self.dt)

    def _d1(self):
        self.p1.update(self.vol[:, self.w1.value], self.clim, y0=self.t0, dy=self.dt)

    def _d2(self):
        self.p2.update(self.vol[:, :, self.w2.value], self.clim)

    def redraw(self):
        self._d0(); self._d1(); self._d2()

    def set_volume(self, vol):
        """Swap the volume (per-shot browsing of 3-D shot gathers)."""
        self.vol = vol
        n0, n1, nt = vol.shape
        for w, n in ((self.w0, n0), (self.w1, n1), (self.w2, nt)):
            w.end = n - 1
            w.value = min(w.value, n - 1)
        self.redraw()

    @property
    def sliders(self):
        return pn.Row(self.w0, self.w1, self.w2)

    def panel(self):
        return pn.Column(self.sliders,
                         pn.Row(self.p0.figure, self.p1.figure),
                         self.p2.figure, sizing_mode="stretch_width")


class ShotBrowser:
    """Slider over the shot axis; each shot is a 2-D panel or a 3-D SliceView.

    ``state`` is a shared dict holding 'clim' (and 'sample'), owned by the
    Workspace so all views stay consistent.
    """

    def __init__(self, g: Gathers, state: dict, cmap="seismic"):
        _ensure_ext()
        self.g, self.state = g, state
        self._per_shot_3d = g.data.ndim == 4
        if self._per_shot_3d:
            self.sv = SliceView(g.shot(0), names=g.axes[1:3], dt=g.dt, t0=g.t0,
                                cmap=cmap, clim=state["clim"])
            self.panes = self.sv.panes
        else:
            self.pane = ImagePane(xlabel=g.axes[1], cmap=cmap)
            self.panes = [self.pane]

        self.w_shot = pn.widgets.IntSlider(name="shot", start=0, end=g.nshot - 1,
                                           value=0, sizing_mode="stretch_width")
        self.w_shot.param.watch(lambda e: self.redraw(), "value")
        self.on_shot_change = None      # hook: called with the new shot index
        self.redraw()

    @property
    def ishot(self) -> int:
        return int(self.w_shot.value)

    def set_shot(self, i: int):
        self.w_shot.value = int(i)      # triggers redraw via watcher

    def redraw(self):
        if self._per_shot_3d:
            self.sv.clim = self.state["clim"]
            self.sv.set_volume(self.g.shot(self.ishot))
        else:
            self.pane.update(self.g.shot(self.ishot), self.state["clim"],
                             y0=self.g.t0, dy=self.g.dt)
        if self.on_shot_change is not None:
            self.on_shot_change(self.ishot)

    def panel(self):
        body = self.sv.panel() if self._per_shot_3d else self.pane.figure
        return pn.Column(self.w_shot, body, sizing_mode="stretch_width")


class LayoutMap:
    """Acquisition map: sources are tappable; active shot + receivers highlighted."""

    def __init__(self, geo):
        _ensure_ext()
        self.geo = geo
        self.on_pick = None             # hook: called with the picked shot index
        fig = figure(height=420, sizing_mode="stretch_width", match_aspect=True,
                     x_axis_label="x", y_axis_label="y",
                     tools="pan,wheel_zoom,box_zoom,reset,tap",
                     active_scroll="wheel_zoom")
        fig.toolbar.logo = None
        rec_all = geo.rec.reshape(-1, 3)
        fig.scatter(rec_all[:, 0], rec_all[:, 1], size=3, color="#9aa0a6",
                    alpha=0.6, legend_label="receivers")
        self._cds_src = ColumnDataSource(dict(x=geo.src[:, 0], y=geo.src[:, 1]))
        r_src = fig.scatter("x", "y", source=self._cds_src, size=7,
                            color="#d93025", legend_label="sources",
                            nonselection_alpha=0.5, selection_color="#fbbc04")
        self._cds_arec = ColumnDataSource(dict(x=[], y=[]))
        self._cds_asrc = ColumnDataSource(dict(x=[], y=[]))
        fig.scatter("x", "y", source=self._cds_arec, size=4, color="#1a73e8")
        fig.scatter("x", "y", source=self._cds_asrc, size=14, color="#fbbc04",
                    marker="star")
        for t in fig.select(TapTool):
            t.renderers = [r_src]
        fig.legend.location = "top_right"
        self._cds_src.selected.on_change("indices", self._on_tap)
        self.figure = fig

    def _on_tap(self, attr, old, new):
        if new and self.on_pick is not None:
            self.on_pick(new[0])

    def set_active(self, i: int):
        rec = self.geo.rec_for(i)
        self._cds_arec.data = dict(x=rec[:, 0], y=rec[:, 1])
        self._cds_asrc.data = dict(x=[self.geo.src[i, 0]], y=[self.geo.src[i, 1]])


def _survey_kind(g: Gathers) -> str:
    if "shot" in g.axes:
        return ("3-D seismic (per-shot 3-D gathers)" if g.data.ndim == 4
                else "2-D seismic line")
    return "single gather" if g.data.ndim == 2 else "volume"


def _info_md(g: Gathers) -> str:
    n_bytes = int(np.prod(g.shape)) * np.dtype(g.data.dtype).itemsize
    size = (f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e9
            else f"{n_bytes / 1e6:.1f} MB")
    lines = [f"### {g.name or 'dataset'}",
             f"**survey**&nbsp; {_survey_kind(g)}",
             f"**shape**&nbsp; {tuple(g.shape)}",
             f"**axes**&nbsp; ({', '.join(g.axes)})",
             f"**dt**&nbsp; {g.dt * 1e3:g} ms &nbsp;·&nbsp; "
             f"record {g.t0 + g.dt * (g.nt - 1):.3f} s",
             f"**size**&nbsp; {size} {np.dtype(g.data.dtype).name}"]
    if g.geometry is not None:
        geo = g.geometry
        kind = "shared spread" if geo.shared else "per-shot spread"
        lines.append(f"**geometry**&nbsp; {geo.ns} sources · "
                     f"{geo.nr} receivers ({kind})")
    return "\n\n".join(lines)


class Workspace:
    """Sidebar (dataset info + display controls) + tabbed views.

    Tabs adapt to the data: shot browsing (with linked layout map when
    geometry is present) and/or whole-volume slicing.
    """

    def __init__(self, g: Gathers, cmap="seismic", perc=98.0, view=None):
        _ensure_ext()
        self.g = g
        self.state = {"sample": g.sample(), "clim": None}
        self.state["clim"] = robust_clim(self.state["sample"], perc)
        self._panes, self._redraws, tabs = [], [], []
        self.map = None

        # -- shot-browsing tab -------------------------------------------
        if "shot" in g.axes:
            self.browser = ShotBrowser(g, self.state, cmap=cmap)
            self._panes += self.browser.panes
            self._redraws.append(self.browser.redraw)
            main = self.browser.panel()
            if g.geometry is not None:
                self.map = LayoutMap(g.geometry)
                self.map.on_pick = self.browser.set_shot
                self.browser.on_shot_change = self.map.set_active
                self.map.set_active(0)
                self._map_col = pn.Column(self.map.figure, width=430)
                main = pn.Row(self._map_col, main, sizing_mode="stretch_width")
            label = "Shot gathers" + (" (3-D)" if g.data.ndim == 4 else "")
            tabs.append((label, main))

        # -- volume-slices tab (3-D data, with or without a shot axis) ----
        if g.data.ndim == 3:
            self.slices = SliceView(g.data, names=tuple(g.axes[:-1]), dt=g.dt,
                                    t0=g.t0, cmap=cmap, clim=self.state["clim"])
            self._panes += self.slices.panes
            self._redraws.append(self._redraw_slices)
            tabs.append(("Volume slices", self.slices.panel()))

        # -- single 2-D gather ---------------------------------------------
        if g.data.ndim == 2:
            self._pane2d = ImagePane(xlabel=g.axes[0], cmap=cmap)
            self._panes.append(self._pane2d)
            self._redraws.append(self._draw2d)
            self._draw2d()
            tabs.append(("Gather", self._pane2d.figure))

        self.tabs = pn.Tabs(*tabs, sizing_mode="stretch_width")
        if view == "slices":
            for i, (label, _) in enumerate(tabs):
                if label == "Volume slices":
                    self.tabs.active = i
        self._controls = self._make_controls(cmap, perc)

    def _draw2d(self):
        self._pane2d.update(self.g.data, self.state["clim"],
                            y0=self.g.t0, dy=self.g.dt)

    def _redraw_slices(self):
        self.slices.clim = self.state["clim"]
        self.slices.redraw()

    def _make_controls(self, cmap, perc):
        w_cmap = pn.widgets.Select(name="colormap", options=list(_CMAPS),
                                   value=cmap)
        w_perc = pn.widgets.FloatSlider(name="clip percentile", start=80.0,
                                        end=100.0, step=0.5, value=perc)
        w_cmap.param.watch(
            lambda e: [p.set_cmap(e.new) for p in self._panes], "value")

        def _on_perc(event):
            self.state["clim"] = robust_clim(self.state["sample"], event.new)
            for r in self._redraws:
                r()

        w_perc.param.watch(_on_perc, "value")
        widgets = [w_cmap, w_perc]
        if self.map is not None:
            w_map = pn.widgets.Checkbox(name="show layout map", value=True)
            w_map.param.watch(
                lambda e: setattr(self._map_col, "visible", e.new), "value")
            widgets.append(w_map)
        return widgets

    def sidebar(self):
        return [pn.pane.Markdown(_info_md(self.g)), *self._controls]

    def panel(self):
        """Plain layout (notebook-friendly)."""
        return pn.Row(pn.Column(*self.sidebar(), width=280),
                      self.tabs, sizing_mode="stretch_width")

    def template(self, title="gathervis"):
        """Served page with a proper header + sidebar."""
        name = self.g.name
        return pn.template.FastListTemplate(
            title=f"{title} — {name}" if name else title,
            sidebar=self.sidebar(), main=[self.tabs],
            accent_base_color="#1a5276", header_background="#1a5276",
            sidebar_width=300)


# ---------------------------------------------------------------------------
# serving helpers + dispatcher
# ---------------------------------------------------------------------------
def _resolve_port(port, address) -> int:
    """Return a usable port. If the requested one is busy (common on shared
    servers), silently fall back to an OS-assigned free port. ``port=0``
    always means 'pick a free one for me'."""
    import socket
    want = int(port)
    s = socket.socket()
    try:
        s.bind((address, want))
    except OSError:
        s.close()
        s = socket.socket()
        s.bind((address, 0))
        print(f"[gathervis] port {want} is busy -> using a free one instead")
    p = s.getsockname()[1]
    s.close()
    return p


def _banner(port):
    """Print how to reach the viewer, adapted to where we are running:
    local machine / VS Code Remote (auto-forwards) / bare SSH session."""
    import os
    url = f"http://localhost:{port}"
    in_ssh = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))
    in_vscode = (os.environ.get("TERM_PROGRAM") == "vscode"
                 or any(k.startswith("VSCODE_") for k in os.environ))
    if not in_ssh or in_vscode:
        print(f"[gathervis] open  {url}")
        if in_ssh:
            print("[gathervis] VS Code detected: the port is auto-forwarded "
                  "(see the PORTS panel); just open the URL above.")
        return
    # deliberately NOT auto-filling the real user@host: terminal output gets
    # pasted into issues/screenshots, and the self-reported hostname is often
    # not even resolvable from the user's machine anyway.
    print(f"""[gathervis] open  {url}
[gathervis] NOTE: you are in a plain SSH session. The browser on your OWN
[gathervis] machine cannot reach this server until the port is forwarded:
[gathervis]   1. on your LOCAL machine, open a new terminal (keep it open)
[gathervis]   2. run:   ssh -L {port}:127.0.0.1:{port} <user>@<this-server>
[gathervis]   3. open:  {url}
[gathervis] one-time fix: add  LocalForward {port} 127.0.0.1:{port}  to this
[gathervis] host's entry in your local ~/.ssh/config.
[gathervis] (tip: VS Code Remote-SSH / Jupyter forward ports automatically)""")


def show(obj, axes=None, view=None, dt=1.0, t0=0.0, src=None, rec=None,
         shape=None, dtype="float32", name=None, cmap="seismic", perc=98.0,
         port=None, address="127.0.0.1", title="gathervis", verbose=True):
    """Open a viewer for ``obj``.

    Parameters
    ----------
    obj : ndarray | Gathers | path
        2-D (trace, time) gather; 3-D data (default semantics (shot, rec, time),
        browsed shot by shot); 4-D (shot, recy, recx, time); a ``Gathers``
        dataset; or a path to a raw binary file (then ``shape`` is required).
    view : None | 'browse' | 'slices'
        Selects the initially active tab for 3-D data (both remain available).
    name : str, optional
        Free-text description shown in the info card, e.g.
        "2-D acoustic modelling, 3-layer 2-D velocity model".
    port : int, optional
        If given, serve blocking on ``address:port`` (use SSH port forwarding
        on a remote server). If the port is busy a free one is picked
        automatically; ``port=0`` always auto-picks. If None, return the
        Panel app (notebook-friendly).
    """
    import time
    t_start = time.perf_counter()
    if view not in (None, "browse", "slices"):
        raise ValueError("view must be None, 'browse' or 'slices'")
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        if shape is None:
            raise ValueError("shape= is required when opening a raw binary file")
        obj = from_file(obj, shape, dtype=dtype, axes=axes, dt=dt, t0=t0,
                        src=src, rec=rec, name=name)
    elif not isinstance(obj, Gathers):
        obj = from_array(obj, src=src, rec=rec, axes=axes, dt=dt, t0=t0,
                         name=name)
    elif name is not None:
        obj.name = name
    t_data = time.perf_counter()

    ws = Workspace(obj, cmap=cmap, perc=perc, view=view)
    t_app = time.perf_counter()
    if verbose:
        print(f"[gathervis] dataset {tuple(obj.shape)} ready in "
              f"{t_data - t_start:.2f}s | app built in {t_app - t_data:.2f}s")

    if port is None:
        return ws.panel()
    port = _resolve_port(port, address)
    if verbose:
        _banner(port)
    pn.serve(ws.template(title), port=port, address=address, show=False,
             title=title, websocket_origin="*")
