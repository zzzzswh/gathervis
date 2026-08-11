"""Panel/Bokeh viewers.

One rendering primitive (ImagePane: a quantized uint8 bokeh image) is reused by
every view:

  * view_gather      -- a single (trace, time) panel
  * SliceView        -- 3 orthogonal slices of a volume (cigvis-style)
  * ShotBrowser      -- slider over the shot axis (2-D panel or per-shot SliceView)
  * acquisition_app  -- layout map (tap a source) linked to a ShotBrowser
  * show()           -- dispatcher; serves in a browser (SSH port-forward friendly)
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
    """Slider over the shot axis; each shot is a 2-D panel or a 3-D SliceView."""

    def __init__(self, g: Gathers, cmap="seismic", perc=98.0):
        _ensure_ext()
        self.g = g
        self.state = {"sample": g.sample(), "clim": None}
        self.state["clim"] = robust_clim(self.state["sample"], perc)
        self._per_shot_3d = g.data.ndim == 4

        if self._per_shot_3d:
            self.sv = SliceView(g.shot(0), names=g.axes[1:3], dt=g.dt, t0=g.t0,
                                cmap=cmap, clim=self.state["clim"])
            panes = self.sv.panes
        else:
            self.pane = ImagePane(xlabel=g.axes[1], cmap=cmap)
            panes = [self.pane]

        self.w_shot = pn.widgets.IntSlider(name="shot", start=0, end=g.nshot - 1,
                                           value=0, sizing_mode="stretch_width")
        self.w_shot.param.watch(lambda e: self.redraw(), "value")
        self.controls = _display_controls(self.state, panes, self.redraw, cmap, perc)
        self._on_shot_change = None  # hook for the acquisition map
        self.redraw()

    @property
    def ishot(self) -> int:
        return int(self.w_shot.value)

    def set_shot(self, i: int):
        self.w_shot.value = int(i)  # triggers redraw via watcher

    def redraw(self):
        if self._per_shot_3d:
            self.sv.clim = self.state["clim"]
            self.sv.set_volume(self.g.shot(self.ishot))
        else:
            self.pane.update(self.g.shot(self.ishot), self.state["clim"],
                             y0=self.g.t0, dy=self.g.dt)
        if self._on_shot_change is not None:
            self._on_shot_change(self.ishot)

    def panel(self):
        body = self.sv.panel() if self._per_shot_3d else self.pane.figure
        return pn.Column(pn.Row(self.w_shot, self.controls), body,
                         sizing_mode="stretch_width")


def acquisition_app(g: Gathers, cmap="seismic", perc=98.0):
    """Layout map (tap a source point) linked to a shot browser."""
    _ensure_ext()
    geo = g.geometry
    browser = ShotBrowser(g, cmap=cmap, perc=perc)

    fig = figure(height=520, sizing_mode="stretch_width", match_aspect=True,
                 x_axis_label="x", y_axis_label="y", title="acquisition layout",
                 tools="pan,wheel_zoom,box_zoom,reset", active_scroll="wheel_zoom")
    fig.toolbar.logo = None
    rec_all = geo.rec.reshape(-1, 3)
    fig.scatter(rec_all[:, 0], rec_all[:, 1], size=3, color="#9aa0a6",
                alpha=0.6, legend_label="receivers")
    cds_src = ColumnDataSource(dict(x=geo.src[:, 0], y=geo.src[:, 1]))
    r_src = fig.scatter("x", "y", source=cds_src, size=7, color="#d93025",
                        legend_label="sources",
                        nonselection_alpha=0.5, selection_color="#fbbc04")
    cds_arec = ColumnDataSource(dict(x=[], y=[]))
    cds_asrc = ColumnDataSource(dict(x=[], y=[]))
    fig.scatter("x", "y", source=cds_arec, size=4, color="#1a73e8")
    fig.scatter("x", "y", source=cds_asrc, size=14, color="#fbbc04", marker="star")
    fig.add_tools(TapTool(renderers=[r_src]))
    fig.legend.location = "top_right"

    def _highlight(i):
        rec = geo.rec_for(i)
        cds_arec.data = dict(x=rec[:, 0], y=rec[:, 1])
        cds_asrc.data = dict(x=[geo.src[i, 0]], y=[geo.src[i, 1]])

    def _on_tap(attr, old, new):
        if new:
            browser.set_shot(new[0])

    cds_src.selected.on_change("indices", _on_tap)
    browser._on_shot_change = _highlight
    _highlight(0)

    return pn.Row(pn.Column(fig, sizing_mode="stretch_width"),
                  browser.panel(), sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
def _build(obj, view, cmap, perc, title):
    if isinstance(obj, Gathers):
        g = obj
    else:
        raise TypeError("internal: _build expects Gathers")

    if g.geometry is not None and view != "slices":
        return acquisition_app(g, cmap=cmap, perc=perc)

    d = g.data
    if d.ndim == 2:
        return view_gather(d, dt=g.dt, t0=g.t0, cmap=cmap, perc=perc, title=title)
    if d.ndim == 3:
        browse = ("shot" in g.axes) if view is None else (view == "browse")
        if browse:
            return ShotBrowser(g, cmap=cmap, perc=perc).panel()
        names = [a for a in g.axes[:-1]]
        return SliceView(d, names=tuple(names), dt=g.dt, t0=g.t0,
                         cmap=cmap, perc=perc).panel()
    if d.ndim == 4:
        return ShotBrowser(g, cmap=cmap, perc=perc).panel()
    raise ValueError(f"cannot build a view for {d.ndim}-D data")


def show(obj, axes=None, view=None, dt=1.0, t0=0.0, src=None, rec=None,
         shape=None, dtype="float32", cmap="seismic", perc=98.0,
         port=None, address="127.0.0.1", title="gathervis"):
    """Open a viewer for ``obj``.

    Parameters
    ----------
    obj : ndarray | Gathers | path
        2-D (trace, time) gather; 3-D data (default semantics (shot, rec, time),
        browsed shot by shot); 4-D (shot, recy, recx, time); a ``Gathers``
        dataset; or a path to a raw binary file (then ``shape`` is required).
    view : None | 'browse' | 'slices'
        Overrides the default viewing mode of 3-D data.
    port : int, optional
        If given, serve blocking on ``address:port`` (use SSH port forwarding
        on a remote server). If None, return the Panel app (notebook-friendly).
    """
    if view not in (None, "browse", "slices"):
        raise ValueError("view must be None, 'browse' or 'slices'")
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        if shape is None:
            raise ValueError("shape= is required when opening a raw binary file")
        obj = from_file(obj, shape, dtype=dtype, axes=axes, dt=dt, t0=t0,
                        src=src, rec=rec)
    elif not isinstance(obj, Gathers):
        obj = from_array(obj, src=src, rec=rec, axes=axes, dt=dt, t0=t0)

    app = _build(obj, view, cmap, perc, title)
    if port is None:
        return app
    pn.serve(app, port=int(port), address=address, show=False, title=title,
             websocket_origin="*")
