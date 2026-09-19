"""Panel/Bokeh viewers.

Two rendering primitives, both fed by the same decimate -> uint8 pipeline:
ImagePane (a quantized bokeh image) for 2-D panels, and VolumeView3D (plotly
WebGL slice planes in a cuboid) for volumes:

  * view_gather      -- a single (trace, time) panel
  * VolumeView3D     -- a 3-D cuboid with three slice planes (cigvis-style)
  * ShotBrowser      -- slider over the shot axis (2-D panel or per-shot cuboid)
  * LayoutMap        -- acquisition map with tap-to-pick-a-shot
  * Workspace        -- sidebar (dataset info + display controls) + view tabs
  * show()           -- dispatcher; serves in a browser (SSH port-forward friendly)

All components are plain objects exposing bokeh figures / panel layouts, so
power users can also compose their own dashboards with pn.Row / pn.Column.
"""
from __future__ import annotations

import itertools

import numpy as np
import panel as pn
from bokeh.models import (BoxEditTool, ColumnDataSource, CrosshairTool,
                          CustomJS, Div, LinearColorMapper, PointDrawTool,
                          PolyDrawTool, TapTool)
from bokeh.plotting import figure

from .core import Gathers, from_array, from_file
from .process import (robust_clim, decimate, quantize, bandpass,
                      agc, trace_balance, pick_first_breaks)
from . import render as _render
from .render import (CMAPS as _CMAPS, palette_hex as _palette, png_bytes,
                     raster_size, render_gather)

__all__ = ["show"]

MAX_PX = (1600, 1600)  # decimation budget per displayed 2-D panel
MAX_PX_3D = (400, 400)  # per-slice budget in the 3-D view (WebGL vertex count)
MAX_WIGGLE = (96, 1500)  # wiggle budget: (traces drawn, samples per trace)

_ext_done = False


_PLOT_CSS = """
.gv-plot:fullscreen {
  background: var(--panel-surface-color, #ffffff);
  padding: 10px 14px;
  box-sizing: border-box;
}
.gv-plot:fullscreen .bk-Figure { height: 100% !important; }
"""

# Panel's stock card.css gives every card a heavy drop shadow, an outline, a
# 1.4em bold title and a shadow that pops in on header hover. Four cards of
# that in a sidebar read as four floating dialogs. These stylesheets (applied
# per component, since Panel renders them into shadow roots where global CSS
# cannot reach) flatten them into one quiet, consistent block.
_INK, _MUTED, _LINE = "#3c4043", "#5f6368", "#dadce0"

_CARD_CSS = f"""
:host(.card) {{
  box-shadow: none;
  outline: none;
  border: 1px solid {_LINE};
  border-radius: 8px;
  overflow: hidden;
}}
.card-header {{
  background-color: #f6f7f9;
  border-radius: 0;
  padding: 1px 10px;
  min-height: 34px;
}}
.card-header:hover {{ box-shadow: none; background-color: #eceef1; }}
.card-header:not(:hover) {{ box-shadow: none; }}
.card-title {{ font-size: 1em; font-weight: 600; color: {_INK}; }}
"""

# The native file picker is the one control we cannot rebuild; at least give
# it the same type size, border and radius as the buttons above it.
_FILE_CSS = f"""
input[type="file"] {{
  font: inherit;
  font-size: 12px;
  width: 100%;
  color: {_MUTED};
}}
input[type="file"]::file-selector-button {{
  font: inherit;
  font-size: 12px;
  padding: 5px 12px;
  margin-right: 8px;
  border: 1px solid {_LINE};
  border-radius: 6px;
  background: #ffffff;
  color: {_INK};
  cursor: pointer;
}}
input[type="file"]::file-selector-button:hover {{ background: #f1f3f4; }}
"""

# One shape for every control inside a tool card: full width, same rhythm.
# Widths used to be hand-picked per widget (205 here, 97 there), which is
# what made the cards look assembled from spare parts.
# width=None is explicit: several Panel widgets ship a 300 px default that
# would otherwise sit in the param even though stretch_width wins in layout.
_W = dict(sizing_mode="stretch_width", width=None, margin=(4, 0))
_WFILE = dict(_W, stylesheets=[_FILE_CSS])


def _ensure_ext():
    global _ext_done
    if not _ext_done:
        pn.extension("plotly", design="material", raw_css=[_PLOT_CSS])
        _ext_done = True


def _set_throttled(widget, value):
    """Programmatically write a slider's ``value_throttled`` (a constant param
    Panel normally only writes from JS on mouse release). Needed to drive
    release-applied widgets such as the clip-percentile slider from code."""
    from param.parameterized import edit_constant
    with edit_constant(widget):
        widget.value_throttled = value


# Palettes live in render.py, which needs the same colors as a uint8 lookup
# table for offscreen rasterizing; _palette / _CMAPS above are its hex-string
# face, kept under the old names so the rest of this module reads unchanged.
_DISPLAYS = ("density", "wiggle")   # variable density image vs. wiggle traces


# client-side cursor readout: trace / time / amplitude under the mouse.
# Runs entirely in the browser (zero server round trips); the amplitude is
# de-quantized from the displayed uint8 pixel via the lo/hi columns, i.e. it
# is exactly the value of the pixel you are looking at.
_READOUT_JS = """
const x = cb_obj.x, y = cb_obj.y;
if (x == null || y == null) { div.text = "&nbsp;"; return; }
const yfix = (yu == "s") ? 3 : 1;
let txt = xl + " " + x.toFixed(1) + " &nbsp;&nbsp; " + yn + " "
        + y.toFixed(yfix) + " " + yu;
const d = src.data;
if (d["image"] != null && d["image"].length > 0) {
  const img = d["image"][0];
  const nyr = img.shape[0], nxc = img.shape[1];
  const x0 = d["x"][0], y0 = d["y"][0], dw = d["dw"][0], dh = d["dh"][0];
  const j = Math.floor((x - x0) / dw * nxc);
  const i = Math.floor((y - y0) / dh * nyr);
  if (i >= 0 && i < nyr && j >= 0 && j < nxc) {
    const q = img[i * nxc + j];
    const lo = d["lo"][0], hi = d["hi"][0];
    const v = lo + (q / 255.0) * (hi - lo);
    txt += " &nbsp;&nbsp; amp " + v.toPrecision(3);
  }
}
div.text = txt;
"""


def _process_gather(arr, dt, state):
    """The display processing chain: filter -> gain.

    Returns (processed array, clim to display it with). With gain on, the raw
    global clim no longer matches the ~unit-amplitude output, so the clim is
    recomputed from the processed gather at the current clip percentile.
    """
    flt = state.get("filter")
    if flt:
        arr = bandpass(arr, dt, **flt)
    gain = state.get("gain")
    clim = state["clim"]
    if gain:
        if gain[0] == "agc":
            arr = agc(arr, dt, window=gain[1])
        else:
            arr = trace_balance(arr)
        clim = robust_clim(arr, state.get("perc", 98.0),
                           symmetric=state["symmetric"])
    if state.get("flip"):
        arr = -np.asarray(arr)          # SEG normal <-> reverse polarity
    return arr, clim


_GESTURES_HTML = """
<div style='font-size:12px;color:#3c4043;line-height:1.6;background:#f8f9fa;
            border:1px solid #dadce0;border-radius:8px;padding:10px 14px'>
<b>Toolbar gestures</b> &nbsp;<span style='color:#5f6368'>(click a toolbar
icon to activate its tool; click again to turn it off)</span><br>
<b>Box window</b> &mdash; SHIFT+drag to draw (or click, move, click) &middot;
drag to move &middot; tap to select, then BACKSPACE to delete<br>
<b>Polygon window</b> &mdash; click each vertex, double-click or ESC to
finish &middot; drag to move &middot; tap + BACKSPACE deletes<br>
<b>Pick</b> &mdash; tap to add &middot; drag to move &middot; tap + BACKSPACE
deletes one (or use the clear buttons)<br>
<b>Axes</b> &mdash; the x/y wheel-zoom tools scale one axis at a time &middot;
the crosshair icon toggles cursor lines
</div>"""


def _gesture_help():
    """A small '?' button toggling an inline gesture cheat-sheet."""
    body = pn.pane.HTML(_GESTURES_HTML, visible=False,
                        sizing_mode="stretch_width", margin=(4, 0))
    btn = pn.widgets.Button(
        name="? gestures", **_W,
        description="how to draw, move and delete with the toolbar tools "
                    "(SHIFT, BACKSPACE, ESC)")
    btn.on_click(lambda e: setattr(body, "visible", not body.visible))
    return btn, body


def _card_style(sidebar):
    """Geometry for a tool card, depending on where it is hosted.

    Below the figure the cards are fixed 240 px tiles flowing in a FlexBox.
    In the sidebar they become full-width stacked sections, and start
    collapsed so the sidebar stays scannable -- the display controls above
    them are the ones reached on every shot.
    """
    common = dict(stylesheets=[_CARD_CSS], header_color=_INK)
    if sidebar:
        return dict(common, sizing_mode="stretch_width", margin=(4, 0),
                    collapsed=True)
    return dict(common, width=248, margin=(6, 6), collapsed=False)


def _slug(name, default="gather", maxlen=40):
    """A dataset name reduced to a safe file-name stem (for downloads)."""
    import os
    import re
    stem = os.path.basename(str(name or "")).rsplit(".", 1)[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem[:maxlen] or default


def _hint(html):
    """The one small-print style: same size, color and inset everywhere."""
    return pn.pane.HTML(
        f"<div style='font-size:11px;color:{_MUTED};line-height:1.45'>"
        f"{html}</div>", margin=(4, 0, 0, 0), sizing_mode="stretch_width")


def _labeled(widget, caption):
    """A widget with a small caption above it (labels bare FileInputs)."""
    return pn.Column(_hint(caption), widget, margin=0,
                     sizing_mode="stretch_width")


def _coord_readout(fig, xn, xu, yn, yu, xdigits=1, ydigits=1, yaxis=None):
    """Client-side cursor readout (coordinates only) under any figure.

    ``yaxis``: take the y unit from that axis' label at runtime rather than
    baking ``yu`` in, for a plot whose y scale is switchable -- the spectrum
    is either dB or a linear amplitude ratio. The parenthesised part of the
    label is the unit; no parentheses means a bare ratio, which needs more
    decimals than a dB value to be readable.
    """
    div = Div(text="&nbsp;", height=18,
              styles={"color": "#5f6368", "font-size": "12px",
                      "font-family": "monospace"})
    if yaxis is None:
        ytxt = f'cb_obj.y.toFixed({ydigits}) + " {yu}"'
        args = dict(div=div)
    else:
        ytxt = ('(function () {\n'
                '  const m = ax.axis_label.match(/\\(([^)]+)\\)/);\n'
                f'  return cb_obj.y.toFixed(m ? {ydigits} : 3)'
                ' + (m ? " " + m[1] : "");\n})()')
        args = dict(div=div, ax=yaxis)
    code = (f'if (cb_obj.x == null) {{ div.text = "&nbsp;"; return; }}\n'
            f'div.text = "{xn} " + cb_obj.x.toFixed({xdigits}) + " {xu}"'
            f' + " &nbsp;&nbsp; {yn} " + {ytxt};')
    fig.js_on_event("mousemove", CustomJS(args=args, code=code))
    fig.js_on_event("mouseleave", CustomJS(
        args=dict(div=div), code='div.text = "&nbsp;";'))
    return div


def _add_crosshair(fig):
    """Toolbar-toggleable crosshair, off by default (click the icon)."""
    fig.add_tools(CrosshairTool())
    fig.toolbar.active_inspect = []


# ---------------------------------------------------------------------------
# ImagePane: the 2-D rendering primitive (variable density or wiggle)
# ---------------------------------------------------------------------------
_PLOT_UID = itertools.count()

_FIT_CHROME_PX = 215   # fallback only; see _FIT_FN_JS

# Panel renders some components into shadow roots, so a plain
# document.querySelector can miss the container; walk shadow roots too.
_DEEP_FIND_JS = """
function deepFind(root, cls) {
  const hit = root.querySelector('.' + cls);
  if (hit) return hit;
  for (const e of root.querySelectorAll('*')) {
    if (e.shadowRoot) {
      const r = deepFind(e.shadowRoot, cls);
      if (r) return r;
    }
  }
  return null;
}
"""

# How tall can the figure be and still end at the bottom of the window?
# Measured, not guessed: the plot column's own box gives both the chrome
# above it (its top offset) and the chrome inside it (its height minus the
# figure's). That holds in every layout we render in -- template, notebook,
# tools='below', fullscreen -- where a single constant could only ever be
# right in one of them. ``chrome`` stays as a fallback for the (impossible
# in practice) case where the container cannot be found.
_FIT_FN_JS = """
function fitHeight(el, sl, chrome, pad) {
  if (!el) return Math.round(window.innerHeight - chrome);
  const r = el.getBoundingClientRect();
  const inner = r.height - sl.value;      // controls row + readout + extras
  return Math.round(window.innerHeight - r.top - inner - pad);
}
"""

_FIT_JS = _DEEP_FIND_JS + _FIT_FN_JS + """
const h = fitHeight(deepFind(document, cls), sl, chrome, 8);
sl.value = Math.min(sl.end, Math.max(sl.start, h));
"""

_FULLSCREEN_JS = _DEEP_FIND_JS + _FIT_FN_JS + """
if (document.fullscreenElement) {
  document.exitFullscreen();
} else {
  const el = deepFind(document, cls);
  if (!el) {
    console.warn('gathervis: fullscreen container .' + cls + ' not found');
  } else {
    if (!el._gvFsHooked) {
      el._gvFsHooked = true;
      document.addEventListener('fullscreenchange', () => {
        if (document.fullscreenElement === el) {
          el._gvPrev = sl.value;
          requestAnimationFrame(() => {      // let the layout settle first
            const h = fitHeight(el, sl, 110, 4);
            sl.value = Math.min(sl.end, Math.max(sl.start, h));
          });
        } else if (el._gvPrev !== undefined) {
          sl.value = el._gvPrev;
        }
      });
    }
    el.requestFullscreen().catch(
      (e) => console.warn('gathervis: fullscreen refused: ' + e));
  }
}
"""


class ImagePane:
    """A 2-D (space, time) panel with two display modes on one bokeh figure:

    * ``density`` -- quantized uint8 image through a colormap (the default);
    * ``wiggle``  -- wiggle traces with positive-lobe fill (variable area).

    ``flip_y=True`` gives the seismic convention (time increasing downward).
    """

    def __init__(self, xlabel="trace", ylabel="time (s)", flip_y=True,
                 cmap="seismic", height=520, match_aspect=False):
        self.mapper = LinearColorMapper(palette=_palette(cmap), low=0, high=255)
        self.cds = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
        self.display = "density"
        self.cmap = cmap                # kept so the download uses them too
        self.flip_y = bool(flip_y)
        self._last = None               # args of the latest update()
        fig = figure(height=height, sizing_mode="stretch_width",
                     x_axis_label=xlabel, y_axis_label=ylabel,
                     match_aspect=match_aspect,
                     tools="pan,wheel_zoom,xwheel_zoom,ywheel_zoom,"
                           "box_zoom,reset,save",
                     active_scroll="wheel_zoom")
        self._r_img = fig.image(image="image", x="x", y="y", dw="dw", dh="dh",
                                source=self.cds, color_mapper=self.mapper)
        self._cds_fill = ColumnDataSource(dict(xs=[], ys=[]))
        self._cds_wig = ColumnDataSource(dict(xs=[], ys=[]))
        self._r_fill = fig.patches(xs="xs", ys="ys", source=self._cds_fill,
                                   fill_color="#111111", line_alpha=0,
                                   visible=False)
        self._r_wig = fig.multi_line(xs="xs", ys="ys", source=self._cds_wig,
                                     line_color="#111111", line_width=0.8,
                                     visible=False)
        fig.y_range.flipped = bool(flip_y)
        fig.x_range.range_padding = fig.y_range.range_padding = 0
        fig.toolbar.logo = None
        self.readout = Div(text="&nbsp;", height=18,
                           styles={"color": "#5f6368", "font-size": "12px",
                                   "font-family": "monospace"})
        yn = ylabel.split(" ")[0]
        yu = ylabel[ylabel.find("(") + 1:ylabel.find(")")] \
            if "(" in ylabel else ""
        cb = CustomJS(args=dict(src=self.cds, div=self.readout,
                                xl=xlabel, yn=yn, yu=yu), code=_READOUT_JS)
        fig.js_on_event("mousemove", cb)
        fig.js_on_event("mouseleave", CustomJS(
            args=dict(div=self.readout), code='div.text = "&nbsp;";'))
        _add_crosshair(fig)
        self.figure = fig

        # -- size controls ------------------------------------------------
        # A unique class per pane so fullscreen targets this plot, not the
        # first one on the page (a workspace can hold several).
        self._cls = f"gv-plot-{next(_PLOT_UID)}"
        self.w_height = pn.widgets.IntSlider(
            name="height", start=240, end=2400, step=20, value=height,
            width=180, margin=(2, 10, 2, 8))
        self.w_height.param.watch(
            lambda e: setattr(self.figure, "height", int(e.new)), "value")
        self.w_fit = pn.widgets.Button(
            name="⤢ fit", width=72, button_type="light", margin=(18, 3),
            description="grow the panel to fill the browser window")
        self.w_fullscreen = pn.widgets.Button(
            name="⛶ fullscreen", width=112, button_type="light",
            margin=(18, 3),
            description="fullscreen this panel (ESC to leave)")
        self.js_fit = self.w_fit.js_on_click(
            args=dict(sl=self.w_height, chrome=_FIT_CHROME_PX, cls=self._cls),
            code=_FIT_JS)
        self.js_fullscreen = self.w_fullscreen.js_on_click(
            args=dict(sl=self.w_height, cls=self._cls), code=_FULLSCREEN_JS)
        self.w_download = pn.widgets.FileDownload(
            callback=self._export_png, filename="gather.png",
            label="⤓ full image", button_type="light", width=118,
            margin=(18, 3), disabled=True,
            description="download full image")

    def size_controls(self):
        """Compact height slider + fit + fullscreen + download row."""
        return pn.Row(self.w_height, self.w_fit, self.w_fullscreen,
                      self.w_download, margin=0)

    def frame(self, *extra, controls=True):
        """The plot, its readout and ``extra``, in the fullscreen container."""
        items = [self.size_controls()] if controls else []
        items += [self.figure, self.readout, *extra]
        return pn.Column(*items, css_classes=["gv-plot", self._cls],
                         sizing_mode="stretch_width")

    def set_cmap(self, name: str):
        self.mapper.palette = _palette(name)   # density mode only
        self.cmap = name

    def set_display(self, mode: str):
        """Switch density <-> wiggle; redraws from the latest data."""
        if mode not in _DISPLAYS:
            raise ValueError(f"display must be one of {_DISPLAYS}")
        if mode == self.display:
            return
        self.display = mode
        wig = mode == "wiggle"
        self._r_wig.visible = self._r_fill.visible = wig
        self._r_img.visible = not wig
        if wig:                              # drop the hidden mode's payload
            self.cds.data = dict(image=[], x=[], y=[], dw=[], dh=[])
        else:
            self._cds_wig.data = dict(xs=[], ys=[])
            self._cds_fill.data = dict(xs=[], ys=[])
        if self._last is not None:
            self.update(*self._last)         # ... which resyncs the download

    def update(self, arr2d, clim, x0=0.0, dx=1.0, y0=0.0, dy=1.0):
        """arr2d: (nx, ny) with axis 0 on x. Decimates before shipping."""
        self._last = (arr2d, clim, x0, dx, y0, dy)
        if self.display == "wiggle":
            self._update_wiggle(arr2d, clim, x0, dx, y0, dy)
        else:
            self._update_image(arr2d, clim, x0, dx, y0, dy)
        self._sync_download()

    # -- full-resolution download -------------------------------------------
    # What the browser gets is stride-decimated to MAX_PX and, in wiggle mode,
    # to MAX_WIGGLE traces; bokeh's own save tool then snapshots the canvas at
    # whatever size the figure happens to be on screen. This button instead
    # re-renders ``_last`` -- the processed array and its clim, i.e. filter,
    # gain, polarity and clip percentile already applied -- at one pixel per
    # trace and per sample, through this pane's colormap and display mode.
    def _export_png(self):
        import io
        arr, clim = self._last[0], self._last[1]
        rgb = render_gather(arr, clim, display=self.display, cmap=self.cmap,
                            flip_y=self.flip_y)
        return io.BytesIO(png_bytes(rgb))

    def _sync_download(self):
        """Enable the button and put the output size in its tooltip."""
        if self._last is None:
            self.w_download.disabled = True
            return
        nx, nt = self._last[0].shape
        w, h = raster_size(nx, nt, self.display)
        over = w * h > _render.MAX_EXPORT_PIXELS
        self.w_download.disabled = over
        self.w_download.description = (
            f"gather too large to export ({w * h / 1e6:.0f} MP)" if over
            else f"download full image ({w} × {h} px)")

    def _update_image(self, arr2d, clim, x0, dx, y0, dy):
        nx, ny = arr2d.shape
        img = quantize(decimate(arr2d, MAX_PX), clim)
        self.cds.data = dict(image=[img.T], x=[x0], y=[y0],
                             dw=[nx * dx], dh=[ny * dy],
                             lo=[float(clim[0])], hi=[float(clim[1])])

    def _update_wiggle(self, arr2d, clim, x0, dx, y0, dy):
        """Wiggle + variable-area fill. Amplitudes are normalized by the clim
        (the clip-percentile slider doubles as wiggle gain), excursion is one
        displayed trace spacing, clipped at +-2 spacings."""
        st = max(1, -(-arr2d.shape[0] // MAX_WIGGLE[0]))   # ceil-div strides,
        sy = max(1, -(-arr2d.shape[1] // MAX_WIGGLE[1]))   # same as decimate()
        a = np.asarray(arr2d[::st, ::sy], dtype=np.float32)
        t = (y0 + dy * sy * np.arange(a.shape[1], dtype=np.float32))
        amp = np.clip(a / max(clim[1], 1e-30), -2.0, 2.0) * (dx * st)
        xs, ys, fxs, fys = [], [], [], []
        for i in range(a.shape[0]):
            base = np.float32(x0 + dx * st * i)
            w = base + amp[i]
            xs.append(w)
            ys.append(t)
            fxs.append(np.concatenate([np.maximum(w, base), [base, base]]))
            fys.append(np.concatenate([t, [t[-1], t[0]]]))
        self._cds_fill.data = dict(xs=fxs, ys=fys)
        self._cds_wig.data = dict(xs=xs, ys=ys)


_WIN_COLORS = ("#e6741e", "#2ca02c", "#9467bd", "#d62728", "#17becf",
               "#8c564b")
_SPEC_SIZES = {"S": (180, 650), "M": (240, 900), "L": (320, None)}
# 'per trace' draws the window's traces individually; past this many the
# curves stop being distinguishable and the browser starts to hurt, so the
# traces are evenly subsampled and the legend says how many are shown.
_MAX_SPEC_CURVES = 48
PROC_MAX_BYTES = 512 * 1024 ** 2   # process whole volumes up to this size


class WindowTool:
    """Rectangle / polygon analysis windows drawn on a gather panel, plus
    in-window amplitude spectra and JSON export.

    Windows live in data coordinates (trace on x, time on y), so they stay in
    place while browsing shots; spectra are computed from what the panel
    currently shows (i.e. after any filtering), one curve per window, gated
    (boxcar) and averaged over the traces the window covers. Toolbar gestures:
    box tool -- SHIFT+drag (or click, move, click); polygon tool -- click each
    vertex, double-click / ESC to finish; BACKSPACE deletes a selected window.
    """

    def __init__(self, pane: ImagePane):
        self.pane = pane
        self._cards = {}          # (name, sidebar) -> pn.Card, built once
        fig = pane.figure
        self.cds_rect = ColumnDataSource(dict(x=[], y=[], width=[], height=[],
                                              color=[]))
        self.cds_poly = ColumnDataSource(dict(xs=[], ys=[], color=[]))
        r_rect = fig.rect(x="x", y="y", width="width", height="height",
                          source=self.cds_rect, fill_color="color",
                          fill_alpha=0.10, line_color="color", line_width=2)
        r_poly = fig.patches(xs="xs", ys="ys", source=self.cds_poly,
                             fill_color="color", fill_alpha=0.10,
                             line_color="color", line_width=2)
        fig.add_tools(BoxEditTool(renderers=[r_rect],
                                  empty_value=_WIN_COLORS[0],
                                  description="draw rectangle window"),
                      PolyDrawTool(renderers=[r_poly],
                                   empty_value=_WIN_COLORS[0],
                                   description="draw polygon window"))

        self.cds_spec = ColumnDataSource(dict(xs=[], ys=[], color=[], label=[],
                                              alpha=[], width=[]))
        sfig = figure(height=_SPEC_SIZES["M"][0], max_width=_SPEC_SIZES["M"][1],
                      sizing_mode="stretch_width",
                      x_axis_label="frequency (Hz)",
                      y_axis_label="amplitude",
                      tools="pan,wheel_zoom,box_zoom,reset,save")
        sfig.multi_line(xs="xs", ys="ys", line_color="color",
                        line_width="width", line_alpha="alpha",
                        legend_field="label", source=self.cds_spec)
        sfig.legend.location = "top_right"
        sfig.toolbar.logo = None
        _add_crosshair(sfig)              # compare peak heights along a line
        sfig.visible = False
        self.spec_fig = sfig
        self.spec_readout = _coord_readout(sfig, "freq", "Hz", "amp", "dB",
                                           yaxis=sfig.yaxis[0])
        self.spec_readout.visible = False

        # f-k panel (first window): freq on y, wavenumber on x
        self._fk_mapper = LinearColorMapper(palette=_palette("rainbow"),
                                            low=-60, high=0)
        self.cds_fk = ColumnDataSource(dict(image=[], x=[], y=[], dw=[],
                                            dh=[]))
        kfig = figure(height=_SPEC_SIZES["M"][0], max_width=_SPEC_SIZES["M"][1],
                      sizing_mode="stretch_width",
                      x_axis_label="wavenumber (1/trace)",
                      y_axis_label="frequency (Hz)",
                      tools="pan,wheel_zoom,box_zoom,reset,save")
        kfig.image(image="image", x="x", y="y", dw="dw", dh="dh",
                   source=self.cds_fk, color_mapper=self._fk_mapper)
        kfig.x_range.range_padding = kfig.y_range.range_padding = 0
        kfig.toolbar.logo = None
        _add_crosshair(kfig)
        kfig.visible = False
        self.fk_fig = kfig
        self.fk_readout = _coord_readout(kfig, "k", "1/trace", "f", "Hz",
                                         xdigits=3, ydigits=1)
        self.fk_readout.visible = False

        self.w_btn = pn.widgets.Button(name="compute spectrum",
                                       button_type="primary", **_W)
        self.w_btn.on_click(lambda e: self.compute())
        self.w_fk = pn.widgets.Button(
            name="f-k spectrum", **_W,
            description="2-D f-k amplitude spectrum of the first window")
        self.w_fk.on_click(lambda e: self.compute_fk())
        self.w_export = pn.widgets.FileDownload(callback=self._export,
                                                filename="windows.json",
                                                label="export JSON", **_W)
        self.w_import = pn.widgets.FileInput(accept=".json", **_WFILE)
        self.w_import.param.watch(self._import, "value")
        self.w_clear = pn.widgets.Button(
            name="clear windows", **_W,
            description="remove all drawn windows (and their spectra)")
        self.w_clear.on_click(lambda e: self.clear_windows())
        self.w_scale = pn.widgets.Select(
            name="y axis", value="amplitude", **_W,
            options=["amplitude", "dB"],
            description="linear amplitude, normalized to the window's peak -- "
                        "shows where the energy actually is; or dB, which "
                        "stretches the weak tail and the noise floor")
        self.w_scale.param.watch(lambda e: self.refresh(), "value")
        self.w_traces = pn.widgets.Select(
            name="traces", value="per trace", **_W,
            options=["per trace", "mean", "middle trace"],
            description="how the window's traces reach the plot: each one "
                        "drawn (nothing combined), their magnitudes averaged, "
                        "or just the middle trace")
        self.w_traces.param.watch(lambda e: self.refresh(), "value")
        self.w_size = pn.widgets.Select(name="size", value="M", **_W,
                                        options=list(_SPEC_SIZES),
                                        description="spectrum panel size")
        self.w_size.param.watch(lambda e: self._resize(e.new), "value")

    def _resize(self, key):
        h, w = _SPEC_SIZES[key]
        for f in (self.spec_fig, self.fk_fig):
            f.height = h
            f.max_width = w

    # -- window definitions in data coordinates -----------------------------
    def windows(self):
        out = []
        d = self.cds_rect.data
        for x, y, w, h in zip(d["x"], d["y"], d["width"], d["height"]):
            out.append(dict(kind="rect",
                            trace=sorted((float(x - w / 2), float(x + w / 2))),
                            time=sorted((float(y - h / 2), float(y + h / 2)))))
        d = self.cds_poly.data
        for xs, ys in zip(d["xs"], d["ys"]):
            if len(xs) >= 3:
                out.append(dict(kind="polygon",
                                trace=[float(v) for v in xs],
                                time=[float(v) for v in ys]))
        return out

    @staticmethod
    def _mask(win, nx, nt, x0, dx, y0, dy):
        """Boolean (nx, nt) in-window mask on the displayed sample grid."""
        xi = x0 + dx * np.arange(nx)
        ti = y0 + dy * np.arange(nt)
        if win["kind"] == "rect":
            (tx0, tx1), (ty0, ty1) = win["trace"], win["time"]
            return (((xi >= tx0) & (xi <= tx1))[:, None]
                    & ((ti >= ty0) & (ti <= ty1))[None, :])
        px = np.asarray(win["trace"], float)          # ray casting, vectorized
        py = np.asarray(win["time"], float)
        X, Y = xi[:, None], ti[None, :]
        inside = np.zeros((nx, nt), bool)
        j = len(px) - 1
        for i in range(len(px)):
            num = (px[j] - px[i]) * (Y - py[i])
            den = (py[j] - py[i]) or 1e-30
            inside ^= ((py[i] > Y) != (py[j] > Y)) & (X < num / den + px[i])
            j = i
        return inside

    # -- spectra -------------------------------------------------------------
    def compute(self):
        """Raw DFT amplitude spectrum of the data inside each window.

        No taper, no averaging, no smoothing of any kind: each trace's gated
        samples go straight into ``rfft`` and what is plotted is |X| in dB,
        normalized by the window's own peak. ``traces`` decides how the
        window's traces reach the plot -- ``per trace`` draws each one
        (nothing combined), ``mean`` averages the magnitudes, ``middle
        trace`` takes the one at the window's centre.

        Two honest consequences of "no processing", both inherent to the DFT
        of a finite, rectangularly cut record rather than bugs:

        * the gate is a boxcar, so its own sidelobes fall away only as 1/f.
          Roughly 40 dB below the peak you stop reading the data and start
          reading the window. A taper would fix that, but a taper is exactly
          a convolution of the spectrum (periodic Hann == (-1/4, 1/2, -1/4)
          on the complex spectrum) -- i.e. smoothing, which is the thing
          being asked for here.
        * a single trace's periodogram is a 2-degree-of-freedom estimate: its
          scatter is real, does not shrink as the record gets longer, and is
          partly the process, partly the estimator. ``mean`` trades that
          scatter for sqrt(N) less variance, which is why it looks smooth.
        """
        if self.pane._last is None:
            return
        arr, _clim, x0, dx, y0, dy = self.pane._last
        arr = np.asarray(arr, dtype=np.float32)
        nx, nt = arr.shape
        mode = self.w_traces.value
        in_db = self.w_scale.value == "dB"
        xs, ys, colors, labels, alphas, widths = [], [], [], [], [], []
        rect_colors, poly_colors = [], []
        for k, win in enumerate(self.windows()):
            color = _WIN_COLORS[k % len(_WIN_COLORS)]
            (rect_colors if win["kind"] == "rect" else poly_colors).append(color)
            m = self._mask(win, nx, nt, x0, dx, y0, dy)
            cols = np.where(m.any(axis=1))[0]
            rows = np.where(m.any(axis=0))[0]
            if cols.size == 0 or rows.size < 8:       # window too small
                continue
            k0, k1 = int(rows[0]), int(rows[-1]) + 1
            gated = arr[cols, k0:k1] * m[cols, k0:k1]     # the window, as cut
            spec = np.abs(np.fft.rfft(gated, axis=-1))    # the DFT. that's it.
            freq = np.fft.rfftfreq(k1 - k0, dy).astype("f4")
            ntr = cols.size
            if mode == "mean":
                curves = [spec.mean(axis=0)]
                note, alpha, width = f"mean of {ntr} tr", 1.0, 2
            elif mode == "middle trace":
                j = ntr // 2
                curves = [spec[j]]
                note, alpha, width = f"trace {int(cols[j])}", 1.0, 2
            else:                                          # per trace: as-is
                step = max(1, -(-ntr // _MAX_SPEC_CURVES))
                curves = list(spec[::step])
                note = (f"{len(curves)} tr" if step == 1
                        else f"{len(curves)} of {ntr} tr")
                alpha, width = (0.9, 1.5) if len(curves) < 4 else (0.35, 1)
            # normalize by the peak of what is actually drawn, so the top of
            # the plot is 1.0 (0 dB) in every mode; one ref for all curves of
            # a window keeps their relative amplitudes readable
            ref = max(float(c.max()) for c in curves) + 1e-30
            for c in curves:
                y = c / ref                      # amplitude, 1.0 at the peak
                if in_db:
                    y = 20.0 * np.log10(y + 1e-6)
                xs.append(freq)
                ys.append(y.astype("f4"))
                colors.append(color)
                # bokeh groups equal legend_field values into a single entry,
                # so an overlay of 48 curves still shows one legend row
                labels.append(f"{win['kind']} {k + 1} · {note} · "
                              f"df {freq[1]:.3g} Hz")
                alphas.append(alpha)
                widths.append(width)
        # recolor the drawn windows so they match their curves
        if len(rect_colors) == len(self.cds_rect.data["x"]):
            self.cds_rect.data = dict(self.cds_rect.data, color=rect_colors)
        if len(poly_colors) == len(self.cds_poly.data["xs"]):
            self.cds_poly.data = dict(self.cds_poly.data, color=poly_colors)
        self.spec_fig.yaxis.axis_label = "amplitude (dB)" if in_db \
            else "amplitude"
        self.cds_spec.data = dict(xs=xs, ys=ys, color=colors, label=labels,
                                  alpha=alphas, width=widths)
        self.spec_fig.visible = bool(xs)
        self.spec_readout.visible = bool(xs)

    def compute_fk(self):
        """f-k amplitude spectrum (dB, peak-normalized) of the first window,
        from the currently displayed (processed) gather. Positive k = events
        dipping toward increasing trace number."""
        wins = self.windows()
        if not wins or self.pane._last is None:
            return
        arr, _clim, x0, dx, y0, dy = self.pane._last
        arr = np.asarray(arr, dtype=np.float32)
        nx, nt = arr.shape
        m = self._mask(wins[0], nx, nt, x0, dx, y0, dy)
        cols = np.where(m.any(axis=1))[0]
        rows = np.where(m.any(axis=0))[0]
        if cols.size < 4 or rows.size < 8:
            return
        j0, j1 = int(cols[0]), int(cols[-1]) + 1
        k0, k1 = int(rows[0]), int(rows[-1]) + 1
        gated = arr[j0:j1, k0:k1] * m[j0:j1, k0:k1]  # the window, as cut
        spec = np.fft.rfft(gated, axis=1)            # time -> f (>= 0)
        # trace -> k with the e^{+2pi i k x} kernel (ifft; scale irrelevant
        # after normalization) so that events dipping toward increasing trace
        # number land at positive k -- the usual geophysics convention.
        spec = np.fft.ifft(spec, axis=0)
        amp = np.abs(np.fft.fftshift(spec, axes=0))  # (nk, nf), k centered
        db = 20.0 * np.log10(amp / (amp.max() + 1e-30) + 1e-6)
        freqs = np.fft.rfftfreq(k1 - k0, dy)
        ks = np.fft.fftshift(np.fft.fftfreq(j1 - j0, dx))
        self.cds_fk.data = dict(image=[db.T.astype("f4")],   # rows = f
                                x=[float(ks[0])], y=[float(freqs[0])],
                                dw=[float(ks[-1] - ks[0])],
                                dh=[float(freqs[-1] - freqs[0])])
        self.fk_fig.visible = self.fk_readout.visible = True

    def _import(self, event):
        """Load windows from an exported windows.json (FileDropper dict or
        raw bytes)."""
        import json
        raw = event.new
        if isinstance(raw, dict):                  # FileDropper: {name: data}
            raw = next(iter(raw.values()), None)
        if not raw:
            return
        if isinstance(raw, str):
            raw = raw.encode()
        try:
            wins = json.loads(bytes(raw).decode()).get("windows", [])
        except Exception:
            return
        rect = dict(x=[], y=[], width=[], height=[], color=[])
        poly = dict(xs=[], ys=[], color=[])
        for w in wins:
            if w.get("kind") == "rect":
                (tx0, tx1), (ty0, ty1) = w["trace"], w["time"]
                rect["x"].append((tx0 + tx1) / 2)
                rect["y"].append((ty0 + ty1) / 2)
                rect["width"].append(tx1 - tx0)
                rect["height"].append(ty1 - ty0)
                rect["color"].append(_WIN_COLORS[0])
            elif w.get("kind") == "polygon":
                poly["xs"].append(list(w["trace"]))
                poly["ys"].append(list(w["time"]))
                poly["color"].append(_WIN_COLORS[0])
        self.cds_rect.data = rect
        self.cds_poly.data = poly
        if self.spec_fig.visible:
            self.compute()
        if self.fk_fig.visible:
            self.compute_fk()

    def clear_windows(self):
        """Remove every drawn window and hide the (now empty) spectra."""
        self.cds_rect.data = dict(x=[], y=[], width=[], height=[], color=[])
        self.cds_poly.data = dict(xs=[], ys=[], color=[])
        self.cds_spec.data = dict(xs=[], ys=[], color=[], label=[],
                                  alpha=[], width=[])
        self.cds_fk.data = dict(image=[], x=[], y=[], dw=[], dh=[])
        self.spec_fig.visible = self.spec_readout.visible = False
        self.fk_fig.visible = self.fk_readout.visible = False

    def refresh(self):
        """Recompute visible spectra (after a shot change or filter change)."""
        if self.spec_fig.visible:
            self.compute()
        if self.fk_fig.visible:
            self.compute_fk()

    def _export(self):
        import io
        import json
        payload = {"coords": {"x": "trace index", "y": "time (s)"},
                   "windows": self.windows()}
        if self.pane._last is not None:
            payload["dt"] = float(self.pane._last[5])
        return io.StringIO(json.dumps(payload, indent=2))

    def window_card(self, sidebar=False):
        """Analysis-window I/O (drawing happens via the toolbar tools)."""
        key = ("window", sidebar)
        if key not in self._cards:
            self._cards[key] = pn.Card(
                _hint("box tool: SHIFT+drag &middot; polygon tool: click "
                      "vertices, ESC ends &middot; BACKSPACE deletes"),
                self.w_clear, self.w_export,
                _labeled(self.w_import, "import windows (.json)"),
                title="Window", **_card_style(sidebar))
        return self._cards[key]

    def spectrum_card(self, sidebar=False):
        """Spectrum computations on the drawn windows."""
        key = ("spectrum", sidebar)
        if key not in self._cards:
            self._cards[key] = pn.Card(
                self.w_btn, self.w_traces, self.w_scale, self.w_fk,
                self.w_size,
                title="Spectrum", **_card_style(sidebar))
        return self._cards[key]

    def controls(self, sidebar=False):
        return pn.FlexBox(self.window_card(sidebar),
                          self.spectrum_card(sidebar))

    def figures(self):
        """The spectrum / f-k panels (full width, shown after computing)."""
        return pn.Column(self.spec_fig, self.spec_readout,
                         self.fk_fig, self.fk_readout,
                         sizing_mode="stretch_width")

    def panel(self):
        return pn.Column(self.controls(), self.figures(),
                         sizing_mode="stretch_width")


class PickTool:
    """First-arrival / event picking on a gather panel.

    Toolbar point tool: tap to add a pick, drag to move it, BACKSPACE deletes
    the selected one. Picks are stored *per shot* -- they follow the shot you
    are browsing (unlike analysis windows, which deliberately persist across
    shots). A dotted line connects picks in trace order so the moveout reads
    at a glance. Export / import as CSV: ``shot,trace,time_s``.
    """

    _COLOR = "#d81b60"

    def __init__(self, pane: ImagePane):
        self.pane = pane
        self._cards = {}          # (name, sidebar) -> pn.Card, built once
        self.cds = ColumnDataSource(dict(x=[], y=[]))
        self._cds_line = ColumnDataSource(dict(x=[], y=[]))
        pane.figure.line("x", "y", source=self._cds_line, line_width=1.2,
                         line_dash="dotted", line_color=self._COLOR)
        r = pane.figure.scatter("x", "y", source=self.cds, size=9,
                                marker="inverted_triangle",
                                fill_color=self._COLOR, fill_alpha=0.9,
                                line_color="white", line_width=1)
        pane.figure.add_tools(PointDrawTool(renderers=[r],
                                            description="pick events"))
        self.cds.on_change("data", lambda a, o, n: self._sync_line())
        self._store = {}                    # shot index -> (xs, ys)
        self._shot = 0
        self.w_export = pn.widgets.FileDownload(callback=self._export,
                                                filename="picks.csv",
                                                label="export CSV", **_W)
        self.w_import = pn.widgets.FileInput(accept=".csv", **_WFILE)
        self.w_import.param.watch(self._import, "value")
        self.w_snap = pn.widgets.Select(
            name="snap", value="off", **_W,
            options=["off", "peak", "trough", "|max|"],
            description="snap picks to the nearest extremum of their trace "
                        "(searches +-30 ms)")
        self.w_sta = pn.widgets.FloatInput(name="STA (s)", value=0.02,
                                           start=0.001, step=0.01, **_W,
                                           description="short-term window")
        self.w_lta = pn.widgets.FloatInput(name="LTA (s)", value=0.2,
                                           start=0.01, step=0.05, **_W,
                                           description="long-term window")
        self.w_thr = pn.widgets.FloatInput(name="threshold", value=4.0,
                                           start=1.1, step=0.5, **_W,
                                           description="STA/LTA trigger ratio")
        self.w_auto = pn.widgets.Button(
            name="auto pick (STA/LTA)", button_type="primary", **_W,
            description="first breaks of the current shot from the displayed "
                        "(filtered) gather; refine by dragging or snapping")
        self.w_auto.on_click(lambda e: self.auto_pick())
        self.w_clear = pn.widgets.Button(
            name="clear shot", **_W,
            description="remove all picks on the current shot")
        self.w_clear.on_click(lambda e: self.clear_shot())
        self.w_clear_all = pn.widgets.Button(
            name="clear all", **_W,
            description="remove the picks of every shot")
        self.w_clear_all.on_click(lambda e: self.clear_all())
        self.w_clear_fb = pn.widgets.Button(
            name="clear picks", **_W,
            description="remove the current shot's picks (retry with new "
                        "STA/LTA parameters)")
        self.w_clear_fb.on_click(lambda e: self.clear_shot())
        self._busy = False
        self.cds.on_change("data", lambda a, o, n: self._apply_snap())

    def _apply_snap(self):
        """Snap picks to the nearest peak / trough / |max| of their trace,
        searching +-30 ms on the *displayed* (processed) gather. Reentrancy-
        guarded: writing the snapped positions re-fires this handler."""
        if (self._busy or self.w_snap.value == "off"
                or self.pane._last is None or not len(self.cds.data["x"])):
            return
        arr, _c, x0, dx, y0, dy = self.pane._last
        arr = np.asarray(arr)
        nx, nt = arr.shape
        w = max(2, int(round(0.03 / dy)))
        mode = self.w_snap.value
        xs, ys, changed = [], [], False
        for x, y in zip(self.cds.data["x"], self.cds.data["y"]):
            j = int(np.clip(round((x - x0) / dx), 0, nx - 1))
            k = int(np.clip(round((y - y0) / dy), 0, nt - 1))
            k0, k1 = max(0, k - w), min(nt, k + w + 1)
            seg = arr[j, k0:k1]
            if mode == "peak":
                kk = k0 + int(np.argmax(seg))
            elif mode == "trough":
                kk = k0 + int(np.argmin(seg))
            else:
                kk = k0 + int(np.argmax(np.abs(seg)))
            nx_, ny_ = x0 + dx * j, y0 + dy * kk
            changed |= (nx_ != x) or (ny_ != y)
            xs.append(nx_)
            ys.append(ny_)
        if changed:
            self._busy = True
            try:
                self.cds.data = dict(x=xs, y=ys)
            finally:
                self._busy = False
            self._sync_line()

    def auto_pick(self):
        """STA/LTA first breaks for the current shot, from the displayed
        (processed) gather -- replaces this shot's picks; drag / snap to
        refine afterwards. Traces that never trigger get no pick."""
        if self.pane._last is None:
            return
        arr, _c, x0, dx, y0, dy = self.pane._last
        arr = np.asarray(arr, dtype=np.float32)
        t = pick_first_breaks(arr, dy, sta=float(self.w_sta.value),
                              lta=float(self.w_lta.value),
                              thresh=float(self.w_thr.value))
        good = np.flatnonzero(np.isfinite(t))
        step = max(1, len(good) // 240)      # keep the CDS light
        good = good[::step]
        self.cds.data = dict(x=(x0 + dx * good).tolist(),
                             y=(y0 + np.asarray(t)[good]).tolist())

    def _sync_line(self):
        x = np.asarray(self.cds.data["x"], dtype=float)
        y = np.asarray(self.cds.data["y"], dtype=float)
        o = np.argsort(x)
        self._cds_line.data = dict(x=x[o].tolist(), y=y[o].tolist())

    def _flush(self):
        self._store[self._shot] = (list(self.cds.data["x"]),
                                   list(self.cds.data["y"]))

    def set_shot(self, i: int):
        """Swap the displayed picks to shot ``i`` (saving the current ones)."""
        i = int(i)
        if i == self._shot:
            return
        self._flush()
        xs, ys = self._store.get(i, ([], []))
        self._shot = i
        self.cds.data = dict(x=list(xs), y=list(ys))

    def clear_shot(self):
        """Remove the picks of the current shot."""
        self._store.pop(self._shot, None)
        self.cds.data = dict(x=[], y=[])

    def clear_all(self):
        """Remove the picks of every shot."""
        self._store = {}
        self.cds.data = dict(x=[], y=[])

    def picks(self) -> dict:
        """{shot: (traces, times)} for every shot that has picks."""
        self._flush()
        return {k: v for k, v in sorted(self._store.items()) if len(v[0])}

    def _export(self):
        import io
        buf = io.StringIO()
        buf.write("shot,trace,time_s\n")
        for s, (xs, ys) in self.picks().items():
            for x, y in sorted(zip(xs, ys)):
                buf.write(f"{s},{x:.2f},{y:.6f}\n")
        buf.seek(0)
        return buf

    def _import(self, event):
        raw = event.new
        if isinstance(raw, dict):                  # FileDropper: {name: data}
            raw = next(iter(raw.values()), None)
        if not raw:
            return
        if isinstance(raw, str):
            raw = raw.encode()
        store = {}
        for ln in bytes(raw).decode().splitlines():
            parts = ln.strip().split(",")
            if len(parts) != 3:
                continue
            try:
                s, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue                    # header or junk line
            store.setdefault(s, ([], []))
            store[s][0].append(x)
            store[s][1].append(y)
        self._store = store
        xs, ys = store.get(self._shot, ([], []))
        self.cds.data = dict(x=list(xs), y=list(ys))

    def picking_card(self, sidebar=False):
        """Manual event picking: snap refinement + pick I/O."""
        key = ("picking", sidebar)
        if key not in self._cards:
            self._cards[key] = pn.Card(
                _hint("point toolbar tool: tap adds, drag moves, tap-select "
                      "+ BACKSPACE deletes one"),
                self.w_snap,
                pn.Row(self.w_clear, self.w_clear_all, **_W),
                self.w_export,
                _labeled(self.w_import, "import picks (.csv)"),
                title="Event picking", **_card_style(sidebar))
        return self._cards[key]

    def fb_card(self, sidebar=False):
        """Automatic first-break picking (gapped STA/LTA)."""
        key = ("fb", sidebar)
        if key not in self._cards:
            self._cards[key] = pn.Card(
                self.w_auto,
                pn.Row(self.w_sta, self.w_lta, self.w_thr, **_W),
                self.w_clear_fb,
                title="FB picking", **_card_style(sidebar))
        return self._cards[key]

    def controls(self, sidebar=False):
        return pn.FlexBox(self.picking_card(sidebar), self.fb_card(sidebar))

    def panel(self):
        return self.controls()


def _display_controls(state, panes, redraw, cmap, perc):
    """Shared display-mode + cmap + clim-percentile widgets.
    ``state['clim']`` is read by redraw()."""
    w_disp = pn.widgets.Select(name="display", options=list(_DISPLAYS),
                               value="density", width=110)
    w_cmap = pn.widgets.Select(name="colormap", options=list(_CMAPS), value=cmap,
                               width=130)
    w_perc = pn.widgets.FloatSlider(name="clip percentile", start=80.0, end=100.0,
                                    step=0.5, value=perc, width=180)

    def _on_disp(event):
        for p in panes:
            p.set_display(event.new)

    def _on_cmap(event):
        for p in panes:
            p.set_cmap(event.new)

    def _on_perc(event):
        state["clim"] = robust_clim(state["sample"], event.new)
        redraw()

    w_disp.param.watch(_on_disp, "value")
    w_cmap.param.watch(_on_cmap, "value")
    w_perc.param.watch(_on_perc, "value_throttled")   # apply on release
    return pn.Row(w_disp, w_cmap, w_perc)


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------
def view_gather(arr2d, dt=1.0, t0=0.0, cmap="seismic", perc=98.0, title=""):
    """Single (trace, time) panel."""
    _ensure_ext()
    state = {"sample": np.asarray(arr2d), "clim": None}
    state["clim"] = robust_clim(state["sample"], perc)
    pane = ImagePane(cmap=cmap)

    pane.w_download.filename = f"{_slug(title)}.png"

    def redraw():
        pane.update(arr2d, state["clim"], y0=t0, dy=dt)

    redraw()
    header = pn.pane.Markdown(f"### {title or 'gather'}")
    return pn.Column(header, _display_controls(state, [pane], redraw, cmap, perc),
                     pane.frame(), sizing_mode="stretch_width")


def _colorscale(name: str):
    """Plotly colorscale (33 stops) from the shared 256-color palette."""
    cols = _palette(name)
    idx = np.linspace(0, 255, 33).round().astype(int)
    return [[float(i) / 255.0, cols[i]] for i in idx]


def _dec_coords(n: int, budget: int) -> np.ndarray:
    """Index coordinates matching ``decimate``'s stride on an axis of length n."""
    step = max(1, -(-n // budget))
    return np.arange(0, n, step, dtype=np.float32)


class VolumeView3D:
    """cigvis-style volume view: a 3-D cuboid with three axis-aligned slice
    planes, rendered in the browser as plotly WebGL surfaces.

    Same technique as cigvis's plotly backend (cigvis's primary engine is
    vispy -- desktop OpenGL, not embeddable in a web app), but wired into
    gathervis's lazy pipeline: moving a slider reads only that slice from the
    (possibly memmap-backed) volume, decimates it to ``MAX_PX_3D`` and
    quantizes to uint8 before shipping.

    The figure is kept as a *plain dict*, never a ``go.Figure``: with a live
    Figure, Panel routes in-place mutations through ``Plotly.restyle``, and
    plotly.js gl3d traces mis-apply restyled ``surfacecolor`` (the whole plane
    collapses to the colormap's minimum color -- plotly.js #2866). With a dict,
    Panel diffs the columns server-side (only changed slices go on the wire)
    and re-renders via ``Plotly.react``, which is correct.
    """

    _LIGHT = dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0)

    def __init__(self, vol, names=("recy", "recx"), dt=1.0, t0=0.0,
                 cmap="seismic", perc=98.0, clim=None, vertical="time (s)",
                 symmetric=True):
        _ensure_ext()
        self.vol, self.names, self.dt, self.t0 = vol, names, dt, t0
        self.vertical = vertical            # z-axis label: time (s) / depth (m)
        self.clim = clim or robust_clim(np.asarray(vol), perc,
                                        symmetric=symmetric)
        self._cs = _colorscale(cmap)

        n0, n1, nt = vol.shape
        # start on the three camera-facing faces -> the classic closed cuboid
        self.w0 = pn.widgets.IntSlider(name=names[0], start=0, end=n0 - 1,
                                       value=n0 - 1)
        self.w1 = pn.widgets.IntSlider(name=names[1], start=0, end=n1 - 1,
                                       value=n1 - 1)
        self.w2 = pn.widgets.IntSlider(name=f"{vertical.split()[0]} sample",
                                       start=0, end=nt - 1, value=0)
        opts = [round(2.0 ** (i / 4.0 - 3.0), 3) for i in range(25)]
        vname = vertical.split()[0]          # "time" / "depth"
        self.w_stretch = [
            pn.widgets.DiscreteSlider(name=f"{lab} stretch", options=opts,
                                      value=1.0)
            for lab in (names[0], names[1], vname)]
        self._stretch = [1.0, 1.0, 1.0]
        self.fig = {"data": [self._surf(0, self.w0.value),
                             self._surf(1, self.w1.value),
                             self._surf(2, self.w2.value),
                             self._box()],
                    "layout": self._layout()}
        self.pane = pn.pane.Plotly(self.fig, sizing_mode="stretch_width",
                                   height=620, config={"displaylogo": False})
        # live updates while dragging (each step is one slice on the wire)
        for w, axis in ((self.w0, 0), (self.w1, 1), (self.w2, 2)):
            w.param.watch(lambda e, a=axis: self._move(a), "value")
        for axis, w in enumerate(self.w_stretch):
            w.param.watch(lambda e, a=axis: self._on_stretch(a, e.new), "value")
        # Panel's viewport machinery only tracks 2-D axis ranges, so the
        # user's 3-D camera would be lost on every react; capture it from
        # relayout events and keep it in our layout instead.
        self.pane.param.watch(self._on_relayout, "relayout_data")

    def _on_relayout(self, event):
        rd = event.new or {}
        for key, val in rd.items():
            if not key.startswith("scene."):
                continue
            node = self.fig["layout"]
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = val

    def _on_stretch(self, axis, v):
        self._stretch[axis] = float(v)
        key = ("x", "y", "z")[axis]
        r = self.fig["layout"]["scene"]["aspectratio"]
        r[key] = self._ratio[axis] * self._stretch[axis]
        self._push()                       # layout-only diff: tiny message

    def _push(self):
        """Ship the mutated figure. Panel diffs per trace column, so only the
        arrays that actually changed are re-sent to the browser."""
        self.pane.param.trigger("object")

    # -- trace builders ----------------------------------------------------
    def _tcoords(self, nt):
        return (self.t0 + self.dt * _dec_coords(nt, MAX_PX_3D[1])).astype("f4")

    def _surf(self, axis: int, idx: int) -> dict:
        """One axis-aligned slice as plotly surface kwargs.

        Reads only ``vol[...idx...]`` (lazy on memmaps), then decimates and
        quantizes exactly like the 2-D panels.
        """
        n0, n1, nt = self.vol.shape
        if axis == 0:                                     # plane at names[0]=idx
            sl = decimate(self.vol[idx], MAX_PX_3D)       # (n1d, ntd)
            Y, Z = np.meshgrid(_dec_coords(n1, MAX_PX_3D[0]),
                               self._tcoords(nt), indexing="ij")
            X = np.full(sl.shape, float(idx), np.float32)
        elif axis == 1:                                   # plane at names[1]=idx
            sl = decimate(self.vol[:, idx], MAX_PX_3D)    # (n0d, ntd)
            X, Z = np.meshgrid(_dec_coords(n0, MAX_PX_3D[0]),
                               self._tcoords(nt), indexing="ij")
            Y = np.full(sl.shape, float(idx), np.float32)
        else:                                             # time plane at sample idx
            sl = decimate(self.vol[:, :, idx], MAX_PX_3D)  # (n0d, n1d)
            X, Y = np.meshgrid(_dec_coords(n0, MAX_PX_3D[0]),
                               _dec_coords(n1, MAX_PX_3D[1]), indexing="ij")
            Z = np.full(sl.shape, self.t0 + self.dt * idx, np.float32)
        return dict(type="surface", x=X, y=Y, z=Z,
                    surfacecolor=quantize(sl, self.clim),
                    colorscale=self._cs, cmin=0, cmax=255, showscale=False,
                    lighting=self._LIGHT, hoverinfo="skip", name=f"slice{axis}")

    def _box(self) -> dict:
        """Wireframe of the volume extent (12 edges as one polyline)."""
        n0, n1, nt = self.vol.shape
        xs, ys = (0.0, float(n0 - 1)), (0.0, float(n1 - 1))
        zs = (self.t0, self.t0 + self.dt * (nt - 1))
        edges = ([((xs[0], y, z), (xs[1], y, z)) for y in ys for z in zs]
                 + [((x, ys[0], z), (x, ys[1], z)) for x in xs for z in zs]
                 + [((x, y, zs[0]), (x, y, zs[1])) for x in xs for y in ys])
        bx, by, bz = [], [], []
        for a, b in edges:
            bx += [a[0], b[0], None]
            by += [a[1], b[1], None]
            bz += [a[2], b[2], None]
        return dict(type="scatter3d", x=bx, y=by, z=bz, mode="lines",
                    line=dict(color="#7f8388", width=2),
                    hoverinfo="skip", showlegend=False, name="extent")

    def _layout(self) -> dict:
        n0, n1, nt = self.vol.shape
        d = np.array([n0, n1, nt], dtype=float)
        r = np.clip(np.sqrt(d / d.max()), 0.3, 1.0)  # sqrt: soften nt >> n0|n1
        self._ratio = tuple(float(x) for x in r)
        ax = dict(showbackground=False, showspikes=False)
        return dict(
            margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
            # same shape (slice moves, clim, per-shot browse) -> keep camera;
            # different shape -> reset the view
            uirevision=str(self.vol.shape),
            scene=dict(
                xaxis=dict(title=self.names[0], **ax),
                yaxis=dict(title=self.names[1], **ax),
                zaxis=dict(title=self.vertical, autorange="reversed", **ax),
                aspectmode="manual",
                aspectratio=dict(x=float(r[0]) * self._stretch[0],
                                 y=float(r[1]) * self._stretch[1],
                                 z=float(r[2]) * self._stretch[2]),
                camera=dict(eye=dict(x=1.55, y=1.55, z=0.9)),
            ),
        )

    # -- updates: mutate the dict + trigger; Panel diffs and reacts ----------
    def _move(self, axis: int, push: bool = True):
        idx = (self.w0, self.w1, self.w2)[axis].value
        self.fig["data"][axis] = self._surf(axis, idx)
        if push:
            self._push()

    def set_cmap(self, name: str):
        self._cs = _colorscale(name)
        for trace in self.fig["data"][:3]:
            trace["colorscale"] = self._cs
        self._push()

    def set_display(self, mode: str):
        """No-op: wiggle display only applies to the 2-D panels."""

    def redraw(self):
        """Re-slice all three planes (after a clim change or volume swap)."""
        for axis in (0, 1, 2):
            self._move(axis, push=False)
        self._push()

    def set_volume(self, vol):
        """Swap the volume (per-shot browsing of 3-D shot gathers)."""
        same_shape = tuple(vol.shape) == tuple(self.vol.shape)
        camera = self.fig["layout"]["scene"].get("camera")
        self.vol = vol
        n0, n1, nt = vol.shape
        from param.parameterized import discard_events
        for w, n in ((self.w0, n0), (self.w1, n1), (self.w2, nt)):
            with discard_events(w):          # silent: redraw() repaints below
                w.end = n - 1
                w.value = min(w.value, n - 1)
        self.fig["data"][3] = self._box()
        self.fig["layout"] = self._layout()
        if same_shape and camera is not None:   # keep the view while browsing
            self.fig["layout"]["scene"]["camera"] = camera
        self.redraw()

    @property
    def sliders(self):
        return pn.Column(pn.Row(self.w0, self.w1, self.w2),
                         pn.Row(*self.w_stretch))

    def panel(self):
        return pn.Column(self.sliders, self.pane, sizing_mode="stretch_width")


class ShotBrowser:
    """Slider over the shot axis; each shot is a 2-D panel or a 3-D cuboid view.

    ``state`` is a shared dict holding 'clim' (and 'sample'), owned by the
    Workspace so all views stay consistent.
    """

    def __init__(self, g: Gathers, state: dict, cmap="seismic",
                 tools="sidebar"):
        _ensure_ext()
        self.g, self.state = g, state
        self._per_shot_3d = g.data.ndim == 4
        self._tools_sidebar = tools == "sidebar"
        self._help = None
        self.wt = None
        self._stem = _slug(g.name)
        if self._per_shot_3d:
            self.sv = VolumeView3D(g.shot(0), names=g.axes[1:3], dt=g.dt,
                                   t0=g.t0, cmap=cmap, clim=state["clim"])
            self.panes = [self.sv]     # exposes set_cmap like an ImagePane
        else:
            self.pane = ImagePane(xlabel=g.axes[1], cmap=cmap)
            self.panes = [self.pane]
            self.wt = WindowTool(self.pane)   # windows + spectra + export
            self.pt = PickTool(self.pane)     # per-shot event picking

        self.w_shot = pn.widgets.IntSlider(name="shot", start=0, end=g.nshot - 1,
                                           value=0, sizing_mode="stretch_width")
        self.w_jump = pn.widgets.IntInput(name="go to shot", start=0,
                                          end=g.nshot - 1, value=0, width=110)
        # live shot flipping while dragging; typing a number jumps there
        self.w_shot.param.watch(lambda e: self._sync(e.new), "value")
        self.w_jump.param.watch(lambda e: self.set_shot(e.new), "value")
        self.on_shot_change = None      # hook: called with the new shot index
        self.redraw()

    @property
    def ishot(self) -> int:
        return int(self.w_shot.value)

    def set_shot(self, i: int):
        i = max(0, min(int(i), self.g.nshot - 1))
        self.w_shot.value = i        # fires _sync via watcher (if changed)

    def _sync(self, i: int):
        """Post-change bookkeeping: mirror widgets, then redraw."""
        self.w_jump.value = int(i)            # equal values don't re-trigger
        self.redraw()

    def redraw(self):
        arr = self.g.shot(self.ishot)          # reads only this shot's bytes
        arr, clim = _process_gather(arr, self.g.dt, self.state)
        if self._per_shot_3d:
            self.sv.clim = clim
            self.sv.set_volume(arr)
        else:
            self.pane.update(arr, clim, y0=self.g.t0, dy=self.g.dt)
        if self.wt is not None:
            self.wt.refresh()                 # keep spectra in sync
            self.pt.set_shot(self.ishot)      # picks follow the shot
            self.pane.w_download.filename = \
                f"{self._stem}_shot{self.ishot:04d}.png"
        if self.on_shot_change is not None:
            self.on_shot_change(self.ishot)

    def tool_cards(self):
        """Window / Spectrum / picking cards, for hosting in the sidebar.

        Empty for 4-D data, which browses each shot as a 3-D cuboid and has
        no 2-D panel to draw windows or picks on.
        """
        if self.wt is None:
            return []
        if self._help is None:
            self._help = _gesture_help()
        btn, body = self._help
        return [self.wt.window_card(True), self.wt.spectrum_card(True),
                self.pt.picking_card(True), self.pt.fb_card(True), btn, body]

    def panel(self):
        if self._per_shot_3d:
            body = self.sv.panel()
        elif self._tools_sidebar:
            # cards live in the sidebar; only the figure and the spectra
            # it produces stay in the main column
            body = self.pane.frame(self.wt.figures())
        else:
            if self._help is None:
                self._help = _gesture_help()
            help_btn, help_body = self._help
            body = self.pane.frame(
                self.wt.figures(),
                pn.FlexBox(self.wt.window_card(),
                           self.wt.spectrum_card(),
                           self.pt.picking_card(),
                           self.pt.fb_card(), help_btn),
                help_body)
        return pn.Column(pn.Row(self.w_shot, self.w_jump),
                         body, sizing_mode="stretch_width")


class LayoutMap:
    """Acquisition map: sources are tappable; active shot + receivers highlighted."""

    def __init__(self, geo):
        _ensure_ext()
        self.geo = geo
        self.on_pick = None             # hook: called with the picked shot index
        fig = figure(height=520, sizing_mode="stretch_width", match_aspect=True,
                     x_axis_label="x", y_axis_label="y",
                     tools="pan,wheel_zoom,xwheel_zoom,ywheel_zoom,"
                           "box_zoom,reset,tap",
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
    if g.axes[-1] == "depth":
        return "property volume (depth)"
    return "single gather" if g.data.ndim == 2 else "volume"


def _info_md(g: Gathers) -> str:
    n_bytes = int(np.prod(g.shape)) * np.dtype(g.data.dtype).itemsize
    size = (f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e9
            else f"{n_bytes / 1e6:.1f} MB")
    lines = [f"### {g.name or 'dataset'}",
             f"**survey**&nbsp; {_survey_kind(g)}",
             f"**shape**&nbsp; {tuple(g.shape)}",
             f"**axes**&nbsp; ({', '.join(g.axes)})",
             (f"**dz**&nbsp; {g.dt:g} m &nbsp;·&nbsp; "
              f"depth {g.t0 + g.dt * (g.nt - 1):g} m"
              if g.axes[-1] == "depth" else
              f"**dt**&nbsp; {g.dt * 1e3:g} ms &nbsp;·&nbsp; "
              f"record {g.t0 + g.dt * (g.nt - 1):.3f} s"),
             f"**size**&nbsp; {size} {np.dtype(g.data.dtype).name}"]
    if g.geometry is not None:
        geo = g.geometry
        kind = "shared spread" if geo.shared else "per-shot spread"
        lines.append(f"**geometry**&nbsp; {geo.ns} sources · "
                     f"{geo.nr} receivers ({kind})")
    return "\n\n".join(lines)


class Workspace:
    """Sidebar (dataset info + display controls) + tabbed views.

    Tabs adapt to the data: shot browsing, whole-volume slicing, and -- when
    geometry is present -- an acquisition-layout tab. Tapping a source on the
    layout map selects that shot and jumps to the shot-gather tab.
    """

    def __init__(self, g: Gathers, cmap="seismic", perc=98.0, view=None,
                 tools="sidebar"):
        _ensure_ext()
        self.g = g
        self._tools_sidebar = tools == "sidebar"
        self._tool_cards = []          # cards hosted in the sidebar
        self._tool_tabs = set()        # tab indices those cards apply to
        symmetric = g.axes[-1] != "depth"   # property volumes: min/max-style
        self.state = {"sample": g.sample(), "clim": None,
                      "symmetric": symmetric, "perc": perc}
        self.state["clim"] = robust_clim(self.state["sample"], perc,
                                         symmetric=symmetric)
        self._panes, self._redraws, tabs = [], [], []
        self.map = None
        self._shot_tab = None

        # -- tab 1: shot browsing -----------------------------------------
        if "shot" in g.axes:
            self.browser = ShotBrowser(g, self.state, cmap=cmap, tools=tools)
            self._panes += self.browser.panes
            self._redraws.append(self.browser.redraw)
            label = "Shot gathers" + (" (3-D)" if g.data.ndim == 4 else "")
            self._shot_tab = len(tabs)
            tabs.append((label, self.browser.panel()))
            if self._tools_sidebar:
                cards = self.browser.tool_cards()
                if cards:
                    self._tool_cards = cards
                    self._tool_tabs.add(self._shot_tab)

        # -- tab 2: geometry (acquisition layout) --------------------------
        if g.geometry is not None:
            self.map = LayoutMap(g.geometry)
            self.map.on_pick = self._pick_shot
            self.browser.on_shot_change = self.map.set_active
            self.map.set_active(self.browser.ishot)
            tabs.append(("Geometry", pn.Column(self.map.figure,
                                               sizing_mode="stretch_width")))

        # -- tab 3: volume cuboid + slice planes (3-D data) -----------------
        if g.data.ndim == 3:
            vlab = "depth (m)" if g.axes[-1] == "depth" else "time (s)"
            self.slices = VolumeView3D(g.data, names=tuple(g.axes[:-1]),
                                       dt=g.dt, t0=g.t0, cmap=cmap,
                                       clim=self.state["clim"], vertical=vlab)
            self._panes.append(self.slices)
            self._redraws.append(self._redraw_slices)
            tabs.append(("Volume slices", self.slices.panel()))

        # -- single 2-D gather ---------------------------------------------
        if g.data.ndim == 2:
            vlab = "depth (m)" if g.axes[-1] == "depth" else "time (s)"
            self._pane2d = ImagePane(xlabel=g.axes[0], ylabel=vlab, cmap=cmap)
            self._panes.append(self._pane2d)
            self._redraws.append(self._draw2d)
            self._wt2d = WindowTool(self._pane2d)
            self._pt2d = PickTool(self._pane2d)
            self._pane2d.w_download.filename = f"{_slug(g.name)}.png"
            self._draw2d()
            help_btn, help_body = _gesture_help()
            if self._tools_sidebar:
                tabs.append(("Gather",
                             self._pane2d.frame(self._wt2d.figures())))
                self._tool_cards = [self._wt2d.window_card(True),
                                    self._wt2d.spectrum_card(True),
                                    self._pt2d.picking_card(True),
                                    self._pt2d.fb_card(True),
                                    help_btn, help_body]
                self._tool_tabs.add(len(tabs) - 1)
            else:
                tabs.append(("Gather", self._pane2d.frame(
                    self._wt2d.figures(),
                    pn.FlexBox(self._wt2d.window_card(),
                               self._wt2d.spectrum_card(),
                               self._pt2d.picking_card(),
                               self._pt2d.fb_card(), help_btn),
                    help_body)))

        self.tabs = pn.Tabs(*tabs, sizing_mode="stretch_width")
        if view == "slices":
            for i, (label, _) in enumerate(tabs):
                if label == "Volume slices":
                    self.tabs.active = i

        # The tool cards belong to the 2-D gather panel, so they are hidden
        # on the Geometry / Volume-slices tabs. Toggling one container (not
        # each card) keeps every card's collapsed state and the gesture
        # cheat-sheet's own visibility intact across tab switches.
        self._tools_box = None
        if self._tool_cards:
            self._tools_box = pn.Column(*self._tool_cards,
                                        sizing_mode="stretch_width",
                                        visible=self.tabs.active in self._tool_tabs)
            self.tabs.param.watch(
                lambda e: setattr(self._tools_box, "visible",
                                  e.new in self._tool_tabs), "active")
        self._controls = self._make_controls(cmap, perc)

    def _pick_shot(self, i: int):
        """Layout-map tap: select the shot and jump to the shot-gather tab."""
        self.browser.set_shot(i)
        if self._shot_tab is not None:
            self.tabs.active = self._shot_tab

    def _draw2d(self):
        arr, clim = _process_gather(self.g.data, self.g.dt, self.state)
        self._pane2d.update(arr, clim, y0=self.g.t0, dy=self.g.dt)
        if hasattr(self, "_wt2d"):
            self._wt2d.refresh()

    def _redraw_slices(self):
        """The volume tab follows the filter/gain/flip chain too: volumes up
        to PROC_MAX_BYTES are processed whole (once per chain change) so all
        three slice planes -- including the time slice, which has no time
        axis of its own -- stay consistent. Larger memmaps fall back to raw
        with a visible note (processing them would defeat lazy loading)."""
        g = self.g
        chain = (self.state.get("filter") or self.state.get("gain")
                 or self.state.get("flip"))
        nbytes = g.data.size * g.data.dtype.itemsize
        if chain and g.axes[-1] == "time" and nbytes <= PROC_MAX_BYTES:
            arr, clim = _process_gather(np.asarray(g.data, dtype=np.float32),
                                        g.dt, self.state)
            self._proc_note.visible = False
            self.slices.clim = clim
            self.slices.set_volume(arr)
            return
        self._proc_note.visible = bool(chain) and g.axes[-1] == "time"
        self.slices.clim = self.state["clim"]
        if self.slices.vol is not g.data:
            self.slices.set_volume(g.data)     # back to the raw (lazy) volume
        else:
            self.slices.redraw()

    def _make_controls(self, cmap, perc):
        w_disp = pn.widgets.Select(name="display", options=list(_DISPLAYS),
                                   value="density")
        w_cmap = pn.widgets.Select(name="colormap", options=list(_CMAPS),
                                   value=cmap)
        w_perc = pn.widgets.FloatSlider(name="clip percentile", start=80.0,
                                        end=100.0, step=0.5, value=perc)
        # negates the displayed gathers: wiggle fill lobes and density
        # colors swap (SEG normal <-> reverse); spectra and picks unaffected
        w_flip = pn.widgets.Checkbox(name="flip polarity", value=False)

        def _on_flip(event):
            self.state["flip"] = event.new
            for r in self._redraws:
                r()

        w_flip.param.watch(_on_flip, "value")
        w_disp.param.watch(
            lambda e: [p.set_display(e.new) for p in self._panes], "value")
        w_cmap.param.watch(
            lambda e: [p.set_cmap(e.new) for p in self._panes], "value")

        def _on_perc(event):
            self.state["perc"] = event.new
            self.state["clim"] = robust_clim(self.state["sample"], event.new,
                                             symmetric=self.state["symmetric"])
            for r in self._redraws:
                r()

        w_perc.param.watch(_on_perc, "value_throttled")  # apply on release
        self._proc_note = pn.pane.HTML(
            "<div style='font-size:11px;color:#b45309'>volume too large to "
            "process in memory -- Volume slices shows raw data</div>",
            visible=False)
        widgets = [w_disp, w_flip, w_cmap, w_perc, self._proc_note]
        widgets += self._make_filter_controls()
        widgets += self._make_gain_controls()
        return widgets

    def _make_gain_controls(self):
        """AGC / trace balance on the displayed gathers. With gain active the
        color limits auto-rescale from the processed gather (the raw clim no
        longer applies to ~unit-amplitude output)."""
        self._w_gain = pn.widgets.Select(
            name="gain", value="off",
            options=["off", "trace balance", "AGC"])
        self._w_agcwin = pn.widgets.FloatInput(name="AGC window (s)",
                                               value=0.5, start=0.01, step=0.1,
                                               width=110, visible=False)

        def _apply(event=None):
            kind = self._w_gain.value
            self._w_agcwin.visible = kind == "AGC"
            if kind == "AGC":
                self.state["gain"] = ("agc", float(self._w_agcwin.value))
            elif kind == "trace balance":
                self.state["gain"] = ("balance",)
            else:
                self.state["gain"] = None
            for r in self._redraws:
                r()

        self._w_gain.param.watch(_apply, "value")
        self._w_agcwin.param.watch(_apply, "value")
        return [self._w_gain, self._w_agcwin]

    def _make_filter_controls(self):
        """Zero-phase trapezoid filter (Ormsby-style: linear ramps f1-f2 and
        f3-f4) applied to the displayed gathers. The whole-line volume tab is
        left unfiltered: it would require filtering the full (possibly
        memmapped) volume."""
        self._w_ftype = pn.widgets.Select(
            name="filter", value="off",
            options=["off", "low-pass", "high-pass", "band-pass"])
        mk = lambda n, v: pn.widgets.FloatInput(name=n, value=v, start=0.0,
                                                step=1.0, width=95)
        self._w_f = [mk("f1 (Hz)", 5.0), mk("f2 (Hz)", 10.0),
                     mk("f3 (Hz)", 60.0), mk("f4 (Hz)", 80.0)]
        self._row_lo = pn.Row(*self._w_f[:2], visible=False)   # low-cut ramp
        self._row_hi = pn.Row(*self._w_f[2:], visible=False)   # high-cut ramp

        def _apply(event=None):
            kind = self._w_ftype.value
            self._row_lo.visible = kind in ("high-pass", "band-pass")
            self._row_hi.visible = kind in ("low-pass", "band-pass")
            v = [w.value for w in self._w_f]
            flt = {}
            if self._row_lo.visible:
                flt.update(f1=v[0], f2=v[1])
            if self._row_hi.visible:
                flt.update(f3=v[2], f4=v[3])
            self.state["filter"] = flt or None
            for r in self._redraws:
                r()

        self._w_ftype.param.watch(_apply, "value")
        for w in self._w_f:
            w.param.watch(_apply, "value")
        return [self._w_ftype, self._row_lo, self._row_hi]

    def sidebar(self):
        items = [pn.pane.Markdown(_info_md(self.g)), *self._controls]
        if self._tools_box is not None:
            items += [pn.layout.Divider(margin=(12, 0, 4, 0)),
                      self._tools_box]
        return items

    def panel(self):
        """Plain layout (notebook-friendly)."""
        return pn.Row(pn.Column(*self.sidebar(), width=300),
                      self.tabs, sizing_mode="stretch_width")

    def template(self, title="gathervis"):
        """Served page with a proper header + sidebar."""
        name = self.g.name
        return pn.template.FastListTemplate(
            title=f"{title} — {name}" if name else title,
            sidebar=self.sidebar(), main=[self.tabs],
            accent_base_color="#1a5276", header_background="#1a5276",
            sidebar_width=320)


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
         port=None, address="127.0.0.1", title="gathervis", verbose=True,
         tools="sidebar"):
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
    tools : {'sidebar', 'below'}
        Where the Window / Spectrum / Event-picking / FB-picking cards go.
        'sidebar' (default) stacks them, collapsed, under the display
        controls so the gather keeps the full main column. 'below' restores
        the old row of cards under the figure.
    """
    import time
    t_start = time.perf_counter()
    if view not in (None, "browse", "slices"):
        raise ValueError("view must be None, 'browse' or 'slices'")
    if tools not in ("sidebar", "below"):
        raise ValueError("tools must be 'sidebar' or 'below'")
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

    ws = Workspace(obj, cmap=cmap, perc=perc, view=view, tools=tools)
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
