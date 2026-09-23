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

import html as _html
import itertools

import numpy as np
import panel as pn
from bokeh.events import DocumentReady
from bokeh.models import (BasicTicker, BoxEditTool, ColorBar,
                          ColumnDataSource, CrosshairTool, CustomJS, Div,
                          LabelSet, Legend, LegendItem, LinearColorMapper,
                          PointDrawTool, PolyDrawTool, Range1d, TapTool,
                          TextInput)
from bokeh.plotting import figure

from .core import Gathers, from_array, from_file
from .process import (robust_clim, decimate, quantize, bandpass,
                      agc, trace_balance, pick_first_breaks)
from . import mute as _mute
from . import render as _render
from . import state as _state
from . import velocity as _vel
from . import sortkeys as _sort
from . import survey as _survey
from . import theme as _theme
from .theme import pair as _pair, prop as _prop
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

# Tool sections in the sidebar are one list of expandable rows -- a hairline
# between them, no boxes and no shadows -- so five of them read as one block
# of tools rather than five floating dialogs. Applied per card because Panel
# renders cards into shadow roots, where the page's CSS cannot reach.
_INK, _MUTED, _LINE = _theme.INK, _theme.INK_2, _theme.LINE

_CARD_CSS = """
:host(.card) {
  border: 0;
  border-top: 1px solid #e7ebef;
  border-radius: 0;
  box-shadow: none;
  outline: none;
  background: transparent;
  overflow: visible;
  padding-bottom: 2px;
}
.card-header {
  min-height: 36px;
  padding: 0;
  border: 0;
  border-radius: 0;
  background-color: transparent !important;
  box-shadow: none !important;
}
.card-header:hover .card-title h3 { color: #0b6b73; }
.card-button { color: #7b8794; margin-right: 2px; }
.card-title h3 { margin: 0; font-size: 13px; font-weight: 500; color: #17202a; }
"""

# With tools='below' the cards stand alone under the figure, so each one gets
# a frame of its own.
_TILE_CSS = _CARD_CSS + """
:host(.card) {
  border: 1px solid #d5dbe1;
  border-radius: 8px;
  padding: 0 12px 8px;
}
"""

# The native file picker cannot be relabelled, so it is drawn as the Import
# button it stands for: the browser's own button fills the control, its text
# is hidden, and the label sits on top without catching the click.
_FILE_CSS = """
:host .bk-input-group { display: block; position: relative; }
:host input[type="file"].bk-input {
  width: 100%;
  height: 28px;
  padding: 0;
  font-size: 0;
  color: transparent;
  background: transparent;
  border: 0;
  box-shadow: none;
  cursor: pointer;
}
:host input[type="file"]::file-selector-button {
  width: 100%;
  height: 28px;
  margin: 0;
  font-size: 0;
  background: #ffffff;
  border: 1px solid #d5dbe1;
  border-radius: 6px;
  cursor: pointer;
}
:host input[type="file"]:hover::file-selector-button {
  background: #eef1f4;
  border-color: #bfc8d0;
}
:host .bk-input-group::after {
  content: "Import";
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12.5px;
  font-weight: 500;
  color: #17202a;
  pointer-events: none;
}
"""

# One shape for every control inside a tool card: full width, same rhythm.
# width=None is explicit: several Panel widgets ship a 300 px default that
# would otherwise sit in the param even though stretch_width wins in layout.
_W = dict(sizing_mode="stretch_width", width=None, margin=(4, 0))
_WFILE = dict(_W, stylesheets=[_FILE_CSS])

# cursor readouts under every plot: quiet, and digits that do not jitter
_READOUT_STYLE = {"color": _theme.INK_2, "font-size": "12px",
                  "font-family": _theme.FONT,
                  "font-variant-numeric": "tabular-nums",
                  "white-space": "nowrap"}


def _ensure_ext():
    global _ext_done
    if not _ext_done:
        pn.extension("plotly", raw_css=[_PLOT_CSS])
        pn.config.design = _theme.GathervisDesign
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

# Fold is an integer count on a bin grid, so the readout wants the bin index
# and the count, not an interpolated amplitude.
_FOLD_READOUT_JS = """
const x = cb_obj.x, y = cb_obj.y;
const d = src.data;
if (x == null || y == null || d["image"] == null || d["image"].length == 0) {
  div.text = "&nbsp;"; return;
}
const img = d["image"][0];
const nyr = img.shape[0], nxc = img.shape[1];
const x0 = d["x"][0], y0 = d["y"][0], dw = d["dw"][0], dh = d["dh"][0];
const j = Math.floor((x - x0) / dw * nxc);
const i = Math.floor((y - y0) / dh * nyr);
let txt = "x " + x.toFixed(1) + " &nbsp;&nbsp; y " + y.toFixed(1);
if (i >= 0 && i < nyr && j >= 0 && j < nxc) {
  const v = img[i * nxc + j];
  txt += " &nbsp;&nbsp; bin (" + j + ", " + i + ")"
       + " &nbsp;&nbsp; fold " + (isNaN(v) ? "0" : v.toFixed(0));
}
div.text = txt;
"""


def _as_float_array(data):
    """``data`` in memory as float32. Anything numpy understands goes
    through np.asarray; an index-only lazy array (shape, ndim, dtype and
    ``data[i]``) is read one gather at a time and stacked."""
    if isinstance(data, np.ndarray) or hasattr(data, "__array__"):
        return np.asarray(data, dtype=np.float32)
    return np.stack([np.asarray(data[i], dtype=np.float32)
                     for i in range(int(data.shape[0]))])


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


_GESTURES_HTML = """<div class='gv-help'>
<p><b>Toolbar gestures.</b> Hover the plot to show its toolbar. Click a tool
to switch it on, click it again to switch it off.</p>
<dl>
<dt>Box window</dt><dd>SHIFT+drag to draw, or click, move and click. Drag to
move it; tap it and press <kbd>BACKSPACE</kbd> to delete it.</dd>
<dt>Polygon</dt><dd>Click each vertex; double-click or <kbd>ESC</kbd> closes
it. Drag to move, tap and <kbd>BACKSPACE</kbd> to delete.</dd>
<dt>Picks, mutes</dt><dd>Tap to add a point, drag to move it, tap and
<kbd>BACKSPACE</kbd> to delete it.</dd>
<dt>Zoom</dt><dd>The x and y wheel tools zoom one axis at a time. The
crosshair tool toggles cursor lines.</dd>{keys}
</dl></div>"""

# Only where there are shots to step through.
_KEYS_HTML = """
<dt>Keys</dt><dd><kbd>&larr;</kbd> <kbd>&rarr;</kbd> previous / next shot,
ten at a time with <kbd>Shift</kbd>; <kbd>Home</kbd> <kbd>End</kbd> first and
last shot. Ignored while you type in a field.</dd>"""


def _gesture_help(shots=True, sidebar=True):
    """The gesture and shortcut sheet, as one more collapsible tool section.

    Returns ``(card, body)``: the card to place, and the sheet inside it.
    """
    html = _GESTURES_HTML.format(keys=_KEYS_HTML if shots else "")
    body = pn.pane.HTML(html, sizing_mode="stretch_width",
                        margin=(2, 0, 10, 0))
    card = pn.Card(body, title="Gestures and keys", **_card_style(sidebar))
    return card, body


def _card_style(sidebar):
    """Geometry for a tool card, depending on where it is hosted.

    In the sidebar the cards are rows of one list, full width and collapsed,
    so the sidebar stays scannable. Below the figure they are 260 px tiles
    flowing in a FlexBox, open.
    """
    if sidebar:
        return dict(stylesheets=[_CARD_CSS], header_color=_INK,
                    sizing_mode="stretch_width", margin=0, collapsed=True)
    return dict(stylesheets=[_TILE_CSS], header_color=_INK, width=260,
                margin=(6, 6), collapsed=False)


def _sub(title):
    """A sub-heading inside a tool card."""
    return pn.pane.HTML(f"<div class='gv-sub'>{title}</div>",
                        sizing_mode="stretch_width", margin=(2, 0, 0, 0))


def _slug(name, default="gather", maxlen=40):
    """A dataset name reduced to a safe file-name stem (for downloads)."""
    import os
    import re
    stem = os.path.basename(str(name or "")).rsplit(".", 1)[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem[:maxlen] or default


def _hint(html):
    """The one small-print style: same size, colour and inset everywhere."""
    return _theme.hint(html)


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
    div = Div(text="&nbsp;", height=18, styles=_READOUT_STYLE)
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


# ---------------------------------------------------------------------------
# Keyboard: step through shots, move volume slices
# ---------------------------------------------------------------------------
# Only keys that move through the data are claimed -- the arrows, Home and
# End, and the digits 1-3 that choose a volume slice. Everything else
# (colormap, clip, polarity, gain, fullscreen, tabs) is set once and stays
# with its widget: a key claimed is a key swallowed, and swallowing keys
# nobody presses only costs the user their browser's own shortcuts. What a
# claimed key does depends on the tab (see Workspace._on_key); Shift makes
# any step a big one and travels as an "S-" prefix.
#
# The listener sits on ``document``: a bokeh canvas never takes focus, so
# anything attached to the plot itself would only fire after the user first
# clicked something. Three details make that safe:
#
#   * ``composedPath()[0]`` instead of ``e.target`` -- Panel renders widgets
#     into shadow roots, where the retargeted ``e.target`` is the *host*
#     element, so an arrow key walking the caret along "go to shot" would
#     otherwise flip the shot underneath as well;
#   * only the two arrow keys are swallowed, leaving SHIFT / ESC /
#     BACKSPACE (and every browser shortcut) to bokeh's own edit tools;
#   * held-down keys are throttled to ~8/s, so leaning on the arrow key
#     cannot queue up a hundred server-side redraws.
#
# The press reaches Python through a hidden bokeh TextInput: writing its
# value in JS syncs to the server like any other model property. A Panel
# ReactiveHTML component would read better, but it makes the browser
# resolve a custom data model, and if that resolution fails the *whole*
# document fails to load -- a blank page instead of a dead shortcut. Plain
# models only, so a broken key binding can never cost more than the keys.
_KEYS_JS = """
if (window._gvOnKey) { document.removeEventListener('keydown', window._gvOnKey); }
window._gvSeq = window._gvSeq || 0;
window._gvOnKey = function (e) {
  if (e.ctrlKey || e.metaKey || e.altKey) { return; }
  const t = (e.composedPath ? e.composedPath()[0] : e.target) || e.target;
  const tag = (t && t.tagName) ? t.tagName.toUpperCase() : '';
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' ||
      (t && t.isContentEditable)) { return; }
  if (keys.indexOf(e.key) < 0) { return; }
  e.preventDefault();
  const now = (window.performance || Date).now();
  if (e.repeat && now - (window._gvKeyAt || 0) < 120) { return; }
  window._gvKeyAt = now;
  chan.value = (e.shiftKey ? 'S-' : '') + e.key + '|' + (++window._gvSeq);
};
document.addEventListener('keydown', window._gvOnKey);
"""


# arrows step (a shot, or the active volume slice); Home / End jump to
# either end; 1-3 choose the volume slice the arrows move
_SHOT_STEPS = {"ArrowRight": 1, "ArrowLeft": -1}
_KEYS = ("ArrowLeft", "ArrowRight", "Home", "End", "1", "2", "3")


class ImagePane:
    """A 2-D (space, time) panel with two display modes on one bokeh figure:

    * ``density`` -- quantized uint8 image through a colormap (the default);
    * ``wiggle``  -- wiggle traces with positive-lobe fill (variable area).

    ``flip_y=True`` gives the seismic convention (time increasing downward).
    """

    def __init__(self, xlabel="trace", ylabel="time (s)", flip_y=True,
                 cmap="gray", height=520, match_aspect=False):
        self.mapper = LinearColorMapper(palette=_palette(cmap), low=0, high=255)
        self.cds = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
        self.display = "density"
        self.cmap = cmap                # kept so the download uses them too
        self.flip_y = bool(flip_y)
        self._last = None               # args of the latest update()
        self._extent = None             # (x0, x1, y0, y1) currently bounded
        # Explicit ranges rather than bokeh's auto-ranging ones. Auto-ranging
        # decides in the browser whether it still follows the data after a
        # pan, and the answer matters here: step to the next shot with the
        # panel dragged off to one side and the gather has to come back into
        # view, not stay wherever the last drag left it.
        fig = figure(height=height, sizing_mode="stretch_width",
                     x_axis_label=xlabel, y_axis_label=ylabel,
                     match_aspect=match_aspect,
                     x_range=Range1d(0.0, 1.0),
                     y_range=Range1d(1.0, 0.0) if flip_y else Range1d(0.0, 1.0),
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
        # Receiver-line separators for an 'as recorded' 3-D shot: N lines end
        # to end read as N hyperbolas, and the eye needs to know where one
        # line stops. Drawn as data-space segments (not Spans) so they are one
        # glyph with a CDS, updated like everything else on this pane.
        self._cds_sep = ColumnDataSource(dict(xs=[], ys=[]))
        fig.multi_line(xs="xs", ys="ys", source=self._cds_sep,
                       line_color="#5f6368", line_width=1, line_dash="dashed",
                       line_alpha=0.55)
        self._cds_seplab = ColumnDataSource(dict(x=[], y=[], text=[]))
        fig.add_layout(LabelSet(x="x", y="y", text="text",
                                source=self._cds_seplab,
                                text_font_size="9px", text_color="#5f6368",
                                text_align="center", y_offset=3))
        _theme.style_figure(fig)
        self.readout = Div(text="&nbsp;", height=18, styles=_READOUT_STYLE)
        # Wiggle mode cannot draw a thousand traces legibly and will not try,
        # so it subsamples -- and then says so. A panel quietly showing half
        # the gather is worse than one that shows all of it badly, because
        # you cannot tell which you are looking at.
        self.subsampled = Div(text="", height=0,
                              styles=dict(_READOUT_STYLE, color=_theme.WARN))
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

        # -- view controls ------------------------------------------------
        # A unique class per pane so fullscreen targets this plot, not the
        # first one on the page (a workspace can hold several).
        self._cls = f"gv-plot-{next(_PLOT_UID)}"
        self.w_height = _theme.IntSlider(
            name="height", start=240, end=2400, step=20, value=height,
            width=128, margin=(4, 8, 4, 0))
        self.w_height.param.watch(
            lambda e: setattr(self.figure, "height", int(e.new)), "value")
        self._height_label = pn.pane.HTML(
            "<div class='gv-label'>Height</div>", margin=(4, 8, 4, 0),
            align="center")
        icon = lambda cls, name, text: _theme.icon_kw(cls, name, text)
        btn = dict(button_type="light", width=32, margin=(4, 1))
        self.w_reset = pn.widgets.Button(
            **icon(pn.widgets.Button, "reset", "whole"), **btn,
            description="Show the whole gather")
        self.w_reset.on_click(lambda e: self.reset_view())
        self.w_fit = pn.widgets.Button(
            **icon(pn.widgets.Button, "fit", "fit"), **btn,
            description="Grow the panel to the bottom of the window")
        self.w_fullscreen = pn.widgets.Button(
            **icon(pn.widgets.Button, "fullscreen", "fullscreen"), **btn,
            description="Fullscreen (ESC to leave)")
        self.js_fit = self.w_fit.js_on_click(
            args=dict(sl=self.w_height, chrome=_FIT_CHROME_PX, cls=self._cls),
            code=_FIT_JS)
        self.js_fullscreen = self.w_fullscreen.js_on_click(
            args=dict(sl=self.w_height, cls=self._cls), code=_FULLSCREEN_JS)
        # How much of the gather actually reaches the browser. Both default
        # to a budget rather than to everything, because a 5000 x 12000
        # gather is 60 MB of uint8 per redraw and the tab stops responding
        # long before it finishes -- but the budget is a default, not a
        # decision, so both are offered and both say when they bite. They
        # are display settings, so the workspace shows them in the sidebar,
        # each only in the display mode it applies to.
        self.w_res = pn.widgets.Select(
            name="Resolution", options={"Auto": "auto", "Full": "full"},
            value="auto", **_W,
            description="Auto caps the image at 1600 px per axis; Full "
                        "ships every trace and every sample")
        self.w_wig_n = pn.widgets.IntInput(
            name="Traces", value=MAX_WIGGLE[0], start=2, end=20000,
            step=16, **_W,
            description="How many traces the wiggle display draws: more "
                        "for a denser panel, fewer for a cleaner one")
        self.w_res.param.watch(lambda e: self.redraw(), "value")
        self.w_wig_n.param.watch(lambda e: self.redraw(), "value")
        # the one view button with words on it: an arrow alone does not say
        # that this is the full-resolution image, not a screenshot
        dl = dict(icon(pn.widgets.FileDownload, "download", ""),
                  label="Full image")
        self.w_download = pn.widgets.FileDownload(
            callback=self._export_png, filename="gather.png", **dl,
            button_type="light", margin=(4, 1), disabled=True,
            description="Export PNG")

    def size_controls(self):
        """Height, then reset / fit / fullscreen / export, as one cluster.

        ``reset`` resets the *data* range and ``fit`` resizes the *panel* --
        two different things that both feel like "make it right again", so
        they sit next to each other.
        """
        return pn.Row(self._height_label, self.w_height, self.w_reset,
                      self.w_fit, self.w_fullscreen, self.w_download,
                      margin=0, css_classes=["gv-viewbar"])

    def frame(self, *extra, controls=True, nav=()):
        """The plot, its readout and ``extra``, in the fullscreen container.

        ``nav`` -- whatever moves through the data, e.g. the shot slider --
        shares the toolbar row with the view controls, so both stay within
        reach in fullscreen.
        """
        items = []
        if controls or nav:
            bar = list(nav) if nav else [pn.layout.HSpacer()]
            if controls:
                bar.append(self.size_controls())
            items.append(pn.Row(*bar, margin=(6, 0, 4, 0),
                                sizing_mode="stretch_width",
                                css_classes=["gv-toolbar"]))
        items += [self.figure, self.readout, self.subsampled, *extra]
        return pn.Column(*items, css_classes=["gv-plot", self._cls],
                         sizing_mode="stretch_width")

    def redraw(self):
        """Re-render the latest data, e.g. after a resolution change."""
        if self._last is not None:
            self.update(*self._last)

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
        nx, ny = arr2d.shape
        self._set_extent(x0, x0 + nx * dx, y0, y0 + ny * dy, dx)
        if self.display == "wiggle":
            self._update_wiggle(arr2d, clim, x0, dx, y0, dy)
        else:
            self._update_image(arr2d, clim, x0, dx, y0, dy)
        self._sync_download()

    def _set_extent(self, x0, x1, y0, y1, dx):
        """Fence the view to the data, and re-fit when the data changes shape.

        Two things, because they are the same decision seen from either end.
        ``bounds`` stops a pan or a zoom-out from carrying the gather off the
        edge of the panel -- there is nothing out there to look at, and a
        panel showing nothing but background gives no hint of which way to
        drag to get back.

        Re-fitting only when the *extent* changes is what keeps that from
        being annoying. Stepping through shots of the same size leaves the
        zoom where it was, which is the whole point of zooming in on a
        feature and walking the shots past it. A different sort, a different
        gather, a different dataset -- those change the extent, and then the
        view goes back to the whole thing rather than to whatever corner of
        the old one you happened to be in.
        """
        extent = (float(x0), float(x1), float(y0), float(y1))
        if not all(np.isfinite(extent)) or x1 <= x0 or y1 <= y0:
            return
        fig = self.figure
        # Wiggle lobes swing up to two trace spacings past the edge traces,
        # so the fence sits that far out rather than on the image edge.
        pad = 2.0 * abs(float(dx))
        fig.x_range.bounds = (extent[0] - pad, extent[1] + pad)
        fig.y_range.bounds = (extent[2], extent[3])
        if extent == self._extent:
            return
        self._extent = extent
        self.reset_view()

    def reset_view(self):
        """Show the whole panel, whatever it is currently zoomed to."""
        if self._extent is None:
            return
        x0, x1, y0, y1 = self._extent
        fig = self.figure
        fig.x_range.start, fig.x_range.end = x0, x1
        fig.y_range.start, fig.y_range.end = (y1, y0) if self.flip_y else (y0, y1)
        # so the toolbar's reset goes to the whole panel too, not to whatever
        # the range happened to be when the figure was first rendered
        fig.x_range.reset_start, fig.x_range.reset_end = x0, x1
        fig.y_range.reset_start, fig.y_range.reset_end = fig.y_range.start, fig.y_range.end

    def set_separators(self, xs, labels=None, label_x=None):
        """Dashed verticals at ``xs``, spanning the panel; ``labels`` (drawn
        at ``label_x``, defaulting to ``xs``) annotate the blocks between.

        Call it after ``update``: the segments take their vertical extent
        from whatever the panel is currently showing.
        """
        xs = np.asarray(xs, dtype=float).ravel()
        if self._last is None or not xs.size:
            self._cds_sep.data = dict(xs=[], ys=[])
            self._cds_seplab.data = dict(x=[], y=[], text=[])
            return
        arr, _clim, _x0, _dx, y0, dy = self._last
        y1 = y0 + dy * arr.shape[1]
        self._cds_sep.data = dict(xs=[[x, x] for x in xs],
                                  ys=[[y0, y1] for _ in xs])
        if labels is None:
            self._cds_seplab.data = dict(x=[], y=[], text=[])
            return
        lx = xs if label_x is None else np.asarray(label_x, dtype=float).ravel()
        self._cds_seplab.data = dict(x=[float(v) for v in lx],
                                     y=[float(y0)] * len(lx),
                                     text=[str(t) for t in labels])

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
            f"Gather too large to export ({w * h / 1e6:.0f} MP)" if over
            else f"Export PNG at full resolution, {w} × {h} px")

    def _note(self, text):
        self.subsampled.text = text
        self.subsampled.height = 18 if text else 0

    def _say_subsampled(self, shown, total, step):
        """Tell the user when wiggle is not drawing every trace."""
        if shown >= total:
            self._note("")
            return
        self._note(f"Wiggle draws {shown} of {total} traces, every "
                   f"{step}. Raise <b>Traces</b> in the sidebar to draw them "
                   f"all, or switch to density.")

    def _say_pixels(self, shown, total):
        """Tell the user when density is shipping fewer pixels than traces."""
        if tuple(shown) == tuple(total):
            self._note("")
            return
        self._note(f"Density shows {shown[0]} x {shown[1]} of {total[0]} x "
                   f"{total[1]} samples. Set <b>Resolution</b> to Full for "
                   f"every one ({total[0] * total[1] / 1e6:.1f} MB per "
                   f"redraw).")

    def _update_image(self, arr2d, clim, x0, dx, y0, dy):
        nx, ny = arr2d.shape
        src = arr2d if self.w_res.value == "full" else decimate(arr2d, MAX_PX)
        self._say_pixels(src.shape, arr2d.shape)
        img = quantize(src, clim)
        self.cds.data = dict(image=[img.T], x=[x0], y=[y0],
                             dw=[nx * dx], dh=[ny * dy],
                             lo=[float(clim[0])], hi=[float(clim[1])])

    def _update_wiggle(self, arr2d, clim, x0, dx, y0, dy):
        """Wiggle + variable-area fill. Amplitudes are normalized by the clim
        (the clip-percentile slider doubles as wiggle gain), excursion is one
        displayed trace spacing, clipped at +-2 spacings."""
        n_wig = max(2, int(self.w_wig_n.value))
        st = max(1, -(-arr2d.shape[0] // n_wig))           # ceil-div strides,
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
        self._say_subsampled(a.shape[0], arr2d.shape[0], st)


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
        fig.add_tools(
            _theme.set_tool_icon(BoxEditTool(
                renderers=[r_rect], empty_value=_WIN_COLORS[0],
                description="Analysis window: box"), "box"),
            _theme.set_tool_icon(PolyDrawTool(
                renderers=[r_poly], empty_value=_WIN_COLORS[0],
                description="Analysis window: polygon"), "poly"))

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
        _theme.style_figure(sfig)
        sfig.legend.background_fill_alpha = 0.85
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
        _theme.style_figure(kfig, grid=False)
        _add_crosshair(kfig)
        kfig.visible = False
        self.fk_fig = kfig
        self.fk_readout = _coord_readout(kfig, "k", "1/trace", "f", "Hz",
                                         xdigits=3, ydigits=1)
        self.fk_readout.visible = False

        self.w_btn = pn.widgets.Button(name="Compute spectrum",
                                       button_type="primary", **_W)
        self.w_btn.on_click(lambda e: self.compute())
        self.w_fk = pn.widgets.Button(
            name="f-k of first window", **_W,
            description="2-D f-k amplitude spectrum of the first window")
        self.w_fk.on_click(lambda e: self.compute_fk())
        self.w_export = pn.widgets.FileDownload(callback=self._export,
                                                filename="windows.json",
                                                label="Export", **_W)
        self.w_import = pn.widgets.FileInput(accept=".json", **_WFILE)
        self.w_import.param.watch(self._import, "value")
        self.w_clear = pn.widgets.Button(
            name="Clear windows", **_W,
            description="Remove every drawn window, and its spectra")
        self.w_clear.on_click(lambda e: self.clear_windows())
        self.w_scale = pn.widgets.RadioButtonGroup(
            name="Scale", value="amplitude", **_W,
            options={"Linear": "amplitude", "dB": "dB"},
            description="Linear amplitude, normalised to the window's peak, "
                        "shows where the energy is; dB stretches the weak "
                        "tail and the noise floor")
        self.w_scale.param.watch(lambda e: self.refresh(), "value")
        self.w_traces = pn.widgets.Select(
            name="Traces", value="per trace", **_W,
            options={"Per trace": "per trace", "Mean": "mean",
                     "Middle trace": "middle trace"},
            description="How the window's traces reach the plot: each one "
                        "drawn, their magnitudes averaged, or the middle one")
        self.w_traces.param.watch(lambda e: self.refresh(), "value")
        self.w_size = pn.widgets.RadioButtonGroup(
            name="Panel size", value="M", **_W, options=list(_SPEC_SIZES),
            description="Height of the spectrum panels")
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
        """Analysis-window I/O (drawing happens with the toolbar tools)."""
        key = ("window", sidebar)
        if key not in self._cards:
            self._cards[key] = pn.Card(
                _hint("Draw with the <b>box</b> or <b>polygon</b> tool in the "
                      "plot toolbar. Windows stay in place while you browse "
                      "shots."),
                self.w_clear,
                _pair(self.w_import, self.w_export),
                title="Analysis windows", **_card_style(sidebar))
        return self._cards[key]

    def spectrum_card(self, sidebar=False):
        """Spectrum computations on the drawn windows."""
        key = ("spectrum", sidebar)
        if key not in self._cards:
            self._cards[key] = pn.Card(
                self.w_btn, self.w_traces,
                _prop("Scale", self.w_scale),
                _prop("Panel size", self.w_size),
                self.w_fk,
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
        pane.figure.add_tools(_theme.set_tool_icon(
            PointDrawTool(renderers=[r], description="Pick events"), "pick"))
        self.cds.on_change("data", lambda a, o, n: self._sync_line())
        self._store = {}          # shot index -> (trace indices, times)
        self._order = None        # column -> trace; None = the two coincide
        self._shot = 0
        self.w_export = pn.widgets.FileDownload(callback=self._export,
                                                filename="picks.csv",
                                                label="Export", **_W)
        self.w_import = pn.widgets.FileInput(accept=".csv", **_WFILE)
        self.w_import.param.watch(self._import, "value")
        self.w_snap = pn.widgets.Select(
            name="Snap", value="off", **_W,
            options={"Off": "off", "Peak": "peak", "Trough": "trough",
                     "|max|": "|max|"},
            description="Snap picks to the nearest extremum of their trace, "
                        "searching 30 ms either side")
        self.w_sta = pn.widgets.FloatInput(name="STA (s)", value=0.02,
                                           start=0.001, step=0.01, **_W,
                                           description="Short-term window")
        self.w_lta = pn.widgets.FloatInput(name="LTA (s)", value=0.2,
                                           start=0.01, step=0.05, **_W,
                                           description="Long-term window")
        self.w_thr = pn.widgets.FloatInput(name="Trigger ratio", value=4.0,
                                           start=1.1, step=0.5, **_W,
                                           description="STA/LTA ratio that "
                                                       "counts as an arrival")
        self.w_auto = pn.widgets.Button(
            name="Auto pick (STA/LTA)", button_type="primary", **_W,
            description="First breaks of the current shot, from the "
                        "displayed (filtered) gather; refine by dragging or "
                        "snapping")
        self.w_auto.on_click(lambda e: self.auto_pick())
        self.w_clear = pn.widgets.Button(
            name="Clear shot", **_W,
            description="Remove the picks on the current shot")
        self.w_clear.on_click(lambda e: self.clear_shot())
        self.w_clear_all = pn.widgets.Button(
            name="Clear all", **_W,
            description="Remove the picks of every shot")
        self.w_clear_all.on_click(lambda e: self.clear_all())
        self.w_clear_fb = pn.widgets.Button(
            name="Clear picks", **_W,
            description="Remove the current shot's picks, to retry with "
                        "other STA/LTA settings")
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

    # -- trace <-> column ---------------------------------------------------
    # Picks are stored against the *trace* they belong to, in the dataset's
    # own flat numbering, never against the screen column -- re-sorting a 3-D
    # shot moves every column, and a first break belongs to a geophone, not
    # to a position on a panel. ``order[column] = trace``; with no order set
    # (a 2-D line, which has only one arrangement) the two coincide.
    def set_order(self, order):
        """Declare which trace each column currently shows.

        Called by the browser after a re-sort: the picks on display are
        banked against their traces under the old order and laid out again
        under the new one, so picks on traces the new arrangement leaves out
        are hidden rather than lost.
        """
        order = None if order is None else np.asarray(order, dtype=np.int64)
        if self._order is not None and order is not None \
                and np.array_equal(self._order, order):
            return
        self._flush()                       # bank under the old order
        self._order = order
        self._load()

    def _trace_of(self, x) -> int:
        i = int(round(float(x)))
        if self._order is None:
            return i
        return int(self._order[i]) if 0 <= i < self._order.shape[0] else -1

    def _columns_of(self, traces):
        """Where each stored trace sits now; -1 when it is not on screen."""
        traces = np.asarray(traces, dtype=float)
        if self._order is None:
            return traces
        n = int(self._order.max(initial=-1)) + 1
        where = np.full(n, -1, dtype=np.int64)
        where[self._order] = np.arange(self._order.shape[0])
        idx = np.rint(traces).astype(np.int64)
        out = np.where((idx >= 0) & (idx < n), where[np.clip(idx, 0, n - 1)], -1)
        return out.astype(float)

    def _load(self):
        """Put this shot's stored picks on screen under the current order."""
        traces, times = self._store.get(self._shot, ([], []))
        cols = self._columns_of(traces)
        keep = cols >= 0
        self.cds.data = dict(x=cols[keep].tolist(),
                             y=np.asarray(times, dtype=float)[keep].tolist())

    def _flush(self):
        """Bank what is on screen, plus whatever this shot has off screen."""
        shown = {self._trace_of(x): float(y)
                 for x, y in zip(self.cds.data["x"], self.cds.data["y"])}
        shown.pop(-1, None)
        traces, times = self._store.get(self._shot, ([], []))
        cols = self._columns_of(traces)
        merged = {int(t): float(v)                      # off-screen picks stay
                  for t, v, c in zip(traces, times, cols) if c < 0}
        merged.update(shown)
        if merged:
            items = sorted(merged.items())
            self._store[self._shot] = ([float(t) for t, _ in items],
                                       [v for _, v in items])
        else:
            self._store.pop(self._shot, None)

    def set_shot(self, i: int):
        """Swap the displayed picks to shot ``i`` (saving the current ones)."""
        i = int(i)
        if i == self._shot:
            return
        self._flush()
        self._shot = i
        self._load()

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
        self._store = store          # the CSV's traces are trace indices,
        self._load()                 # so they lay out under the current sort

    def picking_card(self, sidebar=False):
        """Manual event picking: snap refinement + pick I/O."""
        key = ("picking", sidebar)
        if key not in self._cards:
            self._cards[key] = pn.Card(
                _hint("With the <b>pick</b> tool: tap adds a pick, drag moves "
                      "it, tap and BACKSPACE deletes it. Picks belong to "
                      "their shot."),
                self.w_snap,
                _pair(self.w_clear, self.w_clear_all),
                _pair(self.w_import, self.w_export),
                title="Event picking", **_card_style(sidebar))
        return self._cards[key]

    def fb_card(self, sidebar=False):
        """Automatic first-break picking (gapped STA/LTA)."""
        key = ("fb", sidebar)
        if key not in self._cards:
            self._cards[key] = pn.Card(
                self.w_sta, self.w_lta, self.w_thr, self.w_auto,
                self.w_clear_fb,
                title="First breaks", **_card_style(sidebar))
        return self._cards[key]

    def controls(self, sidebar=False):
        return pn.FlexBox(self.picking_card(sidebar), self.fb_card(sidebar))

    def panel(self):
        return self.controls()


def _display_controls(state, panes, redraw, cmap, perc):
    """Shared display-mode + cmap + clim-percentile widgets.
    ``state['clim']`` is read by redraw()."""
    w_disp = pn.widgets.RadioButtonGroup(
        name="display", options={"Density": "density", "Wiggle": "wiggle"},
        value="density", width=180, margin=(4, 16, 4, 0))
    w_cmap = pn.widgets.Select(name="Colormap", options=list(_CMAPS),
                               value=cmap, width=250, margin=(4, 16, 4, 0))
    w_perc = _theme.FloatSlider(name="clip percentile", start=80.0, end=100.0,
                                step=0.5, value=perc, width=170, margin=(4, 0))
    extra = []
    if panes and hasattr(panes[0], "w_res"):
        p = panes[0]
        for w in (p.w_res, p.w_wig_n):
            w.param.update(sizing_mode="fixed", width=200, margin=(4, 16, 4, 0))
        p.w_wig_n.visible = False
        extra = [p.w_res, p.w_wig_n]

    def _on_disp(event):
        for p in panes:
            p.set_display(event.new)
        w_cmap.visible = event.new == "density"
        if extra:
            extra[0].visible = event.new == "density"
            extra[1].visible = event.new == "wiggle"

    def _on_cmap(event):
        for p in panes:
            p.set_cmap(event.new)

    def _on_perc(event):
        state["clim"] = robust_clim(state["sample"], event.new)
        redraw()

    w_disp.param.watch(_on_disp, "value")
    w_cmap.param.watch(_on_cmap, "value")
    w_perc.param.watch(_on_perc, "value_throttled")   # apply on release
    return pn.Row(w_disp, w_cmap, *extra, _prop("Clip", w_perc),
                  sizing_mode="stretch_width", margin=0)


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------
def view_gather(arr2d, dt=1.0, t0=0.0, cmap="gray", perc=98.0, title=""):
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


class MuteTool:
    """Top and bottom mute lines drawn on a gather panel.

    Two toolbar point tools, one per line: tap to add a node, drag to move
    it, BACKSPACE deletes the selected one. The line through the nodes is
    drawn across every trace, so what you see is the cut that will actually
    be applied, not just the handles you placed.

    **Nodes are stored against offset, not against the column they were
    dropped on.** A mute is an offset-dependent thing, so a line stored that
    way means the same cut on the next gather even though it has different
    traces in a different order -- and it survives re-sorting the panel,
    which a line pinned to column numbers would not. Without geometry there
    are no offsets and the line falls back to column index, which is honest
    but does not travel.

    **One line for everything, with exceptions.** The scope switch decides
    where an edit goes: *all gathers* writes the default that every gather
    inherits, *this gather* gives the current one a line of its own. Most
    gathers want the same mute and a few do not, so that is the shape --
    nothing is interpolated between gathers, because a viewer should not
    invent a mute for a gather nobody has looked at.

    Turning *apply* on shows the gather muted. That is not decoration: you
    cannot tell whether a cut is in the right place without seeing what it
    removes.
    """

    _TOP = "#1e88e5"
    _BOTTOM = "#f4511e"

    def __init__(self, pane: ImagePane, mutes=None, domain="offset"):
        self.pane = pane
        self.mutes = mutes if mutes is not None else _mute.Mutes(domain)
        self._cards = {}
        self._busy = False
        self._key = 0                 # the gather being shown
        self._x = None                # (ncol,) domain value per column
        # Where each node was last drawn, and the domain value it stands
        # for. A node snaps to the nearest column to be drawn, and this
        # gather's columns rarely sit at exactly the offsets the line was
        # built from -- so reading the column back would quietly move the
        # node every time anything else on the line was touched. Nodes that
        # did not move in x keep the value they came in with.
        self._placed = {k: (np.zeros(0), np.zeros(0)) for k in _mute.KINDS}
        self.on_change = []           # hooks: called when a line is edited

        self.cds = {}                 # kind -> draggable nodes (column, time)
        self._cds_line = {}           # kind -> the rendered cut
        for kind, colour in ((_mute.TOP, self._TOP), (_mute.BOTTOM, self._BOTTOM)):
            self._cds_line[kind] = ColumnDataSource(dict(x=[], y=[]))
            pane.figure.line("x", "y", source=self._cds_line[kind],
                             line_width=2, line_color=colour, line_alpha=0.85)
            self.cds[kind] = ColumnDataSource(dict(x=[], y=[]))
            r = pane.figure.scatter("x", "y", source=self.cds[kind], size=10,
                                    marker="square", fill_color=colour,
                                    fill_alpha=0.9, line_color="white",
                                    line_width=1.5)
            pane.figure.add_tools(_theme.set_tool_icon(
                PointDrawTool(renderers=[r],
                              description=f"{kind.capitalize()} mute"),
                f"mute_{kind}"))
            self.cds[kind].on_change(
                "data", lambda a, o, n, k=kind: self._on_edit(k))

        self._build_widgets()
        self.status = pn.pane.HTML("", sizing_mode="stretch_width")

    # ------------------------------------------------------------------ UI
    def _build_widgets(self):
        self.w_scope = pn.widgets.RadioButtonGroup(
            name="Edits go to", value="all gathers", button_type="default",
            options={"All gathers": "all gathers",
                     "This gather": "this gather"}, **_W)
        self.w_apply = pn.widgets.Checkbox(name="Apply mute", value=False, **_W)
        self.w_taper = pn.widgets.FloatInput(
            name="Taper (s)", value=0.04, start=0.0, step=0.01, **_W,
            description="A hard cut leaves a step along the mute, and a step "
                        "is broadband energy on a coherent trajectory: it "
                        "shows up as an event that was never in the ground")
        self.w_cut_v = pn.widgets.FloatInput(
            name="Velocity (m/s)", value=1500.0, start=1.0, step=100.0, **_W,
            description="Seed a top mute along offset / velocity + delay")
        self.w_cut_pad = pn.widgets.FloatInput(name="Delay (s)", value=0.05,
                                               start=0.0, step=0.01, **_W)
        self.w_seed = pn.widgets.Button(name="Seed top mute", **_W)
        self.w_seed.on_click(lambda e: self.seed_linear())
        self.w_promote = pn.widgets.Button(
            name="Make default", **_W,
            description="Tuned it here and want it on every gather")
        self.w_promote.on_click(lambda e: self.promote())
        self.w_revert = pn.widgets.Button(
            name="Revert", **_W,
            description="Drop this gather's own line; it goes back to the "
                        "default")
        self.w_revert.on_click(lambda e: self.revert())
        self.w_clear = pn.widgets.Button(name="Clear both lines", **_W)
        self.w_clear.on_click(lambda e: self.clear())
        self.w_export = pn.widgets.FileDownload(
            callback=self._export, filename="mutes.json", label="Export",
            **_W)
        self.w_import = pn.widgets.FileInput(accept=".json", **_WFILE)
        self.w_import.param.watch(self._import, "value")
        for w in (self.w_apply, self.w_taper):
            w.param.watch(lambda e: self._fire(), "value")

    # -------------------------------------------------------------- domain
    def set_gather(self, key, x=None):
        """Show the lines in force for gather ``key``.

        ``x`` is the domain value of every column -- offsets when there is
        geometry, otherwise column indices. It changes with the sort, so the
        browser passes it on every redraw.
        """
        if x is not None:
            self._x = np.asarray(x, dtype=np.float64).reshape(-1)
        self._key = key
        self._load()

    def _column_of(self, value):
        """The column whose domain value is nearest ``value``."""
        return int(np.argmin(np.abs(self._x - float(value))))

    def _domain_of(self, column):
        """The domain value of the column nearest ``column``."""
        j = int(np.clip(round(float(column)), 0, self._x.size - 1))
        return float(self._x[j])

    # ------------------------------------------------------------ drawing
    def _load(self):
        """Put the stored lines onto the panel, in this panel's columns."""
        if self._x is None or not self._x.size:
            return
        self._busy = True
        try:
            for kind in _mute.KINDS:
                nodes = self.mutes.line(self._key, kind)
                if not len(nodes):
                    self.cds[kind].data = dict(x=[], y=[])
                    self._placed[kind] = (np.zeros(0), np.zeros(0))
                else:
                    cols = np.array([float(self._column_of(v))
                                     for v in nodes[:, 0]])
                    self.cds[kind].data = dict(
                        x=[float(c) for c in cols],
                        y=[float(t) for t in nodes[:, 1]])
                    self._placed[kind] = (cols, nodes[:, 0].copy())
                self._render(kind)
        finally:
            self._busy = False
        self._refresh_status()

    def _render(self, kind):
        """Draw the cut through every column, not just through the nodes."""
        nodes = self._read(kind)
        if nodes is None or not len(nodes) or self._x is None:
            self._cds_line[kind].data = dict(x=[], y=[])
            return
        times = _mute.evaluate(nodes, self._x)
        self._cds_line[kind].data = dict(
            x=[float(j) for j in range(self._x.size)],
            y=[float(t) for t in times])

    def _read(self, kind):
        """The nodes currently on screen, converted back to the domain.

        A node still sitting on the column it was drawn at keeps the domain
        value it was drawn from; only one that was actually dragged sideways
        is re-derived from its column.
        """
        d = self.cds[kind].data
        if not len(d["x"]) or self._x is None:
            return np.zeros((0, 2))
        cols, values = self._placed[kind]
        free = np.ones(cols.shape, dtype=bool)          # each match used once
        out = []
        for c in d["x"]:
            hit = np.flatnonzero(free & (cols == float(c)))
            if hit.size:
                free[hit[0]] = False
                out.append(float(values[hit[0]]))
            else:
                out.append(self._domain_of(c))
        return np.column_stack([out, [float(t) for t in d["y"]]])

    # ------------------------------------------------------------- editing
    def _on_edit(self, kind):
        if self._busy or self._x is None:
            return
        nodes = self._read(kind)
        self.mutes.set(self._target(), kind, nodes)
        self._placed[kind] = (np.array([float(c) for c in self.cds[kind].data["x"]]),
                              nodes[:, 0].copy() if len(nodes) else np.zeros(0))
        self._render(kind)
        self._refresh_status()
        self._fire()

    def _target(self):
        """Where an edit is written: the default, or this gather."""
        return self._key if self.w_scope.value == "this gather" else None

    def seed_linear(self):
        """Start a top mute from one velocity and one delay."""
        if self._x is None or not self._x.size:
            return
        line = _mute.linear_line(v=float(self.w_cut_v.value),
                                 x_max=float(self._x.max()),
                                 x_min=float(self._x.min()),
                                 pad=float(self.w_cut_pad.value))
        self.mutes.set(self._target(), _mute.TOP, line)
        self._load()
        self._fire()

    def promote(self):
        """Make this gather's lines the default for every gather."""
        self.mutes.promote(self._key)
        self.w_scope.value = "all gathers"
        self._load()
        self._fire()

    def revert(self):
        """Drop this gather's own lines; it goes back to the default."""
        self.mutes.clear(self._key)
        self._load()
        self._fire()

    def clear(self):
        """Remove both lines from whichever scope is being edited."""
        self.mutes.clear(self._target())
        self._load()
        self._fire()

    def _fire(self):
        for hook in self.on_change:
            hook()

    # -------------------------------------------------------------- output
    def apply(self, arr, dt, t0=0.0):
        """Mute ``arr`` with the lines in force, if *apply* is on."""
        if not self.w_apply.value or self._x is None or self._x.size != arr.shape[0]:
            return arr
        top, bottom = (self.mutes.line(self._key, k) for k in _mute.KINDS)
        if not len(top) and not len(bottom):
            return arr
        return _mute.apply(arr, self._x, dt, top=top, bottom=bottom, t0=t0,
                           taper=float(self.w_taper.value))

    def _refresh_status(self):
        own = self.mutes.is_override(self._key)
        where = (f"<b style='color:{self._BOTTOM}'>its own line</b>" if own
                 else "the <b>default</b>")
        n = len(self.mutes)
        extra = (f", and {n} gather{'s' if n != 1 else ''} differ" if n
                 else "")
        unit = "offset" if self.mutes.domain == "offset" else "trace"
        self.status.object = (
            f"<div class='gv-hint'>Gather <b>{self._key}</b> uses {where}"
            f"{extra}. Lines are stored against <b>{unit}</b>.</div>")

    def _export(self):
        import io
        buf = io.StringIO(self.mutes.to_json())
        buf.seek(0)
        return buf

    def _import(self, event):
        raw = event.new
        if isinstance(raw, dict):
            raw = next(iter(raw.values()), None)
        if not raw:
            return
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            self.mutes = _mute.Mutes.from_json(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            self.status.object = (f"<div class='gv-warn'>could not read "
                                  f"that file: {exc}</div>")
            return
        self._load()
        self._fire()

    # --------------------------------------------------------------- cards
    def card(self, sidebar=False):
        key = ("mute", sidebar)
        if key not in self._cards:
            self._cards[key] = pn.Card(
                self.w_apply, self.w_taper,
                _hint("The toolbar has one tool per line, "
                      f"<b style='color:{self._TOP}'>top</b> and "
                      f"<b style='color:{self._BOTTOM}'>bottom</b>: tap adds "
                      "a node, drag moves it, tap and BACKSPACE deletes it."),
                _sub("Straight top mute"),
                self.w_cut_v, self.w_cut_pad, self.w_seed,
                _sub("Scope"),
                _prop("Edits go to", self.w_scope),
                _hint("One line serves every gather; <b>This gather</b> "
                      "gives the current one an exception."),
                _pair(self.w_promote, self.w_revert), self.w_clear,
                _pair(self.w_import, self.w_export),
                self.status,
                title="Mute", **_card_style(sidebar))
        return self._cards[key]


class VolumeView3D:
    """cigvis-style volume view: a 3-D cuboid with three axis-aligned slice
    planes, rendered in the browser as plotly WebGL surfaces.

    Same technique as cigvis's plotly backend (cigvis's primary engine is
    vispy -- desktop OpenGL, not embeddable in a web app), but wired into
    gathervis's lazy pipeline: moving a slider reads only that slice from the
    (possibly memmap-backed) volume, decimates it to ``MAX_PX_3D`` and
    quantizes to uint8 before shipping.

    One slice at a time is *active*: it is outlined in the view, and it is
    the one the keyboard moves when a workspace binds keys -- ``1`` ``2``
    ``3`` choose it, the arrows step it, Shift steps it a long way, Home and
    End jump to either end. Dragging a slider makes that slice active too, so
    the outline always marks the plane you last touched.

    The figure is kept as a *plain dict*, never a ``go.Figure``: with a live
    Figure, Panel routes in-place mutations through ``Plotly.restyle``, and
    plotly.js gl3d traces mis-apply restyled ``surfacecolor`` (the whole plane
    collapses to the colormap's minimum color -- plotly.js #2866). With a dict,
    Panel diffs the columns server-side (only changed slices go on the wire)
    and re-renders via ``Plotly.react``, which is correct.
    """

    _LIGHT = dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0)
    _STRETCH = [round(2.0 ** (i / 4.0 - 3.0), 3) for i in range(25)]
    _EYE = dict(x=1.28, y=1.14, z=0.64)
    _KEYS_HTML = ("<span class='gv-keys'><kbd>1</kbd> <kbd>2</kbd> "
                  "<kbd>3</kbd> choose a slice &emsp; <kbd>&larr;</kbd> "
                  "<kbd>&rarr;</kbd> move it, with <kbd>Shift</kbd> in big "
                  "steps &emsp; <kbd>Home</kbd> <kbd>End</kbd> first and "
                  "last</span>")

    def __init__(self, vol, names=("recy", "recx"), dt=1.0, t0=0.0,
                 cmap="gray", perc=98.0, clim=None, vertical="time (s)",
                 symmetric=True, lazy=False):
        _ensure_ext()
        self.vol, self.names, self.dt, self.t0 = vol, names, dt, t0
        # lazy: read nothing until ensure_ready() -- see there
        self.ready, self._building = not lazy, False
        self.vertical = vertical            # z-axis label: time (s) / depth (m)
        self.clim = clim or robust_clim(np.asarray(vol), perc,
                                        symmetric=symmetric)
        self._cs = _colorscale(cmap)
        self.active = 0                     # the slice the keys move
        self._rev = 0                       # bumped to reset the camera
        self._controls = self._settings = None

        n0, n1, nt = vol.shape
        vname = vertical.split()[0]          # "time" / "depth"
        self.labels = (str(names[0]), str(names[1]), vname)
        # start on the three camera-facing faces -> the classic closed cuboid
        self.w0 = _theme.IntSlider(name=names[0], start=0, end=n0 - 1,
                                   value=n0 - 1, **_W)
        self.w1 = _theme.IntSlider(name=names[1], start=0, end=n1 - 1,
                                   value=n1 - 1, **_W)
        self.w2 = _theme.IntSlider(name=f"{vname} sample", start=0,
                                   end=nt - 1, value=0, **_W)
        self._sliders = (self.w0, self.w1, self.w2)
        self.w_stretch = [
            pn.widgets.DiscreteSlider(name=f"{lab} stretch",
                                      options=list(self._STRETCH), value=1.0)
            for lab in self.labels]
        self._stretch = [1.0, 1.0, 1.0]
        planes = ([self._surf(a, w.value) for a, w in enumerate(self._sliders)]
                  if self.ready else [self._blank(a) for a in range(3)])
        self.fig = {"data": [*planes, self._box(), self._outline()],
                    "layout": self._layout()}
        self.pane = pn.pane.Plotly(self.fig, sizing_mode="stretch_width",
                                   height=640, config={"displaylogo": False})
        # live updates while dragging (each step is one slice on the wire)
        for axis, w in enumerate(self._sliders):
            w.param.watch(lambda e, a=axis: self._on_slide(a), "value")
        for axis, w in enumerate(self.w_stretch):
            w.param.watch(lambda e, a=axis: self._on_stretch(a, e.new), "value")
        # Panel's viewport machinery only tracks 2-D axis ranges, so the
        # user's 3-D camera would be lost on every react; capture it from
        # relayout events and keep it in our layout instead.
        self.pane.param.watch(self._on_relayout, "relayout_data")

        self._tags = [pn.pane.HTML(self._tag_html(a), width=_theme.LABEL_W,
                                   margin=(4, 8, 4, 0), align="center")
                      for a in range(3)]
        self.keys_hint = _theme.hint(self._KEYS_HTML, visible=False,
                                     margin=(0, 0, 2, 0))
        self._stretch_vals = [pn.pane.HTML(self._stretch_html(a), width=40,
                                           margin=(4, 0, 4, 8), align="center")
                              for a in range(3)]
        for a, w in enumerate(self.w_stretch):
            w.param.watch(lambda e, a=a: setattr(
                self._stretch_vals[a], "object", self._stretch_html(a)),
                "value")
        self.w_camera = pn.widgets.Button(
            **_theme.icon_kw(pn.widgets.Button, "camera", "Reset camera"),
            **_W, description="Back to the default view angle")
        if "icon" in pn.widgets.Button.param:
            self.w_camera.param.update(**{("label" if "label" in
                                           pn.widgets.Button.param else "name"):
                                          "Reset camera"})
        self.w_camera.on_click(lambda e: self.reset_camera())

    # -- small html bits ---------------------------------------------------
    def _tag_html(self, axis):
        on = " gv-on" if axis == self.active else ""
        return (f"<div class='gv-slice-key{on}'><kbd>{axis + 1}</kbd>"
                f"<span>{_html.escape(self.labels[axis])}</span></div>")

    def _stretch_html(self, axis):
        return (f"<div class='gv-value'>&times;"
                f"{float(self.w_stretch[axis].value):g}</div>")

    # -- events --------------------------------------------------------------
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

    def _on_slide(self, axis):
        """A slider moved: that plane is redrawn, and becomes the active one."""
        self._set_active(axis)
        self._move(axis, push=False)       # reads nothing until ready
        self.fig["data"][4] = self._outline()
        if self.ready:
            self._push()

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
        # The strides go into the index itself rather than being applied to
        # a full-resolution slice afterwards: on a memmap either is lazy, but
        # an index-only lazy array (a concatenation of memmaps) would read the
        # whole slice first -- for a time slice, one sample of every trace.
        step = lambda n, budget: max(1, -(-n // budget))
        if axis == 0:                                     # plane at names[0]=idx
            sl = np.asarray(self.vol[idx, ::step(n1, MAX_PX_3D[0]),
                                     ::step(nt, MAX_PX_3D[1])])   # (n1d, ntd)
            Y, Z = np.meshgrid(_dec_coords(n1, MAX_PX_3D[0]),
                               self._tcoords(nt), indexing="ij")
            X = np.full(sl.shape, float(idx), np.float32)
        elif axis == 1:                                   # plane at names[1]=idx
            sl = np.asarray(self.vol[::step(n0, MAX_PX_3D[0]), idx,
                                     ::step(nt, MAX_PX_3D[1])])   # (n0d, ntd)
            X, Z = np.meshgrid(_dec_coords(n0, MAX_PX_3D[0]),
                               self._tcoords(nt), indexing="ij")
            Y = np.full(sl.shape, float(idx), np.float32)
        else:                                             # time plane at sample idx
            sl = np.asarray(self.vol[::step(n0, MAX_PX_3D[0]),
                                     ::step(n1, MAX_PX_3D[1]), idx])  # (n0d, n1d)
            X, Y = np.meshgrid(_dec_coords(n0, MAX_PX_3D[0]),
                               _dec_coords(n1, MAX_PX_3D[1]), indexing="ij")
            Z = np.full(sl.shape, self.t0 + self.dt * idx, np.float32)
        return dict(type="surface", x=X, y=Y, z=Z,
                    surfacecolor=quantize(sl, self.clim),
                    colorscale=self._cs, cmin=0, cmax=255, showscale=False,
                    lighting=self._LIGHT, hoverinfo="skip", name=f"slice{axis}")

    def _blank(self, axis: int) -> dict:
        """A stand-in plane for a lazy view that has not read anything yet."""
        z = np.zeros((2, 2), np.float32)
        return dict(type="surface", x=z, y=z, z=z,
                    surfacecolor=np.zeros((2, 2), np.uint8), colorscale=self._cs,
                    cmin=0, cmax=255, showscale=False, visible=False,
                    hoverinfo="skip", name=f"slice{axis}")

    def ensure_ready(self, defer=False):
        """Read the three slices, if that has not happened yet.

        A whole-line volume can be tens of GB stored trace by trace, and its
        time slice needs one sample from every trace it shows -- with the
        kernel's read-ahead, in practice a read through most of the file. A
        lazy view therefore reads nothing until its tab is first opened.
        ``defer`` (in a served session) shows the loading spinner first and
        reads on the next tick, so the click is answered before the wait.
        """
        if self.ready or self._building:
            return
        doc = pn.state.curdoc
        if defer and doc is not None and doc.session_context is not None:
            self._building = True
            self.pane.loading = True
            pn.state.execute(self._build, schedule=True)
            return
        self._build()

    def _build(self):
        self.ready, self._building = True, False
        try:
            self.redraw()
        finally:
            self.pane.loading = False

    def _extent(self):
        n0, n1, nt = self.vol.shape
        return ((0.0, float(max(n0 - 1, 0))), (0.0, float(max(n1 - 1, 0))),
                (self.t0, self.t0 + self.dt * max(nt - 1, 0)))

    def _box(self) -> dict:
        """Wireframe of the volume extent (12 edges as one polyline)."""
        xs, ys, zs = self._extent()
        edges = ([((xs[0], y, z), (xs[1], y, z)) for y in ys for z in zs]
                 + [((x, ys[0], z), (x, ys[1], z)) for x in xs for z in zs]
                 + [((x, y, zs[0]), (x, y, zs[1])) for x in xs for y in ys])
        bx, by, bz = [], [], []
        for a, b in edges:
            bx += [a[0], b[0], None]
            by += [a[1], b[1], None]
            bz += [a[2], b[2], None]
        return dict(type="scatter3d", x=bx, y=by, z=bz, mode="lines",
                    line=dict(color=_theme.INK_3, width=2),
                    hoverinfo="skip", showlegend=False, name="extent")

    def _outline(self) -> dict:
        """The frame of the active slice, in the 'current' colour, so it is
        plain which plane an arrow key will push."""
        (x0, x1), (y0, y1), (z0, z1) = self._extent()
        a = self.active
        v = float(self._sliders[a].value)
        if a == 0:
            pts = [(v, y0, z0), (v, y1, z0), (v, y1, z1), (v, y0, z1)]
        elif a == 1:
            pts = [(x0, v, z0), (x1, v, z0), (x1, v, z1), (x0, v, z1)]
        else:
            t = self.t0 + self.dt * v
            pts = [(x0, y0, t), (x1, y0, t), (x1, y1, t), (x0, y1, t)]
        pts.append(pts[0])
        xs, ys, zs = (list(c) for c in zip(*pts))
        return dict(type="scatter3d", x=xs, y=ys, z=zs, mode="lines",
                    line=dict(color=_theme.CURRENT, width=5),
                    hoverinfo="skip", showlegend=False, name="active")

    def _layout(self) -> dict:
        n0, n1, nt = self.vol.shape
        # The two horizontal axes keep their proportion (softened, so 60
        # shots by 192 receivers is not a sliver); the vertical gets a fixed
        # share of the box. nt dwarfs everything else, and following it
        # made the cube a tall thin column in a sea of white.
        h = np.array([n0, n1], dtype=float)
        rh = np.clip(np.sqrt(h / h.max()), 0.35, 1.0)
        rz = float(np.clip(0.45 * np.sqrt(nt / h.max()), 0.55, 0.9))
        self._ratio = (float(rh[0]), float(rh[1]), rz)
        tick = dict(family=_theme.FONT, size=11, color=_theme.INK_2)
        head = dict(family=_theme.FONT, size=12, color=_theme.INK)

        def ax(title, **kw):
            return dict(title=dict(text=title, font=head), tickfont=tick,
                        nticks=6, showbackground=False, showspikes=False,
                        gridcolor=_theme.LINE_2, zeroline=False,
                        linecolor=_theme.LINE, **kw)

        return dict(
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
            paper_bgcolor=_theme.SURFACE,
            font=dict(family=_theme.FONT, color=_theme.INK_2),
            # same shape (slice moves, clim, per-shot browse) -> keep camera;
            # different shape, or Reset camera -> the default view
            uirevision=f"{tuple(self.vol.shape)}-{self._rev}",
            scene=dict(
                xaxis=ax(self.names[0]),
                yaxis=ax(self.names[1]),
                zaxis=ax(self.vertical, autorange="reversed"),
                aspectmode="manual",
                aspectratio=dict(x=self._ratio[0] * self._stretch[0],
                                 y=self._ratio[1] * self._stretch[1],
                                 z=self._ratio[2] * self._stretch[2]),
                camera=dict(eye=dict(self._EYE),
                            center=dict(x=0.0, y=0.0, z=-0.04)),
            ),
        )

    # -- updates: mutate the dict + trigger; Panel diffs and reacts ----------
    def _move(self, axis: int, push: bool = True):
        if not self.ready:
            return
        idx = self._sliders[axis].value
        self.fig["data"][axis] = self._surf(axis, idx)
        if push:
            self._push()

    def _set_active(self, axis):
        axis = int(axis)
        if axis == self.active:
            return False
        self.active = axis
        for a, tag in enumerate(self._tags):
            tag.object = self._tag_html(a)
        return True

    def select(self, axis: int):
        """Make slice ``axis`` (0, 1, 2) the one the keys move."""
        if self._set_active(axis):
            self.fig["data"][4] = self._outline()
            self._push()

    def big_step(self) -> int:
        """Shift + arrow: ten samples, or 4 % of the axis if that is more."""
        w = self._sliders[self.active]
        return max(10, (int(w.end) - int(w.start) + 1) // 25)

    def step(self, delta: int):
        """Move the active slice by ``delta`` samples, clamped to the volume."""
        w = self._sliders[self.active]
        w.value = int(min(max(int(w.value) + int(delta), w.start), w.end))

    def jump(self, end: bool):
        """Send the active slice to the last (``end``) or first sample."""
        w = self._sliders[self.active]
        w.value = int(w.end if end else w.start)

    def reset_camera(self):
        """Back to the default view angle, whatever the user dragged to."""
        self._rev += 1
        scene = self.fig["layout"]["scene"]
        scene["camera"] = dict(eye=dict(self._EYE),
                               center=dict(x=0.0, y=0.0, z=-0.04),
                               up=dict(x=0.0, y=0.0, z=1.0))
        self.fig["layout"]["uirevision"] = f"{tuple(self.vol.shape)}-{self._rev}"
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
        if not self.ready:
            return
        for axis in (0, 1, 2):
            self._move(axis, push=False)
        self.fig["data"][4] = self._outline()
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
        self.fig["data"][4] = self._outline()
        self.fig["layout"] = self._layout()
        if same_shape and camera is not None:   # keep the view while browsing
            self.fig["layout"]["scene"]["camera"] = camera
        self.redraw()

    # -- layout ------------------------------------------------------------
    @property
    def sliders(self):
        return pn.Column(pn.Row(self.w0, self.w1, self.w2),
                         pn.Row(*self.w_stretch))

    def controls(self):
        """Where the three slices are, and the keys that move them: the
        navigation row above the plot."""
        if self._controls is None:
            rows = [pn.Row(self._tags[a], self._sliders[a], margin=0,
                           sizing_mode="stretch_width",
                           styles={"padding-right": "20px"} if a < 2 else {})
                    for a in range(3)]
            self._controls = pn.Column(
                pn.Row(*rows, margin=0, sizing_mode="stretch_width"),
                self.keys_hint, margin=(6, 0, 4, 0),
                sizing_mode="stretch_width")
        return self._controls

    def settings(self):
        """Axis stretch and the camera: set once, so the sidebar's."""
        if self._settings is None:
            rows = []
            for a, w in enumerate(self.w_stretch):
                label = self.labels[a].capitalize()
                inner = getattr(w, "_slider", None)
                if inner is None:           # a Panel without the inner slider
                    rows.append(_prop(label, w))
                    continue
                try:
                    inner.param.update(sizing_mode="stretch_width",
                                       width=None, margin=(4, 0))
                except Exception:
                    pass
                rows.append(_prop(label, inner, self._stretch_vals[a],
                                  tip=f"Stretch the {self.labels[a]} axis"))
            self._settings = [_hint("Stretch an axis of the cube; &times;1 "
                                    "is its natural shape."),
                              *rows, self.w_camera]
        return self._settings

    def panel(self, settings=True):
        items = [self.controls(), self.pane]
        if settings:
            items.append(pn.Column(*self.settings(), width=320))
        return pn.Column(*items, sizing_mode="stretch_width")

class ShotBrowser:
    """Slider over the shot axis; each shot is one 2-D ``(trace, time)`` panel.

    A 3-D shot is a patch of receiver lines, and the panel shows all of it:
    the *sort* selector decides the order the patch's traces are laid out in
    (acquisition order, one receiver line, by offset, by azimuth -- see
    ``gathervis.sortkeys``). Keeping every arrangement on the same 2-D panel
    is what lets the windows, spectra, picking and full-image export work on
    3-D shots at all.

    ``state`` is a shared dict holding 'clim' (and 'sample'), owned by the
    Workspace so all views stay consistent.
    """

    def __init__(self, g: Gathers, state: dict, cmap="gray",
                 tools="sidebar"):
        _ensure_ext()
        self.g, self.state = g, state
        # A cached chain when the state is display-backed: stepping shots
        # pays in full, but changing the clip percentile or the colormap no
        # longer re-runs the filter and the AGC for nothing.
        display = getattr(state, "display", None)
        self._chain = _state.Chain(display) if display is not None else None
        if self._chain is not None:
            self._chain.set_global_clim(state["clim"])
        self._tools_sidebar = tools == "sidebar"
        self._help = None
        self._stem = _slug(g.name)
        self._order = None                 # trace order currently displayed

        self.sorts = _sort.available(g)
        self.pane = ImagePane(xlabel="trace", cmap=cmap)
        self.panes = [self.pane]
        self.wt = WindowTool(self.pane)    # windows + spectra + export
        self.pt = PickTool(self.pane)      # per-shot event picking
        # Mute lines live in offset when there is geometry to define one, so
        # that one line means the same cut on every shot; without geometry
        # they fall back to column index and stay put on this shot.
        self.mt = MuteTool(self.pane,
                           domain="offset" if g.geometry is not None else "trace")
        self.mt.on_change.append(self.redraw)

        # The shot slider shows no number of its own: the box next to it
        # does, and that one can be typed into.
        self.w_shot = _theme.IntSlider(name="shot", start=0, end=g.nshot - 1,
                                       value=0, show_value=False,
                                       sizing_mode="stretch_width",
                                       margin=(4, 0))
        self.w_jump = pn.widgets.IntInput(name="", start=0, end=g.nshot - 1,
                                          value=0, width=74,
                                          margin=(4, 0, 4, 10),
                                          description="Go to shot")
        self._shot_label = pn.pane.HTML("<div class='gv-label'>Shot</div>",
                                        margin=(4, 10, 4, 0), align="center")
        self._shot_of = pn.pane.HTML(
            f"<div class='gv-value gv-muted'>/ {g.nshot - 1}</div>",
            margin=(4, 20, 4, 6), align="center")
        # Only worth a selector where there is more than one way to sort: a
        # 2-D line without geometry has exactly one.
        self.w_sort = pn.widgets.Select(
            name="Sort", options=list(self.sorts), value=_sort.AS_RECORDED,
            width=224, margin=(4, 14, 4, 0), visible=len(self.sorts) > 1,
            styles={"--gv-label-w": "30px"},
            description="The order this shot's traces are laid out in")
        nl = _sort.nlines(g)
        self.w_line = _theme.IntSlider(
            name="receiver line", start=0, end=max(nl - 1, 1), value=0,
            width=140, margin=(4, 14, 4, 0), visible=False,
            disabled=nl < 2)
        self._line_label = pn.pane.HTML("<div class='gv-label'>Line</div>",
                                        margin=(4, 8, 4, 0), align="center",
                                        visible=False)

        # live shot flipping while dragging; typing a number jumps there
        self.w_shot.param.watch(lambda e: self._sync(e.new), "value")
        self.w_jump.param.watch(lambda e: self.set_shot(e.new), "value")
        self.w_sort.param.watch(lambda e: self._on_sort(e.new), "value")
        self.w_line.param.watch(lambda e: self.redraw(), "value")
        self.on_shot_change = []        # hooks: called with the new shot index
        self.redraw()

    @property
    def ishot(self) -> int:
        return int(self.w_shot.value)

    @property
    def sort(self) -> str:
        return str(self.w_sort.value)

    def set_shot(self, i: int):
        i = max(0, min(int(i), self.g.nshot - 1))
        self.w_shot.value = i        # fires _sync via watcher (if changed)

    def _sync(self, i: int):
        """Post-change bookkeeping: mirror widgets, then redraw."""
        self.w_jump.value = int(i)            # equal values don't re-trigger
        self.redraw()

    def _on_sort(self, kind: str):
        self.w_line.visible = self._line_label.visible = \
            kind == _sort.RECEIVER_LINE
        self.redraw()

    def redraw(self):
        # reads only this shot's bytes, in the requested order
        panel = _sort.arrange(self.g, self.ishot, self.sort,
                              line=int(self.w_line.value))
        arr, clim = self._process(panel)
        # The mute has to know this panel's x before it can be applied: the
        # lines are stored against offset and the columns change with the
        # sort, so the mapping is rebuilt on every redraw.
        self.mt.set_gather(self.ishot, self._domain(panel))
        arr = self.mt.apply(arr, self.g.dt, t0=self.g.t0)
        self.pane.figure.xaxis.axis_label = panel.xlabel
        self.pane.update(arr, clim, y0=self.g.t0, dy=self.g.dt)
        self._draw_separators(panel)
        self.pt.set_shot(self.ishot)      # picks follow the shot
        self.pt.set_order(panel.order)    # ... and the trace order
        self._order = panel.order
        self.wt.refresh()                 # keep spectra in sync
        self.pane.w_download.filename = self._filename()
        for hook in self.on_shot_change:
            hook(self.ishot)

    def _process(self, panel):
        """Filter and gain this panel, reusing whatever is still valid."""
        if self._chain is None:
            return _process_gather(panel.data, self.g.dt, self.state)
        self._chain.set_global_clim(self.state["clim"])
        key = (self.ishot, self.sort, int(self.w_line.value))
        return self._chain.process(panel.data, self.g.dt, key=key)

    def _domain(self, panel):
        """The value each column is at, in whatever the mute is stored in."""
        if self.g.geometry is None:
            return np.arange(panel.data.shape[0], dtype=float)
        return _sort.offsets(self.g, self.ishot)[panel.order]

    def _draw_separators(self, panel):
        """Mark where one receiver line ends and the next begins."""
        if not len(panel.bounds):
            self.pane.set_separators(())
            return
        edges = np.concatenate([[-0.5], panel.bounds,
                                [panel.data.shape[0] - 0.5]])
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.pane.set_separators(panel.bounds,
                                 labels=[f"L{i}" for i in range(len(centers))],
                                 label_x=centers)

    def _filename(self) -> str:
        tag = {_sort.AS_RECORDED: "", _sort.OFFSET: "_offset",
               _sort.AZIMUTH: "_azimuth"}.get(
                   self.sort, f"_line{int(self.w_line.value):02d}")
        return f"{self._stem}_shot{self.ishot:04d}{tag}.png"

    def tool_cards(self):
        """The tool sections, for hosting in the sidebar."""
        if self._help is None:
            self._help = _gesture_help()
        return [self.wt.window_card(True), self.wt.spectrum_card(True),
                self.pt.picking_card(True), self.pt.fb_card(True),
                self.mt.card(True), self._help[0]]

    def controls(self):
        """Shot slider + jump box, and the sort selector where it applies."""
        items = [self._shot_label, self.w_shot, self.w_jump, self._shot_of]
        if len(self.sorts) > 1:
            items += [self.w_sort, self._line_label, self.w_line]
        return pn.Row(*items, margin=0, sizing_mode="stretch_width")

    def panel(self):
        nav = [self.controls()]
        if self._tools_sidebar:
            # the tools live in the sidebar; only the figure and the spectra
            # it produces stay in the main column
            return self.pane.frame(self.wt.figures(), nav=nav)
        if self._help is None:
            self._help = _gesture_help(sidebar=False)
        return self.pane.frame(
            self.wt.figures(),
            pn.FlexBox(self.wt.window_card(), self.wt.spectrum_card(),
                       self.pt.picking_card(), self.pt.fb_card(),
                       self.mt.card(), self._help[0]),
            nav=nav)


def _map_figure(geo, height=None):
    """An empty map of the survey area, sized to the survey's own shape
    rather than padded out to the window's."""
    src = np.asarray(geo.src, dtype=float)[:, :2]
    rec = np.asarray(geo.unique_receivers(), dtype=float)[:, :2]
    span = np.ptp(np.concatenate([src, rec]), axis=0)
    w, h = max(float(span[0]), 1e-9), max(float(span[1]), 1e-9)
    if height is None:
        # about 1150 px of plot width, plus room for axes and legend
        height = int(np.clip(1150.0 * h / w + 110, 380, 760))
    fig = figure(height=height, sizing_mode="stretch_width", match_aspect=True,
                 x_axis_label="x", y_axis_label="y",
                 tools="pan,wheel_zoom,box_zoom,reset,save,tap",
                 active_scroll="wheel_zoom")
    fig.x_range.range_padding = fig.y_range.range_padding = 0.04
    _theme.style_figure(fig)
    return fig


class LayoutMap:
    """Acquisition map: sources are tappable; the current shot and its
    spread are highlighted.

    It draws on a figure of its own, or -- in the Geometry tab -- on top of
    the fold map, so where the survey is and how well it covers are one
    picture instead of two to hold side by side.
    """

    def __init__(self, geo, figure=None):
        _ensure_ext()
        self.geo = geo
        self.on_pick = None             # hook: called with the picked shot index
        fig = figure if figure is not None else _map_figure(geo)
        rec_all = geo.unique_receivers()      # each position once
        self.r_rec = fig.scatter(rec_all[:, 0], rec_all[:, 1], size=3,
                                 color=_theme.INK_3, alpha=0.75)
        self._cds_src = ColumnDataSource(dict(x=geo.src[:, 0], y=geo.src[:, 1]))
        self.r_src = fig.scatter(
            "x", "y", source=self._cds_src, size=7, fill_color="#ffffff",
            line_color=_theme.INK, line_width=1.3,
            selection_fill_color="#ffffff", selection_line_color=_theme.INK,
            nonselection_fill_alpha=1.0, nonselection_line_alpha=1.0)
        self._cds_arec = ColumnDataSource(dict(x=[], y=[]))
        self._cds_asrc = ColumnDataSource(dict(x=[], y=[]))
        self.r_arec = fig.scatter("x", "y", source=self._cds_arec, size=4,
                                  color=_theme.CURRENT)
        self.r_asrc = fig.scatter("x", "y", source=self._cds_asrc, size=17,
                                  marker="star", fill_color=_theme.CURRENT,
                                  line_color="#ffffff", line_width=1.2)
        taps = list(fig.select(TapTool))
        if not taps:
            taps = [TapTool()]
            fig.add_tools(*taps)
        for t in taps:
            t.renderers = [self.r_src]
        self.legend = _theme.style_legend(Legend(
            items=[LegendItem(label="Receivers", renderers=[self.r_rec]),
                   LegendItem(label="Sources", renderers=[self.r_src]),
                   LegendItem(label="Current shot and spread",
                              renderers=[self.r_arec])],
            orientation="horizontal", location="top_left"))
        fig.add_layout(self.legend, "above")
        self._cds_src.selected.on_change("indices", self._on_tap)
        self.figure = fig

    def _on_tap(self, attr, old, new):
        if new and self.on_pick is not None:
            self.on_pick(new[0])
        if new:
            # forget the selection, so tapping the same source again still
            # counts as a tap
            self._cds_src.selected.indices = []

    def set_active(self, i: int):
        rec = self.geo.rec_for(i)
        self._cds_arec.data = dict(x=rec[:, 0], y=rec[:, 1])
        self._cds_asrc.data = dict(x=[self.geo.src[i, 0]], y=[self.geo.src[i, 1]])

    def set_visible(self, receivers=True, sources=True):
        """Show or hide the receiver layer and the source layer."""
        self.r_rec.visible = self.r_arec.visible = bool(receivers)
        self.r_src.visible = self.r_asrc.visible = bool(sources)

    def near_source(self, x, y, frac=0.012) -> bool:
        """Is a map position within a hair of a source? Lets the fold map
        leave a tap alone when it was meant for the source under it."""
        src = np.asarray(self.geo.src, dtype=float)[:, :2]
        if not len(src) or not self.r_src.visible:
            return False
        r = self.figure.x_range
        try:
            span = abs(float(r.end) - float(r.start))
        except (TypeError, ValueError):
            span = float("nan")
        if not np.isfinite(span) or span <= 0:
            span = float(np.ptp(src[:, 0])) or 1.0
        return float(np.hypot(src[:, 0] - x, src[:, 1] - y).min()) <= frac * span


class FoldMap:
    """CMP fold over the survey: how many traces image each bin.

    The first thing anyone looks at after loading geometry, because a hole in
    the fold map is a hole in the image and it is visible here long before
    anything has been processed. Empty bins are drawn as background rather
    than as the bottom of the colour scale, so the live area's own shape --
    the tapered edges, the gaps -- reads directly.

    Computed from coordinates only (see ``gathervis.survey``), so it costs the
    same on a 100 GB survey as on a toy one.
    """

    _EMPTY = "#f3f5f7"

    def __init__(self, geo, cmap="viridis", height=None):
        _ensure_ext()
        self.geo = geo
        self.on_pick = None          # hook: called with (ix, iy) of a tapped bin
        self.tap_filter = None       # (x, y) -> True to leave a tap alone
        self.grid = None
        dx, dy = _survey.default_bin(geo)
        self.w_dx = pn.widgets.FloatInput(name="Bin x", value=round(dx, 4),
                                          start=0.0, **_W)
        self.w_dy = pn.widgets.FloatInput(name="Bin y", value=round(dy, 4),
                                          start=0.0, **_W)
        self.w_reset = pn.widgets.Button(
            name="Default bin size", **_W,
            description="Half the receiver spacing on each axis")
        self.mapper = LinearColorMapper(palette=_palette(cmap), low=1, high=2,
                                        nan_color=self._EMPTY)
        fig = _map_figure(geo, height)
        fig.background_fill_color = self._EMPTY
        self.cds = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
        self.r_img = fig.image(image="image", x="x", y="y", dw="dw", dh="dh",
                               source=self.cds, color_mapper=self.mapper)
        self.colorbar = _theme.style_colorbar(ColorBar(
            color_mapper=self.mapper, title="Fold", width=10, padding=8,
            ticker=BasicTicker(desired_num_ticks=6)))
        fig.add_layout(self.colorbar, "right")
        self._cds_pick = ColumnDataSource(dict(x=[], y=[], w=[], h=[]))
        fig.rect(x="x", y="y", width="w", height="h", source=self._cds_pick,
                 fill_alpha=0, line_color=_theme.INK, line_width=2)
        self.readout = Div(text="&nbsp;", height=18, styles=_READOUT_STYLE)
        fig.js_on_event("mousemove", CustomJS(
            args=dict(src=self.cds, div=self.readout), code=_FOLD_READOUT_JS))
        fig.js_on_event("mouseleave", CustomJS(
            args=dict(div=self.readout), code='div.text = "&nbsp;";'))
        fig.on_event("tap", self._on_tap)
        self.figure = fig
        self.summary = pn.pane.HTML("", sizing_mode="stretch_width",
                                    margin=(8, 0, 4, 0))
        self.w_dx.param.watch(lambda e: self.compute(), "value")
        self.w_dy.param.watch(lambda e: self.compute(), "value")
        self.w_reset.on_click(lambda e: self._defaults())
        self.compute()

    def _defaults(self):
        dx, dy = _survey.default_bin(self.geo)
        self.w_dx.value, self.w_dy.value = round(dx, 4), round(dy, 4)

    def compute(self):
        """Re-bin at the current bin size; a bad size is reported, not raised."""
        try:
            grid = _survey.fold(self.geo,
                                (float(self.w_dx.value), float(self.w_dy.value)))
        except ValueError as e:
            self.summary.object = f"<div class='gv-warn'>{e}</div>"
            return
        self.grid = grid
        img = grid.counts.astype(np.float32)
        img[img == 0] = np.nan                    # empty bins: background
        x0, y0, w, h = grid.extent
        self.mapper.low, self.mapper.high = 1, max(int(grid.counts.max()), 1)
        self.cds.data = dict(image=[img], x=[x0], y=[y0], dw=[w], dh=[h])
        self._cds_pick.data = dict(x=[], y=[], w=[], h=[])
        live = grid.counts[grid.counts > 0]
        ny, nx = grid.shape
        lo, hi = (int(live.min()), int(live.max())) if live.size else (0, 0)
        mean = float(live.mean()) if live.size else 0.0
        self.summary.object = _theme.kv([
            ("Grid", f"{nx} &times; {ny} bins"),
            ("Bin size", f"{grid.dx:g} &times; {grid.dy:g}"),
            ("Live bins", f"{grid.nlive}"),
            ("Fold", f"{lo}&ndash;{hi}, mean {mean:.1f}"),
            ("Traces", f"{int(grid.counts.sum())}"),
        ])

    def _on_tap(self, event):
        if self.grid is None:
            return
        if self.tap_filter is not None and self.tap_filter(event.x, event.y):
            return
        hit = self.grid.bin_of(event.x, event.y)
        if hit is None:
            return
        self.set_active(*hit)
        if self.on_pick is not None:
            self.on_pick(*hit)

    def set_active(self, ix: int, iy: int):
        """Outline one bin (the one a rose diagram would be drawn for)."""
        g = self.grid
        self._cds_pick.data = dict(x=[g.x0 + (ix + 0.5) * g.dx],
                                   y=[g.y0 + (iy + 0.5) * g.dy],
                                   w=[g.dx], h=[g.dy])

    def set_visible(self, on=True):
        """Show or hide the fold layer (and its colour bar)."""
        self.r_img.visible = self.colorbar.visible = bool(on)

    def settings(self):
        """Bin size and the numbers that come out of it."""
        return [self.w_dx, self.w_dy, self.w_reset, self.summary]

    def panel(self):
        """The map with its readout, and the bin settings under it; the
        Geometry tab puts those in the sidebar instead."""
        return pn.Column(self.figure, self.readout,
                         pn.Column(*self.settings(), width=320),
                         sizing_mode="stretch_width")


def _survey_kind(g: Gathers) -> str:
    if "shot" in g.axes:
        if g.data.ndim == 4:
            return (f"3-D seismic ({_sort.nlines(g)} receiver lines x "
                    f"{_sort.line_length(g)} stations per shot)")
        return "2-D seismic line"
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


def _info_html(g: Gathers, title: bool = True) -> str:
    """The dataset block at the top of the sidebar: what this data is.

    ``title=False`` leaves the name out, for the served page, whose header
    already carries it.
    """
    n_bytes = int(np.prod(g.shape)) * np.dtype(g.data.dtype).itemsize
    size = (f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e9
            else f"{n_bytes / 1e6:.1f} MB")
    last = g.t0 + g.dt * (g.nt - 1)
    rows = [("Shape", " &times; ".join(str(int(n)) for n in g.shape)),
            ("Axes", ", ".join(g.axes)),
            ("Sampling", f"{g.dt:g} m, to {last:g} m" if g.axes[-1] == "depth"
             else f"{g.dt * 1e3:g} ms, to {last:.3f} s"),
            ("Size", f"{size} {np.dtype(g.data.dtype).name}")]
    if g.geometry is not None:
        geo = g.geometry
        spread = "shared spread" if geo.shared else "per shot"
        rows.append(("Geometry", f"{geo.ns} sources, {geo.nr} receivers "
                                 f"{spread}"))
    head = ""
    if title and g.name:
        head = f"<div class='gv-title'>{_html.escape(str(g.name))}</div>"
    head += f"<div class='gv-kind'>{_survey_kind(g)}</div>"
    return head + _theme.kv(rows)


class VelocityPanel:
    """Hand-picked velocity analysis on one CMP gather.

    Three panels reading left to right, which is also the order the work
    happens in: the gather, its semblance spectrum, and the gather with the
    current pick's moveout removed.

    Picking is by hand and only by hand. Tap the spectrum to add a ``(v, t)``
    point, drag to move it, BACKSPACE to delete. The moveout panel follows
    every drag, and it is the thing that decides whether a pick is right --
    a flat event means yes and nothing else does. That loop, pick and look,
    is the whole feature; an automatic picker would be answering the
    question instead of showing it.

    What comes out is the velocity file: ``time_s,velocity`` at the control
    points, with the gather's location in a header comment if it was given.
    It reads back in.

    The input is a gather that was handed in, not one this class went and
    found. A CMP gather from ``cmp.gather_in_bin``, from a SEG-Y sort, from
    an array someone already has -- they are all the same thing here, and
    the panel should not care which.
    """

    _PICK = "#d81b60"

    def __init__(self, data, offsets, dt, t0=0.0, cmap="gray", perc=98.0,
                 label="", x=None, y=None, height=430):
        _ensure_ext()
        self.data = np.asarray(data, dtype=np.float32)
        if self.data.ndim != 2:
            raise ValueError(f"expected a 2-D (ntrace, nt) gather, got "
                             f"shape {self.data.shape}")
        self.offsets = np.asarray(offsets, dtype=np.float32).reshape(-1)
        if self.offsets.size != self.data.shape[0]:
            raise ValueError(f"offsets has {self.offsets.size} entries but "
                             f"the gather has {self.data.shape[0]} traces")
        self.dt, self.t0 = float(dt), float(t0)
        self.label, self.x, self.y = label, x, y
        self.times = (self.t0 + np.arange(self.data.shape[1], dtype=np.float32)
                      * self.dt)
        self._spec = self._vgrid = None
        self._busy = False

        self.pane_cmp = ImagePane(xlabel="trace (offset-sorted)", cmap=cmap,
                                  height=height)
        self.pane_spec = ImagePane(xlabel="velocity (m/s)", cmap="rainbow",
                                   height=height)
        self.pane_nmo = ImagePane(xlabel="trace (offset-sorted)", cmap=cmap,
                                  height=height)
        self.panes = [self.pane_cmp, self.pane_nmo]   # the spectrum keeps its own

        # the pick, on the spectrum
        self._cds_line = ColumnDataSource(dict(x=[], y=[]))
        self.pane_spec.figure.line("x", "y", source=self._cds_line,
                                   line_width=2, line_color=self._PICK)
        self.cds = ColumnDataSource(dict(x=[], y=[]))
        r = self.pane_spec.figure.scatter("x", "y", source=self.cds, size=10,
                                          fill_color=self._PICK, fill_alpha=0.9,
                                          line_color="white", line_width=1.5)
        self.pane_spec.figure.add_tools(
            _theme.set_tool_icon(PointDrawTool(
                renderers=[r], description="Pick velocities"), "vpick"))
        self.cds.on_change("data", lambda a, o, n: self._on_pick())

        # the stacked trace: one number per time, the pick's whole product
        stack = figure(height=height, width=130, y_axis_location="right",
                       x_axis_label="stack", y_axis_label="time (s)",
                       tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom")
        stack.y_range.flipped = True
        _theme.style_figure(stack, grid=False)
        stack.xaxis.ticker = []
        self._cds_stack = ColumnDataSource(dict(x=[], y=[]))
        stack.line("x", "y", source=self._cds_stack, line_width=1,
                   line_color="#111111")
        self.fig_stack = stack

        self._perc = float(perc)
        self._build_widgets()
        self.note = pn.pane.HTML("", sizing_mode="stretch_width")
        self.compute()

    # ------------------------------------------------------------------ UI
    def _build_widgets(self):
        self.w_mute_on = pn.widgets.Checkbox(name="front mute", value=True, **_W)
        self.w_mute_v = pn.widgets.FloatInput(name="mute v (m/s)", value=1500.0,
                                              start=1.0, step=100.0, **_W)
        self.w_mute_pad = pn.widgets.FloatInput(name="delay (s)", value=0.05,
                                                start=0.0, step=0.01, **_W)
        self.w_vmin = pn.widgets.FloatInput(name="v min", value=1400.0,
                                            start=1.0, step=100.0, **_W)
        self.w_vmax = pn.widgets.FloatInput(name="v max", value=4500.0,
                                            start=2.0, step=100.0, **_W)
        self.w_nv = pn.widgets.IntInput(name="n velocities", value=100,
                                        start=2, end=600, step=10, **_W)
        self.w_window = pn.widgets.IntInput(
            name="semblance window", value=21, start=1, end=401, step=2, **_W,
            description="samples; longer smooths the panel and blurs in time")
        self.w_stretch = pn.widgets.FloatInput(
            name="stretch mute", value=0.5, start=0.0, step=0.1, **_W,
            description="drop samples stretched by more than this fraction")
        self.w_clear = pn.widgets.Button(name="clear picks", **_W)
        self.w_clear.on_click(lambda e: self.clear())
        self.w_export = pn.widgets.FileDownload(
            callback=self._export, filename="velocity.csv",
            label="export velocity file", **_W)
        self.w_import = pn.widgets.FileInput(accept=".csv,.txt", **_WFILE)
        self.w_import.param.watch(self._import, "value")
        for w in (self.w_mute_on, self.w_mute_v, self.w_mute_pad, self.w_vmin,
                  self.w_vmax, self.w_nv, self.w_window, self.w_stretch):
            w.param.watch(lambda e: self.compute(), "value")

    # ------------------------------------------------------------ the work
    def _prepared(self):
        """The gather the analysis runs on: front-muted if that is on."""
        if not self.w_mute_on.value:
            return self.data
        return _vel.front_mute(self.data, self.offsets, self.dt,
                               float(self.w_mute_v.value), t0=self.t0,
                               pad=float(self.w_mute_pad.value))

    def compute(self):
        """Draw the gather and (re)compute its velocity spectrum."""
        if self._busy:
            return
        self._busy = True
        try:
            arr = self._prepared()
            clim = robust_clim(arr, self._perc)
            self.pane_cmp.update(arr, clim, y0=self.t0, dy=self.dt)
            vmin, vmax = float(self.w_vmin.value), float(self.w_vmax.value)
            if vmax <= vmin:
                self.note.object = ("<div class='gv-warn'>"
                                    "v max must be greater than v min</div>")
                return
            self._spec, self._vgrid = _vel.velocity_spectrum(
                arr, self.offsets, self.dt, t0=self.t0, vmin=vmin, vmax=vmax,
                nv=int(self.w_nv.value), window=int(self.w_window.value),
                stretch_mute=float(self.w_stretch.value))
            dv = (vmax - vmin) / max(int(self.w_nv.value) - 1, 1)
            self.pane_spec.update(self._spec,
                                  (0.0, float(self._spec.max()) or 1.0),
                                  x0=vmin, dx=dv, y0=self.t0, dy=self.dt)
            self._refresh_note()
        finally:
            self._busy = False
        self._on_pick()

    def velocity_function(self) -> np.ndarray:
        """The current pick as v(t) on the gather's own time axis.

        Control points are interpolated and held flat beyond the ends.
        Outside the range you picked there is no information, and
        extrapolating the trend would invent some.
        """
        d = self.cds.data
        ts, vs = np.asarray(d["y"], float), np.asarray(d["x"], float)
        if ts.size == 0:
            mid = 0.5 * (float(self.w_vmin.value) + float(self.w_vmax.value))
            return np.full(self.times.size, mid, np.float32)
        order = np.argsort(ts)
        ts, vs = ts[order], vs[order]
        return np.interp(self.times, ts, vs, left=vs[0],
                         right=vs[-1]).astype(np.float32)

    def _on_pick(self):
        """A pick moved: redraw the v(t) line, the moveout panel, the stack."""
        v_t = self.velocity_function()
        self._cds_line.data = dict(x=[float(v) for v in v_t],
                                   y=[float(t) for t in self.times])
        arr = self._prepared()
        clim = robust_clim(arr, self._perc)
        corrected = _vel.nmo(arr, self.offsets, v_t, self.dt, t0=self.t0,
                             stretch_mute=float(self.w_stretch.value))
        self.pane_nmo.update(corrected, clim, y0=self.t0, dy=self.dt)
        stacked = _vel.nmo_stack(arr, self.offsets, v_t, self.dt, t0=self.t0,
                                 stretch_mute=float(self.w_stretch.value))
        peak = float(np.abs(stacked).max()) or 1.0
        self._cds_stack.data = dict(x=[float(v) / peak for v in stacked],
                                    y=[float(t) for t in self.times])
        self._refresh_note()

    def clear(self):
        self.cds.data = dict(x=[], y=[])

    def picks(self):
        """``(times, velocities)`` of the control points, in time order."""
        d = self.cds.data
        ts, vs = np.asarray(d["y"], float), np.asarray(d["x"], float)
        order = np.argsort(ts)
        return ts[order], vs[order]

    def _refresh_note(self):
        n = len(self.cds.data["x"])
        where = ""
        if self.x is not None and self.y is not None:
            where = f" at ({self.x:.1f}, {self.y:.1f})"
        head = f"<b>{self.label}</b>{where} &nbsp;·&nbsp; " if self.label or where else ""
        self.note.object = (
            f"<div class='gv-hint'>{head}fold "
            f"<b>{self.data.shape[0]}</b> &nbsp;·&nbsp; offsets "
            f"<b>{self.offsets.min():.0f}-{self.offsets.max():.0f}</b>"
            f" &nbsp;·&nbsp; <b>{n}</b> pick{'' if n == 1 else 's'}"
            + ("" if n else " &nbsp;·&nbsp; tap the spectrum to start"))

    # -------------------------------------------------------------- files
    def _export(self):
        import io
        buf = io.StringIO()
        buf.write("# gathervis velocity picks\n")
        if self.label:
            buf.write(f"# gather: {self.label}\n")
        if self.x is not None and self.y is not None:
            buf.write(f"# x: {self.x:.4f}\n# y: {self.y:.4f}\n")
        buf.write("time_s,velocity\n")
        for t, v in zip(*self.picks()):
            buf.write(f"{t:.6f},{v:.2f}\n")
        buf.seek(0)
        return buf

    def _import(self, event):
        raw = event.new
        if isinstance(raw, dict):
            raw = next(iter(raw.values()), None)
        if not raw:
            return
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        ts, vs = [], []
        for row in raw.splitlines():
            row = row.strip()
            if not row or row.startswith("#") or row.lower().startswith("time"):
                continue
            parts = row.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                ts.append(float(parts[0]))
                vs.append(float(parts[1]))
            except ValueError:
                continue
        if not ts:
            self.note.object = ("<div class='gv-warn'>"
                                "no time,velocity rows in that file</div>")
            return
        self.cds.data = dict(x=[float(v) for v in vs], y=[float(t) for t in ts])

    # --------------------------------------------------------------- cards
    def cards(self, sidebar=False):
        return [
            pn.Card(self.w_mute_on, pn.Row(self.w_mute_v, self.w_mute_pad, **_W),
                    _hint("the direct wave is strong, perfectly coherent and "
                          "not a reflection, so it puts a bright low-velocity "
                          "smear across the shallow spectrum"),
                    title="Front mute", **_card_style(sidebar)),
            pn.Card(pn.Row(self.w_vmin, self.w_vmax, **_W),
                    pn.Row(self.w_nv, self.w_window, **_W), self.w_stretch,
                    title="Velocity spectrum", **_card_style(sidebar)),
            pn.Card(_hint("point toolbar tool on the spectrum: tap adds a "
                          "pick, drag moves it, tap-select + BACKSPACE "
                          "deletes. Watch the moveout panel -- flat is right."),
                    self.w_clear, self.w_export, self.w_import,
                    title="Velocity picks", **_card_style(sidebar)),
        ]

    def panel(self, tools=True):
        def col(title, pane):
            return pn.Column(
                pn.pane.HTML(f"<div class='gv-caption'>{title}</div>"),
                pane.figure, pane.readout, sizing_mode="stretch_width")
        row = pn.Row(col("CMP gather", self.pane_cmp),
                     col("velocity spectrum", self.pane_spec),
                     col("moveout corrected", self.pane_nmo),
                     pn.Column(pn.pane.HTML("<div class='gv-caption'>stack"
                                            "</div>"), self.fig_stack, margin=0),
                     sizing_mode="stretch_width")
        items = [self.note, row]
        if tools:
            items.append(pn.FlexBox(*self.cards()))
        return pn.Column(*items, sizing_mode="stretch_width")


def velocity_analysis(gather, offsets=None, dt=None, t0=0.0, cmap="gray",
                      perc=98.0, label="", x=None, y=None, port=None,
                      address="127.0.0.1", title="gathervis velocity",
                      verbose=True):
    """Open velocity analysis on one CMP gather.

        from gathervis import cmp, survey
        import gathervis as gv

        grid = survey.fold(ds.geometry)
        cg = cmp.gather_in_bin(ds, grid, ix, iy)
        gv.velocity_analysis(cg, port=8080)

    Parameters
    ----------
    gather : CmpGather | (ntrace, nt) array
        A ``CmpGather`` carries its own offsets and bin location, so those
        arguments can be left out. Any other 2-D array works too, given
        ``offsets`` and ``dt``.
    offsets, dt : optional
        Required unless ``gather`` supplies them.
    port : int, optional
        Serve blocking on ``address:port``; ``None`` returns the Panel app.

    Returns the :class:`VelocityPanel` when served, so the picks are still
    reachable afterwards, and the Panel app when not.
    """
    if hasattr(gather, "offsets") and hasattr(gather, "data"):    # CmpGather
        offsets = gather.offsets if offsets is None else offsets
        if x is None:
            x, y = getattr(gather, "x", None), getattr(gather, "y", None)
        if not label:
            label = f"bin {gather.ix}, {gather.iy}"
        gather = gather.data
    if offsets is None:
        raise ValueError("offsets= is required for a plain array")
    if dt is None:
        raise ValueError("dt= is required")

    def _app(vp):
        return pn.Column(vp.panel(tools=False), pn.FlexBox(*vp.cards()),
                         sizing_mode="stretch_width")

    def _make():
        return VelocityPanel(gather, offsets, dt, t0=t0, cmap=cmap, perc=perc,
                             label=label, x=x, y=y)

    vp = _make()
    if port is None:
        return _app(vp)
    port = _resolve_port(port, address)
    if verbose:
        _banner(port)
    build = _per_session(vp, _make)       # one panel per browser session
    pn.serve(lambda: _app(build()), port=port, address=address, show=False,
             title=title, websocket_origin="*")
    return vp


# ---------------------------------------------------------------------------
# views: one per tab, registered rather than branched on
# ---------------------------------------------------------------------------
class _View(_state.View):
    """A tab, and what the session needs to know to place it.

    ``applies`` is what replaces the chain of ``if`` statements the workspace
    used to be: each view says for itself whether this dataset has what it
    needs, and the workspace just asks all of them. Adding a view means
    adding a class to ``VIEWS``, not editing the workspace in four places.

    Views never hold references to each other. A view that changes the
    selection writes ``session.cursor``; a view that cares reads it. That is
    why the layout map can move the shot browser without either one knowing
    the other exists.
    """

    title = "view"          # tab label
    cards = ()              # tool cards, hosted in the sidebar

    @staticmethod
    def applies(g) -> bool:
        return True

    def __init__(self, **kw):
        self.opts = kw
        self.session = None
        # A seam, deliberately visible: the two volume tabs redraw through
        # logic that still lives on the workspace, because it needs the
        # "too large to process" note and the size guard that go with it.
        # Everything else about them is here. See docs/todo.md.
        self._refresh_hook = None

    def panes(self):
        """Image panes that share the display-mode / colormap controls."""
        return []

    def tool_cards(self, sidebar=True):
        return []

    def sidebar_sections(self):
        """Sidebar sections that belong to this tab, shown only on it."""
        return []


class GatherView(_View):
    """The shot browser: one shot at a time as a 2-D (trace, time) panel."""

    name = "gather"
    title = "Shot gathers"

    @staticmethod
    def applies(g):
        return "shot" in g.axes

    def attach(self, session):
        self.session = session
        self.browser = ShotBrowser(session.data, session.state,
                                   cmap=self.opts.get("cmap", "gray"),
                                   tools=self.opts.get("tools", "sidebar"))
        # Selection flows through the cursor in both directions. The cursor
        # only announces real changes, so the two cannot chase each other.
        self.browser.on_shot_change.append(
            lambda i: setattr(session.cursor, "shot", i))
        session.cursor.on_change(
            lambda what: self.browser.set_shot(session.cursor.shot)
            if what == "shot" else None)

    def panel(self):
        return self.browser.panel()

    def panes(self):
        return list(self.browser.panes)

    def tool_cards(self, sidebar=True):
        return self.browser.tool_cards() if sidebar else []

    def refresh(self, stage=_state.DATA):
        self.browser.redraw()


class GeometryView(_View):
    """The survey on one map: CMP fold underneath, the acquisition layout on
    top, the current shot and its spread highlighted.

    One picture, because the two answer one question -- where the survey
    is, and how well it covers -- and a hole in the fold is a gap in the
    layout right on top of it. Bin size and layer switches are settings, so
    they sit in the sidebar and the tab is the map.
    """

    name = "geometry"
    title = "Geometry"

    @staticmethod
    def applies(g):
        return g.geometry is not None

    def attach(self, session):
        self.session = session
        geo = session.data.geometry
        self.fold = FoldMap(geo)
        self.fold.on_pick = lambda ix, iy: setattr(session.cursor, "bin",
                                                   (ix, iy))
        self.map = LayoutMap(geo, figure=self.fold.figure)
        self.map.on_pick = self._picked
        self.fold.tap_filter = self.map.near_source
        self.w_layers = pn.widgets.CheckButtonGroup(
            name="Show", options=["Fold", "Receivers", "Sources"],
            value=["Fold", "Receivers", "Sources"], **_W)
        self.w_layers.param.watch(self._on_layers, "value")
        self._sections = None
        session.cursor.on_change(
            lambda what: self.map.set_active(session.cursor.shot)
            if what == "shot" else None)
        self.map.set_active(session.cursor.shot)

    def _on_layers(self, event):
        on = set(event.new)
        self.fold.set_visible("Fold" in on)
        self.map.set_visible(receivers="Receivers" in on,
                             sources="Sources" in on)

    def _picked(self, i):
        self.session.cursor.shot = int(i)
        self.session.focus("gather")       # ... and show it

    def panel(self):
        return pn.Column(self.fold.figure, self.fold.readout,
                         sizing_mode="stretch_width", margin=(10, 0, 0, 0))

    def sidebar_sections(self):
        if self._sections is None:
            self._sections = [
                _theme.section("Map layers", self.w_layers,
                               _hint("Tap a source to open its shot.")),
                _theme.section("CMP bins", *self.fold.settings())]
        return self._sections


class ShotVolumeView(_View):
    """The current shot as a cuboid: the one place a 3-D shot reads as one.

    It follows the shot slider rather than owning one -- the 2-D panel is
    the working view and this is the second opinion, so having two shot
    controls that could disagree would be worse than having none here.
    """

    name = "shot_volume"
    title = "Shot volume"

    @staticmethod
    def applies(g):
        return g.data.ndim == 4

    def attach(self, session):
        self.session = session
        g = session.data
        self.volume = VolumeView3D(g.shot(0), names=g.axes[1:3], dt=g.dt,
                                   t0=g.t0, cmap=self.opts.get("cmap", "gray"),
                                   clim=session.state["clim"])
        session.cursor.on_change(
            lambda what: self.refresh() if what == "shot" else None)

    def panel(self):
        return self.volume.panel(settings=False)

    def panes(self):
        return [self.volume]

    def sidebar_sections(self):
        if getattr(self, "_sections", None) is None:
            self._sections = [_theme.section("Volume view",
                                             *self.volume.settings())]
        return self._sections

    def refresh(self, stage=_state.DATA):
        if self._refresh_hook is not None:
            self._refresh_hook()


class SlicesView(_View):
    """A 3-D volume as a cuboid with movable slice planes."""

    name = "slices"
    title = "Volume slices"

    @staticmethod
    def applies(g):
        return g.data.ndim == 3

    def attach(self, session):
        self.session = session
        g = session.data
        vertical = "depth (m)" if g.axes[-1] == "depth" else "time (s)"
        # lazy: a whole line is read only when this tab is first opened
        self.volume = VolumeView3D(g.data, names=tuple(g.axes[:-1]), dt=g.dt,
                                   t0=g.t0, cmap=self.opts.get("cmap", "gray"),
                                   clim=session.state["clim"],
                                   vertical=vertical, lazy=True)

    def panel(self):
        return self.volume.panel(settings=False)

    def panes(self):
        return [self.volume]

    def sidebar_sections(self):
        if getattr(self, "_sections", None) is None:
            self._sections = [_theme.section("Volume view",
                                             *self.volume.settings())]
        return self._sections

    def refresh(self, stage=_state.DATA):
        if self._refresh_hook is not None:
            self._refresh_hook()


# The registry. Order here is tab order.
VIEWS = (GatherView, GeometryView, ShotVolumeView, SlicesView)


class Workspace:
    """Sidebar (dataset info + display controls) + tabbed views.

    Tabs adapt to the data: shot browsing, whole-volume slicing, and -- when
    geometry is present -- an acquisition-layout tab. Tapping a source on the
    layout map selects that shot and jumps to the shot-gather tab.
    """

    def __init__(self, g: Gathers, cmap="gray", perc=98.0, view=None,
                 tools="sidebar", keys=True, views=None):
        _ensure_ext()
        self.g = g
        self._synced = False
        self._tools_sidebar = tools == "sidebar"
        self._tool_cards = []          # cards hosted in the sidebar
        self._tool_tabs = set()        # tab indices those cards apply to
        # The session owns the shared display settings; `state` is its dict
        # view, so everything written against the old dict keeps working
        # while the stages it announces drive the caches underneath.
        self.session = _state.Session(g, cmap=cmap, perc=perc)
        self.state = self.session.state
        self._panes, self._redraws, tabs = [], [], []
        self.map = None
        self._shot_tab = None

        # Every view says for itself whether this dataset has what it needs,
        # so the workspace asks all of them instead of branching on shape
        # and geometry five times. Adding a view is adding a class to VIEWS.
        self.session.on_focus(self._focus_view)
        self._view_tabs = {}
        # `views=` is the explicit layer asking for exactly these, in this
        # order; without it every view that fits the data goes in. Either
        # way the loop is the same, which is the point of the registry.
        wanted = VIEWS if views is None else tuple(views)
        for item in wanted:
            v = item if isinstance(item, _View) else item(cmap=cmap, tools=tools)
            if not type(v).applies(g):
                if views is None:
                    continue                  # automatic: skip what does not fit
                raise ValueError(
                    f"{type(v).__name__} needs something this dataset does "
                    f"not have (axes={g.axes}, geometry="
                    f"{'yes' if g.geometry is not None else 'no'})")
            v.opts.setdefault("cmap", cmap)
            v.opts.setdefault("tools", tools)
            v = self.session.add(v)
            self._view_tabs[v.name] = len(tabs)
            self._panes += v.panes()
            tabs.append((v.title, v.panel()))

        # Names the rest of the workspace and its users reach for.
        self.browser = getattr(self.session.view("gather"), "browser", None)
        self.map = getattr(self.session.view("geometry"), "map", None)
        self.fold = getattr(self.session.view("geometry"), "fold", None)
        sv = self.session.view("shot_volume")
        self.shot_volume = getattr(sv, "volume", None)
        sl = self.session.view("slices")
        self.slices = getattr(sl, "volume", None)
        self._shot_tab = self._view_tabs.get("gather")

        if self.browser is not None:
            self._redraws.append(self.browser.redraw)
            if self._tools_sidebar:
                cards = self.browser.tool_cards()
                if cards:
                    self._tool_cards = cards
                    self._tool_tabs.add(self._shot_tab)
        if sv is not None:
            sv._refresh_hook = self._redraw_shot_volume
            self._redraws.append(self._redraw_shot_volume)
        if sl is not None:
            sl._refresh_hook = self._redraw_slices
            self._redraws.append(self._redraw_slices)

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
            # a single gather has no shots, so the sheet lists no keys
            self._help = _gesture_help(shots=False,
                                       sidebar=self._tools_sidebar)
            if self._tools_sidebar:
                tabs.append(("Gather",
                             self._pane2d.frame(self._wt2d.figures())))
                self._tool_cards = [self._wt2d.window_card(True),
                                    self._wt2d.spectrum_card(True),
                                    self._pt2d.picking_card(True),
                                    self._pt2d.fb_card(True),
                                    self._help[0]]
                self._tool_tabs.add(len(tabs) - 1)
            else:
                tabs.append(("Gather", self._pane2d.frame(
                    self._wt2d.figures(),
                    pn.FlexBox(self._wt2d.window_card(),
                               self._wt2d.spectrum_card(),
                               self._pt2d.picking_card(),
                               self._pt2d.fb_card(), self._help[0]))))
            self._tab_2d = len(tabs) - 1

        self.tabs = pn.Tabs(*tabs, sizing_mode="stretch_width")
        if view is not None and view in self._view_tabs:
            self.tabs.active = self._view_tabs[view]
        # tab index -> what is on it, for everything that depends on the tab
        self._tab_names = {i: n for n, i in self._view_tabs.items()}
        if getattr(self, "_tab_2d", None) is not None:
            self._tab_names[self._tab_2d] = "gather2d"
        # the 2-D pane whose resolution / wiggle settings the sidebar shows
        self._pane_main = (self.browser.pane if self.browser is not None
                           else getattr(self, "_pane2d", None))
        self._info_panes = {}

        # The tool cards belong to the 2-D gather panel, so they are hidden
        # on the Geometry / Volume-slices tabs. Toggling one container (not
        # each card) keeps every card's collapsed state and the gesture
        # cheat-sheet's own visibility intact across tab switches.
        #
        # There is one box per group of cards rather than one overall,
        # because the shot-gather tools and the velocity tools belong to
        # different tabs and must not appear on each other's.
        self._tool_boxes = []
        for cards, tabset in ((self._tool_cards, self._tool_tabs),):
            if not cards:
                continue
            box = _theme.section("Tools", *cards,
                                 visible=self.tabs.active in tabset)
            self.tabs.param.watch(
                lambda e, box=box, tabset=tabset: setattr(
                    box, "visible", e.new in tabset), "active")
            self._tool_boxes.append(box)
        self._controls = self._make_controls(cmap, perc)
        self._build_sections()
        if self.slices is not None:
            # the whole-line volume is read the first time its tab is shown
            self.tabs.param.watch(self._on_tab_volume, "active")
            self._on_tab_volume(defer=False)
        self._key_docs = set()
        self._keychan = self._keys_js = None
        if keys:
            self._make_key_binder()

    # -- keyboard ----------------------------------------------------------
    def _make_key_binder(self):
        """Bind the keys, and only where there is something for them to move.

        A layout with neither shots to step through nor a volume to slice --
        a single gather -- gets no listener at all, rather than one that
        swallows keys and does nothing with them.
        """
        vols = [v for v in (self.slices, self.shot_volume) if v is not None]
        if getattr(self, "browser", None) is None and not vols:
            return
        keys = list(_KEYS)
        # the channel carries "[S-]<key>|<counter>" so that pressing the same
        # key twice is still two distinct property changes
        self._keychan = TextInput(value="", visible=False)
        self._keychan.on_change("value", lambda a, o, n: self._on_key(n))
        self._keys_js = CustomJS(args=dict(chan=self._keychan, keys=keys),
                                 code=_KEYS_JS)
        for v in vols:
            v.keys_hint.visible = True     # the keys are real now: say so

    def bind_keys(self):
        """Install the key listener in the session's document.

        DocumentReady rather than render time: the listener is global to the
        page, so it belongs to the document, and attaching it here keeps it
        out of the notebook case, where it would fight the notebook's own
        shortcuts.
        """
        doc = pn.state.curdoc
        if self._keys_js is None or doc is None or doc in self._key_docs:
            return
        self._key_docs.add(doc)
        doc.js_on_event(DocumentReady, self._keys_js)

    def _on_tab_volume(self, event=None, defer=True):
        if self._tab_names.get(self.tabs.active) == "slices":
            self.slices.ensure_ready(defer=defer)

    def _active_volume(self):
        """The volume view on the current tab, if the tab is one."""
        name = self._tab_names.get(self.tabs.active)
        return {"slices": self.slices, "shot_volume": self.shot_volume}.get(name)

    def _on_key(self, value):
        """One key press from the browser, as ``[S-]<key>|<counter>``.

        On a volume tab the digits choose a slice and the arrows move it;
        on every other tab the arrows step through shots. Shift makes the
        step a big one either way; Home and End go to either end.
        """
        key = value.rsplit("|", 1)[0]            # drop the repeat counter
        big = key.startswith("S-")
        key = key[2:] if big else key
        vol = self._active_volume()
        if vol is not None:
            if key in ("1", "2", "3"):
                vol.select(int(key) - 1)
            elif key in _SHOT_STEPS:
                vol.step(_SHOT_STEPS[key] * (vol.big_step() if big else 1))
            elif key in ("Home", "End"):
                vol.jump(key == "End")
            return
        br = getattr(self, "browser", None)
        if br is None:
            return
        if key in _SHOT_STEPS:
            br.set_shot(br.ishot + _SHOT_STEPS[key] * (10 if big else 1))
        elif key == "Home":
            br.set_shot(0)
        elif key == "End":
            br.set_shot(self.g.nshot - 1)

    # -- URL state ---------------------------------------------------------
    def sync_location(self):
        """Mirror the view state into the page URL (``?shot=37&cmap=gray``).

        Two-way: a URL carrying these parameters restores the view on load,
        and every later change rewrites the query string -- so the address
        bar is always a link a colleague can open on the same dataset, and
        the browser's back button undoes the last change.

        Only meaningful in a served session; in a notebook ``pn.state.location``
        is the notebook's own URL, which we must not touch.
        """
        loc = pn.state.location
        if loc is None or self._synced:
            return
        self._synced = True
        pairs = [(self._w_disp, "display"), (self._w_flip, "flip"),
                 (self._w_cmap, "cmap"), (self._w_perc, "clip"),
                 (self._w_gain, "gain"), (self._w_agcwin, "agcwin"),
                 (self._w_ftype, "filter")]
        pairs += [(w, f"f{i + 1}") for i, w in enumerate(self._w_f)]
        if getattr(self, "browser", None) is not None:
            pairs.append((self.browser.w_shot, "shot"))
            if len(self.browser.sorts) > 1:
                pairs.append((self.browser.w_sort, "sort"))
                if _sort.RECEIVER_LINE in self.browser.sorts:
                    pairs.append((self.browser.w_line, "recline"))
        pairs.append((self.tabs, "tab"))

        # A hand-edited or stale link can carry a value this build cannot use
        # (?cmap=nosuchmap). Panel applies it first and only then finds out,
        # which would leave the widget saying one thing and the panel showing
        # another, so put the old value back -- that re-runs the watchers and
        # the view is consistent again.
        def _guard(w, pname, qname, good):
            def _on_error(failed):
                print(f"[gathervis] URL: ignoring unusable {qname}="
                      f"{failed.get(pname)!r}")
                try:
                    setattr(w, pname, good)
                except Exception:
                    pass
            return _on_error

        for w, qname in pairs:
            pname = "active" if w is self.tabs else "value"
            loc.sync(w, {pname: qname},
                     on_error=_guard(w, pname, qname, getattr(w, pname)))
        # Everything above applies itself through its own watcher, except the
        # clip percentile: that one is wired to value_throttled (apply on
        # mouse release), which a URL-supplied value never fires.
        if self._w_perc.value != self.state["perc"]:
            _set_throttled(self._w_perc, self._w_perc.value)

    def _focus_view(self, name: str):
        """A view asked to be shown: switch to its tab if it has one."""
        tab = self._view_tabs.get(name)
        if tab is not None and getattr(self, "tabs", None) is not None:
            self.tabs.active = tab

    def _pick_shot(self, i: int):
        """Select a shot and show it. Kept as the public way in."""
        self.session.cursor.shot = int(i)
        self._focus_view("gather")

    def _draw2d(self):
        arr, clim = _process_gather(self.g.data, self.g.dt, self.state)
        self._pane2d.update(arr, clim, y0=self.g.t0, dy=self.g.dt)
        if hasattr(self, "_wt2d"):
            self._wt2d.refresh()

    def _redraw_shot_volume(self):
        """Keep the 4-D cuboid tab on the same filter/gain/polarity chain as
        the 2-D panel. One shot is small enough to process whole, so unlike
        the whole-line volume tab this needs no size guard."""
        arr, clim = _process_gather(self.g.shot(self.browser.ishot),
                                    self.g.dt, self.state)
        self.shot_volume.clim = clim
        self.shot_volume.set_volume(np.asarray(arr))

    def _redraw_slices(self):
        """The volume tab follows the filter/gain/flip chain too: volumes up
        to PROC_MAX_BYTES are processed whole (once per chain change) so all
        three slice planes -- including the time slice, which has no time
        axis of its own -- stay consistent. Larger memmaps fall back to raw
        with a visible note (processing them would defeat lazy loading)."""
        g = self.g
        chain = (self.state.get("filter") or self.state.get("gain")
                 or self.state.get("flip"))
        # shape, not .size: the data may be any lazy array-like
        nbytes = int(np.prod(g.data.shape)) * np.dtype(g.data.dtype).itemsize
        if chain and g.axes[-1] == "time" and nbytes <= PROC_MAX_BYTES:
            arr, clim = _process_gather(_as_float_array(g.data), g.dt,
                                        self.state)
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
        # kept on self: the keyboard shortcuts and the URL sync drive these
        # same widgets rather than a parallel copy of the state
        self._w_disp = w_disp = pn.widgets.RadioButtonGroup(
            name="display", options={"Density": "density", "Wiggle": "wiggle"},
            value="density", **_W)
        self._w_cmap = w_cmap = pn.widgets.Select(
            name="Colormap", options=list(_CMAPS), value=cmap, **_W)
        self._w_perc = w_perc = _theme.FloatSlider(
            name="clip percentile", start=80.0, end=100.0, step=0.5,
            value=perc, **_W)
        # negates the displayed gathers: wiggle fill lobes and density
        # colors swap (SEG normal <-> reverse); spectra and picks unaffected
        self._w_flip = w_flip = pn.widgets.Checkbox(name="Flip polarity",
                                                    value=False, **_W)

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
        self._proc_note = _hint(
            "This volume is too large to process in memory, so Volume "
            "slices shows the raw data.")
        self._proc_note.object = self._proc_note.object.replace(
            "gv-hint", "gv-warn")
        self._proc_note.visible = False
        widgets = [w_disp, w_flip, w_cmap, w_perc, self._proc_note]
        widgets += self._make_filter_controls()
        widgets += self._make_gain_controls()
        return widgets

    def _make_gain_controls(self):
        """AGC / trace balance on the displayed gathers. With gain active the
        color limits auto-rescale from the processed gather (the raw clim no
        longer applies to ~unit-amplitude output)."""
        self._w_gain = pn.widgets.Select(
            name="Gain", value="off", **_W,
            options={"Off": "off", "Trace balance": "trace balance",
                     "AGC": "AGC"})
        self._w_agcwin = pn.widgets.FloatInput(name="AGC window (s)",
                                               value=0.5, start=0.01, step=0.1,
                                               visible=False, **_W)

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
        f3-f4) applied to the displayed gathers. The whole-line volume tab
        follows it too, up to PROC_MAX_BYTES."""
        self._w_ftype = pn.widgets.Select(
            name="Filter", value="off", **_W,
            options={"Off": "off", "Low-pass": "low-pass",
                     "High-pass": "high-pass", "Band-pass": "band-pass"})
        mk = lambda n, v: pn.widgets.FloatInput(name=n, value=v, start=0.0,
                                                step=1.0,
                                                css_classes=["gv-bare"], **_W)
        self._w_f = [mk("f1 (Hz)", 5.0), mk("f2 (Hz)", 10.0),
                     mk("f3 (Hz)", 60.0), mk("f4 (Hz)", 80.0)]
        dash = lambda: pn.pane.HTML("<div class='gv-value gv-muted'>&ndash;"
                                    "</div>", margin=(4, 6), align="center")
        # low-cut ramp f1 -> f2, high-cut ramp f3 -> f4
        self._row_lo = _prop("Low cut (Hz)", self._w_f[0], dash(),
                             self._w_f[1],
                             tip="Cut below f1, pass above f2, a linear ramp "
                                 "between")
        self._row_hi = _prop("High cut (Hz)", self._w_f[2], dash(),
                             self._w_f[3],
                             tip="Pass below f3, cut above f4, a linear ramp "
                                 "between")
        self._row_lo.visible = self._row_hi.visible = False

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

    # -- sidebar -----------------------------------------------------------
    def _build_sections(self):
        """The sidebar, in the order a look at the data goes: how it is
        drawn, how it is processed, then what the current tab needs of its
        own, then the tools. Each section shows only where it applies (see
        ``_update_context``); the widgets are built once and shared by the
        served page and the notebook layout."""
        pane = self._pane_main
        self._row_mode = _prop("Mode", self._w_disp)
        self._row_perc = _prop(
            "Clip", self._w_perc,
            tip="Clip percentile: the colour limits sit at this percentile "
                "of |amplitude|. In wiggle mode it is the gain.")
        disp = [self._row_mode, self._w_cmap]
        if pane is not None:
            disp += [pane.w_res, pane.w_wig_n]
        disp += [self._row_perc, self._w_flip]
        self._sec_display = _theme.section("Display", *disp)
        self._sec_proc = _theme.section(
            "Processing", self._w_ftype, self._row_lo, self._row_hi,
            self._w_gain, self._w_agcwin, self._proc_note)
        self._view_sections = {}           # tab index -> its own sections
        for v in self.session.views:
            tab = self._view_tabs.get(v.name)
            secs = v.sidebar_sections() if hasattr(v, "sidebar_sections") else []
            if tab is not None and secs:
                self._view_sections.setdefault(tab, []).extend(secs)
        self._sections = [self._sec_display, self._sec_proc]
        for secs in self._view_sections.values():
            self._sections += secs
        self._sections += self._tool_boxes
        self.tabs.param.watch(self._update_context, "active")
        self._w_disp.param.watch(self._update_context, "value")
        self._update_context()

    def _update_context(self, *_):
        """Show the controls that affect what is on screen, and no others.

        Wiggle has no colormap and density no trace count; the geometry map
        takes neither display nor processing settings; a volume is always
        drawn as density, so it keeps the colormap but not the mode switch.
        """
        tab = self.tabs.active
        name = self._tab_names.get(tab)
        flat = name in ("gather", "gather2d")
        vol = name in ("slices", "shot_volume")
        wig = self._w_disp.value == "wiggle"
        self._sec_display.visible = self._sec_proc.visible = flat or vol
        self._row_mode.visible = flat
        self._w_cmap.visible = vol or not wig
        pane = self._pane_main
        if pane is not None:
            pane.w_res.visible = flat and not wig
            pane.w_wig_n.visible = flat and wig
        for t, secs in self._view_sections.items():
            for s in secs:
                s.visible = t == tab

    def sidebar(self, title=True):
        """The sidebar's objects. ``title=False`` leaves the dataset name
        out of the info block (the served page's header carries it)."""
        info = self._info_panes.get(title)
        if info is None:
            info = self._info_panes[title] = pn.pane.HTML(
                _info_html(self.g, title=title), sizing_mode="stretch_width",
                margin=(0, 0, 12, 0))
        items = [info, *self._sections]
        if self._keychan is not None:
            items.append(self._keychan)  # invisible; must be in the layout
        return items

    def panel(self):
        """Plain layout (notebook-friendly)."""
        side = pn.Column(*self.sidebar(), width=312, margin=(0, 20, 0, 0),
                         styles={"background": _theme.CANVAS,
                                 "padding": "14px 16px",
                                 "border-right": f"1px solid {_theme.LINE}"})
        return pn.Row(side, self.tabs, sizing_mode="stretch_width")

    def template(self, title="gathervis"):
        """Served page: header with the dataset's name, sidebar, work area."""
        header = []
        if self.g.name:
            header.append(pn.pane.HTML(
                f"<div class='gv-crumb'><span>"
                f"{_html.escape(str(self.g.name))}</span></div>",
                sizing_mode="stretch_width", margin=0))
        return _theme.GathervisTemplate(
            title=title, sidebar=self.sidebar(title=False), main=[self.tabs],
            header=header, sidebar_width=312)


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



# ---------------------------------------------------------------------------
# the explicit layer: build a session and put views in it
# ---------------------------------------------------------------------------
def _as_gathers(obj, axes=None, dt=1.0, t0=0.0, src=None, rec=None,
                shape=None, dtype="float32", name=None) -> Gathers:
    """Whatever the caller had -> a Gathers. Shared by show() and session()."""
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        if shape is None:
            raise ValueError("shape= is required when opening a raw binary file")
        return from_file(obj, shape, dtype=dtype, axes=axes, dt=dt, t0=t0,
                         src=src, rec=rec, name=name)
    if not isinstance(obj, Gathers):
        return from_array(obj, src=src, rec=rec, axes=axes, dt=dt, t0=t0,
                          name=name)
    if name is not None:
        obj.name = name
    return obj


def _per_session(first, make):
    """A factory handing out ``first`` once, then a fresh ``make()`` each call.

    Every browser session needs objects of its own: a bokeh model belongs to
    exactly one document, so a second tab (or a reload) that tried to reuse
    the first one's figures would render empty frames. The object built up
    front -- which is where construction errors surface -- serves the first
    session instead of being thrown away.
    """
    pending = [first]

    def build():
        return pending.pop() if pending else make()
    return build


def _serve(ws, port=None, address="127.0.0.1", title="gathervis", verbose=True,
           make=None):
    """Serve a built workspace, or hand back the Panel app when port is None.

    ``make`` builds another workspace like ``ws``; with it, every session
    after the first gets its own, so tabs and reloads are independent (and
    each one reads its own URL). Without it all sessions share ``ws``.
    """
    if port is None:
        return ws.panel()
    port = _resolve_port(port, address)
    if verbose:
        _banner(port)
    build = _per_session(ws, make) if make is not None else (lambda: ws)

    # Served as a factory, not as a finished object: pn.state.location only
    # exists inside a session, so the URL binding happens here rather than
    # while the Workspace is being built.
    def _open():
        w = build()
        tmpl = w.template(title)
        w.bind_keys()
        w.sync_location()
        return tmpl

    pn.serve(_open, port=port, address=address, show=False, title=title,
             websocket_origin="*")


def session(obj, axes=None, dt=1.0, t0=0.0, src=None, rec=None, shape=None,
            dtype="float32", name=None, cmap="gray", perc=98.0,
            tools="sidebar", views=None, keys=None):
    """A viewer you compose yourself, rather than one chosen for you.

        s = gv.session(data)
        s.add(gv.gather())
        s.add(gv.geometry())
        s.show(port=8080)

    :func:`show` is this function with the view list filled in for you --
    every view that the data supports, in the usual order. Reach for this one
    when that is not what you want: fewer tabs, a different order, or a view
    the automatic choice would not have included.

    ``views`` accepts the classes or instances to add; omit it to start empty
    and use ``.add()``. Views are in ``gathervis.viewer.VIEWS``, and each is
    also a plain function here -- :func:`gather`, :func:`geometry`,
    :func:`shot_volume`, :func:`slices`.

    Returns a :class:`Composer`: a session with ``add``, ``panel`` and
    ``show`` on it.
    """
    obj = _as_gathers(obj, axes=axes, dt=dt, t0=t0, src=src, rec=rec,
                      shape=shape, dtype=dtype, name=name)
    comp = Composer(obj, cmap=cmap, perc=perc, tools=tools, keys=keys)
    for v in (views or ()):
        comp.add(v)
    return comp


class Composer:
    """A session plus the plumbing that turns its views into a viewer.

    Kept apart from ``Workspace`` on purpose: a workspace decides which views
    a dataset gets, and this decides nothing -- it shows what it was given,
    in the order it was given. The shared display controls, the sidebar, the
    URL sync and the keyboard are the same either way, so they live in
    ``Workspace`` and this drives it once the view list is settled.
    """

    def __init__(self, data, cmap="gray", perc=98.0, tools="sidebar",
                 keys=None):
        self.data = data
        self._opts = dict(cmap=cmap, perc=perc, tools=tools, keys=keys)
        self._views = []
        self._built = None

    def add(self, view):
        """Add a view -- a class, or an instance with options set."""
        if self._built is not None:
            raise RuntimeError("add views before the viewer is built")
        self._views.append(view)
        return self

    def build(self) -> "Workspace":
        """The workspace holding exactly the views that were added."""
        if self._built is None:
            keys = self._opts["keys"]
            self._built = Workspace(
                self.data, cmap=self._opts["cmap"], perc=self._opts["perc"],
                tools=self._opts["tools"], keys=bool(keys),
                views=self._views or None)
        return self._built

    def _fresh(self) -> "Workspace":
        """Another workspace with the same views, for another session. View
        instances carry their session, so each one is re-made from its
        options rather than shared."""
        views = [type(v)(**dict(v.opts)) if isinstance(v, _View) else v
                 for v in self._views]
        return Workspace(self.data, cmap=self._opts["cmap"],
                         perc=self._opts["perc"], tools=self._opts["tools"],
                         keys=bool(self._opts["keys"]), views=views or None)

    @property
    def session(self):
        return self.build().session

    def panel(self):
        return self.build().panel()

    def show(self, port=None, address="127.0.0.1", title="gathervis",
             verbose=True):
        """Serve it, or return the Panel app when ``port`` is None."""
        return _serve(self.build(), port=port, address=address, title=title,
                      make=self._fresh,
                      verbose=verbose)

    def __repr__(self):
        names = [getattr(v, "name", getattr(v, "__name__", "?"))
                 for v in self._views]
        return f"Composer({self.data!r}, views={names})"


def gather(**kw):
    """The shot browser, as a view to put in a session."""
    return GatherView(**kw)


def geometry(**kw):
    """The acquisition map, as a view to put in a session."""
    return GeometryView(**kw)


def shot_volume(**kw):
    """The current shot as a cuboid, as a view to put in a session."""
    return ShotVolumeView(**kw)


def slices(**kw):
    """A volume with slice planes, as a view to put in a session."""
    return SlicesView(**kw)


def show(obj, axes=None, view=None, dt=1.0, t0=0.0, src=None, rec=None,
         shape=None, dtype="float32", name=None, cmap="gray", perc=98.0,
         port=None, address="127.0.0.1", title="gathervis", verbose=True,
         tools="sidebar", keys=None):
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
    keys : bool, optional
        The left/right arrow keys step through shots (the only shortcut,
        and only where there is a shot axis). Default: on when served, off
        in a notebook, where a document-level key listener would fight the
        notebook's own.
        Served views additionally mirror their state into the URL, so the
        address bar stays a shareable link to exactly what is on screen.
    """
    import time
    t_start = time.perf_counter()
    if view not in (None, "browse", "slices"):
        raise ValueError("view must be None, 'browse' or 'slices'")
    if tools not in ("sidebar", "below"):
        raise ValueError("tools must be 'sidebar' or 'below'")
    obj = _as_gathers(obj, axes=axes, dt=dt, t0=t0, src=src, rec=rec,
                      shape=shape, dtype=dtype, name=name)
    t_data = time.perf_counter()

    kw = dict(cmap=cmap, perc=perc, view=view, tools=tools,
              keys=(port is not None) if keys is None else keys)
    ws = Workspace(obj, **kw)
    t_app = time.perf_counter()
    if verbose:
        print(f"[gathervis] dataset {tuple(obj.shape)} ready in "
              f"{t_data - t_start:.2f}s | app built in {t_app - t_data:.2f}s")

    return _serve(ws, port=port, address=address, title=title, verbose=verbose,
                  make=lambda: Workspace(obj, **kw))
