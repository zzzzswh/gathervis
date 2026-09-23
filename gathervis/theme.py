"""The look of gathervis, in one place.

One Panel design (:class:`GathervisDesign`) styles every widget from a single
stylesheet, ``assets/gathervis.css``; one page template
(:class:`GathervisTemplate`) frames the served app; and :func:`style_figure`
gives every bokeh figure the same quiet axes. The viewer decides *what* is on
screen and this module decides what it looks like, so either can change
without touching the other.

Nothing here is fetched from outside -- no web fonts, no icon CDN. gathervis
is usually opened through an SSH tunnel, often from networks where a blocked
font request holds up the first paint for seconds. The type is the
platform's own UI face, which also covers CJK text in dataset names.
"""
from __future__ import annotations

import base64
import html as _html
import pathlib

import param
import panel as pn
from panel.template import VanillaTemplate
from panel.theme.base import Design, Inherit
from panel.theme.native import Native
from panel.viewable import Viewable

__all__ = ["GathervisDesign", "GathervisTemplate", "style_figure",
           "IntSlider", "FloatSlider", "prop", "section", "pair", "hint",
           "kv", "ICONS"]

_ASSETS = pathlib.Path(__file__).parent / "assets"

# -- tokens (mirrored as CSS variables in assets/*.css) -------------------
INK = "#17202a"          # text
INK_2 = "#52606d"        # secondary text, axis labels
INK_3 = "#7b8794"        # tertiary: chevrons, placeholders
LINE = "#d5dbe1"         # input borders, plot frame
LINE_2 = "#e7ebef"       # hairlines, grid
CANVAS = "#f6f8fa"       # sidebar
SURFACE = "#ffffff"
ACCENT = "#0b6b73"       # state only: active tab, primary action, focus
ACCENT_SOFT = "#e2f0f1"
WARN = "#9a5b00"
CURRENT = "#f0642a"      # "the one you are on": current shot, spread, slice
FONT = ('"Segoe UI Variable Text", "Segoe UI", -apple-system, '
        'BlinkMacSystemFont, "Helvetica Neue", "PingFang SC", '
        '"Hiragino Sans GB", "Microsoft YaHei", "Noto Sans", '
        '"Noto Sans CJK SC", Arial, sans-serif')
LABEL_W = 92             # the property label column, sidebar and cards alike


# -- design + template ----------------------------------------------------
class GathervisDesign(Native):
    """Native widgets, restyled by one stylesheet shipped with the package."""

    modifiers = {
        Viewable: {"stylesheets": [Inherit, "assets/gathervis.css"]},
    }


class GathervisTemplate(VanillaTemplate):
    """A plain page: white header, grey sidebar, white work area."""

    design = param.ClassSelector(
        class_=Design, default=GathervisDesign, is_instance=False,
        instantiate=False, doc="The design applied to every component.")

    # a single path in Panel 1.4, a list in later releases
    _css = [*(VanillaTemplate._css if isinstance(VanillaTemplate._css,
                                                  (list, tuple))
              else [VanillaTemplate._css]), _ASSETS / "page.css"]

    # VanillaTemplate pulls Lato from Google Fonts; nothing from outside.
    _resources = {"css": {}}


# -- widgets --------------------------------------------------------------
def _quiet(base):
    """``base``, but its name stays in Python instead of titling the widget.

    A slider's own title reads ``name: value`` above the track. In a
    property row the label already sits in the label column, so the slider
    should show its value and nothing else; keeping ``name`` on the Python
    side means code and tests can still find it by name.
    """
    rename = dict(base._rename)
    rename["name"] = None
    if "label" in base.param:            # Panel >= 1.8 titles from label
        rename["label"] = None
    return type(f"Quiet{base.__name__}", (base,),
                {"_rename": rename, "__module__": __name__,
                 "__doc__": base.__doc__})


IntSlider = _quiet(pn.widgets.IntSlider)
FloatSlider = _quiet(pn.widgets.FloatSlider)


def _row_label(text, tip=""):
    tip = f" title='{_html.escape(tip, quote=True)}'" if tip else ""
    return pn.pane.HTML(f"<div class='gv-label'{tip}>{text}</div>",
                        width=LABEL_W, margin=(4, 8, 4, 0), align="center")


def prop(label, *widgets, tip=""):
    """One property row: label column on the left, control(s) on the right."""
    return pn.Row(_row_label(label, tip), *widgets, margin=0,
                  sizing_mode="stretch_width", css_classes=["gv-prop"])


def pair(*widgets, **kw):
    """Controls sharing one row equally, with a gutter between them."""
    return pn.Row(*widgets, margin=0, sizing_mode="stretch_width",
                  styles={"gap": "8px"}, **kw)


def section(title, *objects, first=False, visible=True, aside=""):
    """A titled group of controls, separated from the one above by a rule."""
    cls = "gv-sec gv-first" if first else "gv-sec"
    aside = f"<small>{aside}</small>" if aside else ""
    head = pn.pane.HTML(f"<div class='{cls}'><span>{title}</span>{aside}</div>",
                        sizing_mode="stretch_width", margin=(0, 0, 2, 0))
    return pn.Column(head, *objects, margin=(0, 0, 6, 0), visible=visible,
                     sizing_mode="stretch_width", css_classes=["gv-section"])


def hint(html, warn=False, **kw):
    """The one small-print style."""
    cls = "gv-warn" if warn else "gv-hint"
    kw.setdefault("margin", (2, 0, 4, 0))
    return pn.pane.HTML(f"<div class='{cls}'>{html}</div>",
                        sizing_mode="stretch_width", **kw)


def kv(rows):
    """``[(label, value), ...]`` as an aligned definition list (HTML)."""
    body = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return f"<dl class='gv-kv'>{body}</dl>"


def icon_kw(cls, icon, text=""):
    """Keyword arguments giving widget class ``cls`` an icon, if it can take
    one (older Panel builds cannot), falling back to a text label."""
    key = "label" if "label" in cls.param else "name"
    if "icon" in cls.param:
        return {"icon": ICONS[icon], key: ""}
    return {key: text or icon}


# -- figures --------------------------------------------------------------
def style_figure(fig, grid=True):
    """Quiet axes, hairline grid, system type, a toolbar that stays out of
    the way until the pointer is over the plot."""
    fig.toolbar.logo = None
    fig.toolbar.autohide = True
    fig.border_fill_color = SURFACE
    fig.outline_line_color = LINE
    for ax in fig.axis:
        ax.axis_label_text_font = FONT
        ax.axis_label_text_font_size = "12px"
        ax.axis_label_text_font_style = "normal"
        ax.axis_label_text_color = INK_2
        ax.axis_label_standoff = 8
        ax.major_label_text_font = FONT
        ax.major_label_text_font_size = "11px"
        ax.major_label_text_color = INK_2
        ax.axis_line_color = LINE
        ax.major_tick_line_color = LINE
        ax.minor_tick_line_color = None
        ax.major_tick_in, ax.major_tick_out = 0, 4
    for g in fig.grid:
        g.grid_line_color = LINE_2 if grid else None
    for leg in fig.legend:
        style_legend(leg)
    return fig


def style_legend(leg):
    leg.label_text_font = FONT
    leg.label_text_font_size = "11.5px"
    leg.label_text_color = INK_2
    leg.border_line_color = None
    leg.background_fill_alpha = 0.0
    leg.glyph_width = leg.glyph_height = 12
    leg.spacing, leg.padding, leg.margin = 4, 4, 0
    return leg


def style_colorbar(cb):
    cb.title_text_font = cb.major_label_text_font = FONT
    cb.title_text_font_style = "normal"
    cb.title_text_font_size = "11.5px"
    cb.major_label_text_font_size = "11px"
    cb.title_text_color = cb.major_label_text_color = INK_2
    cb.major_tick_line_color = None
    cb.bar_line_color = None
    cb.background_fill_alpha = 0.0
    return cb


def set_tool_icon(tool, name):
    """A tool icon that says what the tool draws (bokeh gives every point
    tool the same one, so three of them in a toolbar are indistinguishable)."""
    if "icon" not in tool.properties():
        return tool
    try:
        tool.icon = _TOOL_ICONS[name]
    except Exception:           # an older bokeh that rejects data URIs
        pass
    return tool


# -- icons ----------------------------------------------------------------
def _svg(body, size=16, sw=1.8):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' "
            f"height='{size}' viewBox='0 0 24 24' fill='none' "
            f"stroke='currentColor' stroke-width='{sw}' stroke-linecap='round' "
            f"stroke-linejoin='round'>{body}</svg>")


ICONS = {
    "reset": _svg("<path d='M4 12a8 8 0 1 0 2.4-5.7'/><path d='M4 4v4h4'/>"),
    "fit": _svg("<path d='M5 3h14M5 21h14'/><path d='M12 6.5v11'/>"
                "<path d='M9 9.5l3-3 3 3M9 14.5l3 3 3-3'/>"),
    "fullscreen": _svg("<path d='M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5'/>"),
    "download": _svg("<path d='M12 4v11'/><path d='M7.5 10.5 12 15l4.5-4.5'/>"
                     "<path d='M5 20h14'/>"),
    "camera": _svg("<path d='M3 12h3M18 12h3M12 3v3M12 18v3'/>"
                   "<circle cx='12' cy='12' r='4.5'/>"),
}


def _data_uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(
        svg.encode()).decode()


def _tool_svg(body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' "
            f"viewBox='0 0 24 24'>{body}</svg>")


_TOOL_ICONS = {k: _data_uri(_tool_svg(v)) for k, v in {
    # analysis windows: the window colour, the window shape
    "box": "<rect x='4' y='6' width='16' height='12' rx='1.5' fill='#e6741e' "
           "fill-opacity='.14' stroke='#e6741e' stroke-width='2'/>",
    "poly": "<path d='M5 17 7 6.5l8.5-2.5L20 13l-6.5 6z' fill='#e6741e' "
            "fill-opacity='.14' stroke='#e6741e' stroke-width='2' "
            "stroke-linejoin='round'/>",
    # event picks: the marker that is drawn
    "pick": "<path d='M3 8.5h18' stroke='#d81b60' stroke-width='1.6' "
            "stroke-dasharray='2 2.2'/><path d='M7 10.5h10L12 19z' "
            "fill='#d81b60'/>",
    # mutes: the line, and the side it removes
    "mute_top": "<path d='M3 3h18v4.5L12 12l-9 3.5z' fill='#1e88e5' "
                "fill-opacity='.18'/><path d='M3 15.5 12 12l9-4.5' "
                "stroke='#1e88e5' stroke-width='2' fill='none'/>"
                "<rect x='10' y='10' width='4' height='4' fill='#1e88e5'/>",
    "mute_bottom": "<path d='M3 21h18v-8.5L12 10l-9 3z' fill='#f4511e' "
                   "fill-opacity='.18'/><path d='M3 13 12 10l9 2.5' "
                   "stroke='#f4511e' stroke-width='2' fill='none'/>"
                   "<rect x='10' y='8' width='4' height='4' fill='#f4511e'/>",
    "vpick": "<path d='M6 20 9 13l5-4 4-6' stroke='#d81b60' stroke-width='2' "
             "fill='none'/><circle cx='9' cy='13' r='2.6' fill='#d81b60'/>"
             "<circle cx='14' cy='9' r='2.6' fill='#d81b60'/>",
}.items()}
