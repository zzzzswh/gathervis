"""Offscreen full-resolution rendering: gather -> RGB raster -> PNG bytes.

The interactive panes deliberately ship a *decimated* uint8 image (``MAX_PX``)
so the wire payload stays small. This module re-renders the very same gather
at its native sample resolution -- one pixel per (trace, sample) by default --
and encodes it as a PNG using nothing but numpy and the standard library
(``zlib``), so "download the original image" costs no new dependency and no
headless-browser screenshot.

It also owns the color palettes, because both consumers need them in a
different form: the browser panes want hex strings (bokeh / plotly), the
rasterizer wants a uint8 lookup table.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

from .process import quantize

__all__ = ["CMAPS", "palette_hex", "palette_rgb", "raster_size", "auto_px",
           "render_gather", "render_density", "render_wiggle", "png_bytes",
           "save_png"]

# variable-density colormaps: (positions, rgb anchors). 'petrel' anchors are
# taken from cigvis's customcmap (MIT), the rest are ours.
PALETTES = {
    "seismic": ([0.0, 0.25, 0.5, 0.75, 1.0],          # blue - white - red
                [[0.0, 0.0, 0.45], [0.1, 0.3, 1.0], [1.0, 1.0, 1.0],
                 [1.0, 0.25, 0.1], [0.45, 0.0, 0.0]]),
    "gray": ([0.0, 1.0], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    "petrel": ([0.0, 0.33, 0.4, 0.5, 0.6, 0.67, 1.0],  # cyan-blue-gray-red-yellow
               [[0.631, 1.0, 1.0], [0.0, 0.0, 0.749],
                [0.302, 0.302, 0.302], [0.8, 0.8, 0.8],
                [0.380, 0.271, 0.0], [0.749, 0.0, 0.0], [1.0, 1.0, 0.0]]),
    "rainbow": ([0.0, 0.125, 0.375, 0.625, 0.875, 1.0],  # jet-style
                [[0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
                 [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
}
CMAPS = tuple(PALETTES) + tuple(f"{n}_r" for n in PALETTES)

WIGGLE_CLIP = 2.0            # excursion clip, in trace spacings (as on screen)
WIGGLE_FG = (0x11, 0x11, 0x11)   # same ink as the bokeh wiggle/fill renderers
WIGGLE_BG = (0xFF, 0xFF, 0xFF)
AUTO_PX_TRACE = {"density": 1, "wiggle": 8}   # 0 = auto -> these
MAX_EXPORT_PIXELS = 64_000_000   # refuse bigger rasters (memory + PNG time)
_BLOCK_CELLS = 2_000_000         # wiggle scanline accumulator, cells per block


def palette_rgb(name: str) -> np.ndarray:
    """(256, 3) uint8 lookup table for a colormap name ('_r' = reversed)."""
    base = name[:-2] if name.endswith("_r") else name
    if base not in PALETTES:
        raise ValueError(f"unknown cmap {name!r}; available: {CMAPS}")
    pos, anchors = PALETTES[base]
    pos, anchors = np.asarray(pos), np.asarray(anchors)
    t = np.linspace(0.0, 1.0, 256)
    rgb = np.stack([np.interp(t, pos, anchors[:, c]) for c in range(3)], axis=1)
    if name.endswith("_r"):
        rgb = rgb[::-1]
    return (rgb * 255 + 0.5).astype(np.uint8)


def palette_hex(name: str) -> list:
    """256 '#rrggbb' strings for bokeh / plotly color mappers."""
    return ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in palette_rgb(name)]


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------
def auto_px(display: str, px_trace: int = 0, px_sample: int = 0):
    """Resolve the '0 = auto' pixel settings to concrete values.

    Density defaults to the true original resolution (1 px per trace and per
    sample); wiggle needs room to draw an excursion, so a trace gets 8 px.
    """
    pt = int(px_trace) or AUTO_PX_TRACE.get(display, 1)
    return max(1, pt), max(1, int(px_sample) or 1)


def raster_size(nx: int, nt: int, display: str = "density", px_trace: int = 0,
                px_sample: int = 0):
    """(width, height) in pixels of the raster ``render_gather`` would make."""
    pt, ps = auto_px(display, px_trace, px_sample)
    w = nx * pt
    if display == "wiggle":
        w += 2 * int(round(WIGGLE_CLIP * pt))     # room for the clipped swing
    return int(w), int(nt * ps)


# ---------------------------------------------------------------------------
# rasterizers
# ---------------------------------------------------------------------------
def render_gather(arr2d, clim, display: str = "density", cmap: str = "seismic",
                  px_trace: int = 0, px_sample: int = 0,
                  flip_y: bool = True) -> np.ndarray:
    """Render a processed (nx, nt) gather as an (h, w, 3) uint8 RGB raster.

    ``arr2d`` is expected to be exactly what the pane holds: filtering, gain
    and polarity are already applied, and ``clim`` is the color range that
    goes with it. ``flip_y=True`` is the seismic convention (time downward),
    i.e. row 0 of the raster is the first sample.
    """
    if display not in ("density", "wiggle"):
        raise ValueError(f"display must be 'density' or 'wiggle', got {display!r}")
    if display == "wiggle":
        return render_wiggle(arr2d, clim, px_trace=px_trace,
                             px_sample=px_sample, flip_y=flip_y)
    return render_density(arr2d, clim, cmap=cmap, px_trace=px_trace,
                          px_sample=px_sample, flip_y=flip_y)


def render_density(arr2d, clim, cmap: str = "seismic", px_trace: int = 0,
                   px_sample: int = 0, flip_y: bool = True) -> np.ndarray:
    """Variable-density raster: the screen image without the decimation.

    Quantizes to uint8 against ``clim`` exactly like the on-screen panel, then
    looks the colors up in the same 256-entry palette the browser was given.
    """
    a = np.asanyarray(arr2d)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D (trace, sample) gather, got {a.shape}")
    pt, ps = auto_px("density", px_trace, px_sample)
    _check_budget(*raster_size(a.shape[0], a.shape[1], "density", pt, ps))
    img = quantize(a, clim).T                     # (nt, nx): rows = samples
    rgb = palette_rgb(cmap)[img]                  # (nt, nx, 3)
    if ps > 1:
        rgb = np.repeat(rgb, ps, axis=0)
    if pt > 1:
        rgb = np.repeat(rgb, pt, axis=1)
    return rgb if flip_y else rgb[::-1]


def render_wiggle(arr2d, clim, px_trace: int = 0, px_sample: int = 0,
                  flip_y: bool = True, line_px: int = 1,
                  fg=WIGGLE_FG, bg=WIGGLE_BG) -> np.ndarray:
    """Wiggle + positive-lobe (variable area) raster, every trace drawn.

    Unlike the interactive panel -- which decimates to ``MAX_WIGGLE`` traces so
    the browser stays responsive -- this draws *all* traces of the gather.
    Amplitudes are normalized by ``clim`` and clipped at ``WIGGLE_CLIP`` trace
    spacings, matching what the screen shows.
    """
    a = np.asarray(arr2d, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D (trace, sample) gather, got {a.shape}")
    nx, nt = a.shape
    pt, ps = auto_px("wiggle", px_trace, px_sample)
    w, h = raster_size(nx, nt, "wiggle", pt, ps)
    _check_budget(w, h)
    pad = int(round(WIGGLE_CLIP * pt))

    scale = max(float(clim[1]), 1e-30)            # same normalization as bokeh
    amp = np.clip(a / scale, -WIGGLE_CLIP, WIGGLE_CLIP) * pt
    base = pad + pt * (np.arange(nx, dtype=np.float32) + 0.5)
    x = base[:, None] + amp                       # (nx, nt) pixel columns
    if ps > 1:                                    # interpolate onto the rows
        tq = np.arange(h, dtype=np.float32) / ps
        k0 = np.floor(tq).astype(np.int64)
        k1 = np.minimum(k0 + 1, nt - 1)
        f = (tq - k0).astype(np.float32)
        x = x[:, k0] * (1.0 - f) + x[:, k1] * f
    xi = np.rint(x).astype(np.int32)              # (nx, h)
    bi = np.rint(base).astype(np.int32)[:, None]
    nxt = np.empty_like(xi)                       # next row's column, per trace
    nxt[:, :-1] = xi[:, 1:]
    nxt[:, -1] = xi[:, -1]

    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:] = np.asarray(bg, dtype=np.uint8)
    ink = np.asarray(fg, dtype=np.uint8)
    rows = np.arange(h, dtype=np.int64)
    blk = max(1, _BLOCK_CELLS // (w + 1))         # bound the accumulator
    for r0 in range(0, h, blk):
        r1 = min(h, r0 + blk)
        xs, bs, ns_ = xi[:, r0:r1], bi, nxt[:, r0:r1]
        rr = np.broadcast_to(rows[r0:r1] - r0, xs.shape)
        fill = xs > bs                            # variable-area lobe
        lo = np.concatenate([np.broadcast_to(bs, xs.shape)[fill],
                             np.minimum(xs, ns_).ravel()])
        hi = np.concatenate([xs[fill] + 1,
                             np.maximum(xs, ns_).ravel() + max(1, line_px)])
        rr = np.concatenate([rr[fill], rr.ravel()])
        m = _span_mask(rr, lo, hi, r1 - r0, w)
        out[r0:r1][m] = ink
    return out if flip_y else out[::-1]


def _span_mask(rows, lo, hi, h, w) -> np.ndarray:
    """(h, w) bool mask of the half-open column spans [lo, hi) on ``rows``.

    A difference array + cumsum: one bincount per edge, no Python loop over
    the (trace, sample) spans.
    """
    lo = np.clip(lo, 0, w)
    hi = np.clip(hi, 0, w)
    keep = hi > lo
    rows, lo, hi = rows[keep], lo[keep], hi[keep]
    stride = w + 1
    off = rows.astype(np.int64) * stride
    n = h * stride
    d = (np.bincount(off + lo, minlength=n)
         - np.bincount(off + hi, minlength=n)).reshape(h, stride)
    return np.cumsum(d[:, :w], axis=1) > 0


def _check_budget(w: int, h: int):
    if w * h > MAX_EXPORT_PIXELS:
        raise MemoryError(
            f"{w} x {h} px = {w * h / 1e6:.0f} MP exceeds the "
            f"{MAX_EXPORT_PIXELS / 1e6:.0f} MP export budget; zoom into a "
            "window, or lower px/trace and px/sample")


# ---------------------------------------------------------------------------
# PNG encoding (stdlib only: zlib + struct)
# ---------------------------------------------------------------------------
def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def png_bytes(rgb: np.ndarray, level: int = None) -> bytes:
    """Encode an (h, w, 3) uint8 array as an 8-bit RGB PNG.

    Rows are written with PNG's 'Up' filter (delta against the row above) --
    three lines of numpy that typically halve the file for seismic images --
    and fed to zlib row by row, so a large raster is never duplicated in
    memory. ``level`` defaults to 6, dropping to 3 above 8 MP where full
    compression would make the user wait for little gain.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (h, w, 3) RGB array, got {rgb.shape}")
    h, w = rgb.shape[:2]
    if level is None:
        level = 6 if h * w <= 8_000_000 else 3
    flat = rgb.reshape(h, w * 3)
    co = zlib.compressobj(level)
    line = np.empty(w * 3 + 1, dtype=np.uint8)
    line[0] = 2                                   # filter type 2 = Up
    prev = np.zeros(w * 3, dtype=np.uint8)        # row -1 is all zeros
    parts = []
    for r in range(h):
        cur = flat[r]
        line[1:] = cur - prev                     # uint8 wraps: PNG semantics
        parts.append(co.compress(line.tobytes()))
        prev = cur
    parts.append(co.flush())
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", b"".join(parts)) + _chunk(b"IEND", b""))


def save_png(path, arr2d, clim=None, display: str = "density",
             cmap: str = "seismic", perc: float = 98.0, px_trace: int = 0,
             px_sample: int = 0, flip_y: bool = True, level: int = None):
    """Write a gather straight to a full-resolution PNG, no viewer involved.

    The scriptable twin of the viewer's "download original image" button, for
    batch figures::

        from gathervis.process import bandpass, agc
        from gathervis.render import save_png
        a = agc(bandpass(g.shot(7), g.dt, f3=60, f4=80), g.dt)
        save_png("shot007.png", a, cmap="gray")

    ``arr2d`` is (trace, sample) and is taken as already processed; ``clim``
    defaults to the robust symmetric limits at ``perc``, matching the viewer's
    clip-percentile slider.
    """
    from .process import robust_clim
    if clim is None:
        clim = robust_clim(np.asanyarray(arr2d), perc)
    rgb = render_gather(arr2d, clim, display=display, cmap=cmap,
                        px_trace=px_trace, px_sample=px_sample, flip_y=flip_y)
    with open(path, "wb") as f:
        f.write(png_bytes(rgb, level=level))
    return path
