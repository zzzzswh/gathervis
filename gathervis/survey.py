"""CMP binning and the acquisition-QC maps that come out of it.

Everything here works on ``Geometry`` alone -- no trace data is touched, so a
fold map of a 100 GB survey costs whatever the coordinate arrays cost and
nothing more.

The one concept: every source-receiver pair has a **midpoint**, and the survey
is cut into rectangular **bins**; the number of midpoints landing in a bin is
its **fold**. Fold is the first thing anyone looks at after loading geometry,
because a hole in the fold map is a hole in the image, and it shows up here
long before anything has been processed.

Bin size defaults to half the receiver station interval in-line and half the
receiver-line interval cross-line, which is the usual choice: the natural CMP
sampling of a straight line is half the station spacing.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["BinGrid", "midpoints", "default_bin", "bin_grid", "fold",
           "traces_in_bin", "offsets_azimuths_in_bin", "azimuth_sectors",
           "MAX_BINS"]

MAX_BINS = 4_000_000        # refuse grids finer than this (a 2000x2000 map)
_CHUNK = 2_000_000          # midpoints computed per chunk


class BinGrid(NamedTuple):
    """A regular CMP bin grid and the fold counted on it.

    ``counts`` is indexed ``[iy, ix]`` -- row = cross-line, column = in-line --
    so it drops straight into a bokeh image glyph anchored at
    ``(x0, y0)`` with extent ``(nx * dx, ny * dy)``.
    """

    counts: np.ndarray          # (ny, nx) int32 traces per bin
    x0: float
    y0: float
    dx: float
    dy: float

    @property
    def shape(self):
        return self.counts.shape

    @property
    def extent(self):
        ny, nx = self.counts.shape
        return (self.x0, self.y0, nx * self.dx, ny * self.dy)

    @property
    def nlive(self) -> int:
        """Bins with at least one trace in them."""
        return int(np.count_nonzero(self.counts))

    def bin_of(self, x: float, y: float):
        """(ix, iy) of the bin containing a map position, or None if outside."""
        ny, nx = self.counts.shape
        ix = int(np.floor((x - self.x0) / self.dx))
        iy = int(np.floor((y - self.y0) / self.dy))
        if 0 <= ix < nx and 0 <= iy < ny:
            return ix, iy
        return None


def _shot_pairs(geo, ishot: int):
    """(src (2,), rec (nr, 2)) for one shot, as map coordinates."""
    return geo.src[ishot][:2], geo.rec_for(ishot)[:, :2]


def midpoints(geo, ishot=None) -> np.ndarray:
    """Source-receiver midpoints, ``(n, 2)``.

    With ``ishot`` given, only that shot's; otherwise the whole survey, which
    is ``ns * nr`` points -- built shot by shot so the peak allocation is one
    shot, not the survey.
    """
    if ishot is not None:
        src, rec = _shot_pairs(geo, int(ishot))
        return 0.5 * (rec + src)
    out = np.empty((geo.ns * geo.nr, 2), dtype=np.float64)
    for i in range(geo.ns):
        src, rec = _shot_pairs(geo, i)
        out[i * geo.nr:(i + 1) * geo.nr] = 0.5 * (rec + src)
    return out


def _spacing(v: np.ndarray) -> float:
    """Typical gap between distinct coordinate values along one axis."""
    u = np.unique(np.round(np.asarray(v, dtype=float), 6))
    if u.size < 2:
        return 0.0
    return float(np.median(np.diff(u)))


def default_bin(geo) -> tuple:
    """(dx, dy): half the receiver spacing on each axis, with fallbacks.

    A survey laid out on one line has no cross-line spacing to measure, and a
    single-point spread has neither; rather than divide by zero, the missing
    axis borrows the other one, and a survey with no spacing at all gets a
    bin of 1 map unit.
    """
    rec = geo.rec.reshape(-1, geo.rec.shape[-1])[:, :2]
    dx, dy = _spacing(rec[:, 0]) / 2.0, _spacing(rec[:, 1]) / 2.0
    if dx <= 0 and dy <= 0:
        return 1.0, 1.0
    return (dx or dy), (dy or dx)


def bin_grid(geo, bin_size=None, max_bins: int = MAX_BINS) -> tuple:
    """(x0, y0, dx, dy, nx, ny) covering every midpoint of the survey.

    Raises if the requested bin size would make a grid larger than
    ``max_bins`` -- a typo in a bin size otherwise allocates the machine.
    """
    dx, dy = default_bin(geo) if bin_size is None else bin_size
    dx, dy = float(dx), float(dy)
    if dx <= 0 or dy <= 0:
        raise ValueError(f"bin size must be positive, got {(dx, dy)}")
    mid = midpoints(geo)
    x0, y0 = float(mid[:, 0].min()), float(mid[:, 1].min())
    nx = int(np.floor((mid[:, 0].max() - x0) / dx)) + 1
    ny = int(np.floor((mid[:, 1].max() - y0) / dy)) + 1
    if nx * ny > max_bins:
        raise ValueError(
            f"bin size {(dx, dy)} gives a {nx} x {ny} grid "
            f"({nx * ny / 1e6:.1f}M bins); use a coarser bin")
    return x0, y0, dx, dy, nx, ny


def fold(geo, bin_size=None, max_bins: int = MAX_BINS) -> BinGrid:
    """CMP fold: how many traces fall in each bin.

    Counted with ``bincount`` over chunks of midpoints, so the whole survey
    never exists as one index array.
    """
    x0, y0, dx, dy, nx, ny = bin_grid(geo, bin_size, max_bins)
    counts = np.zeros(nx * ny, dtype=np.int64)
    for start in range(0, geo.ns, max(1, _CHUNK // max(geo.nr, 1))):
        stop = min(start + max(1, _CHUNK // max(geo.nr, 1)), geo.ns)
        mid = np.concatenate([midpoints(geo, i) for i in range(start, stop)])
        ix = np.floor((mid[:, 0] - x0) / dx).astype(np.int64)
        iy = np.floor((mid[:, 1] - y0) / dy).astype(np.int64)
        np.clip(ix, 0, nx - 1, out=ix)           # guard float edge cases
        np.clip(iy, 0, ny - 1, out=iy)
        counts += np.bincount(iy * nx + ix, minlength=nx * ny)
    return BinGrid(counts.reshape(ny, nx).astype(np.int32), x0, y0, dx, dy)


def _hits_in_bin(geo, grid: BinGrid, ix: int, iy: int):
    """``(shots, recs, vectors)`` of every trace whose midpoint is in one bin.

    ``recs`` is the receiver's index *within its shot*, which is what turns a
    bin into an addressable set of traces -- offsets and azimuths alone
    describe the illumination but cannot fetch it.
    """
    ny, nx = grid.shape
    if not (0 <= ix < nx and 0 <= iy < ny):
        raise ValueError(f"bin ({ix}, {iy}) is outside the {nx} x {ny} grid")
    x_lo = grid.x0 + ix * grid.dx
    y_lo = grid.y0 + iy * grid.dy
    shots, recs, vecs = [], [], []
    for i in range(geo.ns):
        src, rec = _shot_pairs(geo, i)
        mid = 0.5 * (rec + src)
        hit = np.flatnonzero((mid[:, 0] >= x_lo) & (mid[:, 0] < x_lo + grid.dx)
                             & (mid[:, 1] >= y_lo) & (mid[:, 1] < y_lo + grid.dy))
        if not hit.size:
            continue
        shots.append(np.full(hit.size, i, dtype=np.int64))
        recs.append(hit.astype(np.int64))
        vecs.append(rec[hit] - src)
    if not shots:
        return (np.empty(0, np.int64), np.empty(0, np.int64),
                np.empty((0, 2), np.float64))
    return np.concatenate(shots), np.concatenate(recs), np.concatenate(vecs)


def traces_in_bin(geo, grid: BinGrid, ix: int, iy: int):
    """``(shots, recs, offsets)`` addressing every trace that images one bin.

    This is the CMP gather's index: the pairs are sorted by offset, which is
    the order a CMP gather is displayed and the order moveout expects, so
    reading ``data[shot, rec]`` down the arrays gives the gather directly.
    See ``gathervis.cmp`` for the version that does the reading.
    """
    shots, recs, d = _hits_in_bin(geo, grid, ix, iy)
    offs = np.hypot(d[:, 0], d[:, 1])
    order = np.argsort(offs, kind="stable")
    return shots[order], recs[order], offs[order]


def offsets_azimuths_in_bin(geo, grid: BinGrid, ix: int, iy: int):
    """(offsets, azimuths, shots) of every trace whose midpoint is in one bin.

    This is what a rose / spider diagram for that bin is drawn from: how far
    and in which direction the survey illuminates that subsurface point.
    Azimuth is the survey convention -- degrees clockwise from +y.
    """
    shots, _, d = _hits_in_bin(geo, grid, ix, iy)
    if not shots.size:
        return (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))
    return (np.hypot(d[:, 0], d[:, 1]),
            np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0,
            shots)


def azimuth_sectors(azimuths, nsector: int = 24):
    """Count azimuths into ``nsector`` equal sectors starting at north.

    Returns ``(edges, counts)`` with ``edges`` of length ``nsector + 1`` in
    degrees, so ``counts[i]`` covers ``[edges[i], edges[i + 1])``. A rose
    diagram is this histogram drawn as wedges; the number of non-empty
    sectors is the plain answer to "does this bin see all round?".
    """
    if nsector < 1:
        raise ValueError(f"nsector must be >= 1, got {nsector}")
    edges = np.linspace(0.0, 360.0, nsector + 1)
    a = np.asarray(azimuths, dtype=float)
    if not a.size:
        return edges, np.zeros(nsector, dtype=np.int64)
    idx = np.floor(a % 360.0 / (360.0 / nsector)).astype(np.int64)
    np.clip(idx, 0, nsector - 1, out=idx)
    return edges, np.bincount(idx, minlength=nsector)