"""CMP gathers: the traces that image one bin, read off the data.

``gathervis.survey`` says *which* traces image a bin -- it works on
coordinates alone and never touches the data. This module is the other half:
given that index, read those traces out of the array.

The split matters on big files. A fold map is geometry, so it costs the same
on a 100 GB survey as on a toy one; pulling a CMP gather is the first
operation that has to go to disk, and it reads one bin's worth of traces --
typically a few dozen -- rather than anything resembling the whole file. The
reads are grouped by shot so a memmap sees one fancy-index per shot instead
of one seek per trace.

A CMP gather is the domain velocity analysis happens in: the same subsurface
point seen at a spread of offsets, which is what makes moveout measurable.
Hence :func:`gather_in_bin` sorting by offset -- that is both how a CMP
gather is displayed and what ``gathervis.velocity`` expects.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import survey as _survey

__all__ = ["CmpGather", "gather_in_bin", "best_bin"]


class CmpGather(NamedTuple):
    """One bin's traces, offset-sorted, plus what it takes to label them."""

    data: np.ndarray          # (ntrace, nt)
    offsets: np.ndarray       # (ntrace,) source-receiver distance, map units
    shots: np.ndarray         # (ntrace,) which shot each trace came from
    recs: np.ndarray          # (ntrace,) receiver index within that shot
    ix: int
    iy: int
    x: float                  # bin centre
    y: float

    @property
    def fold(self) -> int:
        return int(self.data.shape[0])

    def __repr__(self):
        if not self.fold:
            return f"CmpGather(bin=({self.ix}, {self.iy}), empty)"
        return (f"CmpGather(bin=({self.ix}, {self.iy}) at "
                f"({self.x:.1f}, {self.y:.1f}), fold={self.fold}, "
                f"offsets {self.offsets.min():.0f}-{self.offsets.max():.0f}, "
                f"nt={self.data.shape[1]})")


def gather_in_bin(g, grid, ix: int, iy: int) -> CmpGather:
    """Read the CMP gather imaging bin ``(ix, iy)``.

    Parameters
    ----------
    g : Gathers
        Must have a shot axis and geometry.
    grid : survey.BinGrid
        The binning to use, as returned by :func:`gathervis.survey.fold`.
    ix, iy : int
        Bin indices, as :meth:`BinGrid.bin_of` returns for a map position.

    Returns
    -------
    CmpGather
        ``data`` is ``(ntrace, nt)`` ordered by increasing offset. An empty
        bin gives a gather of fold 0 rather than an error -- tapping a hole
        in the fold map is a normal thing to do.
    """
    # Gathers refuses geometry without a shot axis, so this one check covers
    # both: if there is geometry, there are shots to index into.
    if g.geometry is None:
        raise ValueError("CMP gathers need geometry (src= and rec= coordinates)")

    shots, recs, offsets = _survey.traces_in_bin(g.geometry, grid, ix, iy)
    x = grid.x0 + (ix + 0.5) * grid.dx
    y = grid.y0 + (iy + 0.5) * grid.dy
    if not shots.size:
        return CmpGather(np.zeros((0, g.nt), np.float32), offsets.astype(np.float32),
                         shots, recs, int(ix), int(iy), float(x), float(y))

    # One fancy-index per shot, not one seek per trace: on a memmap the
    # difference between these two is the whole reason this is usable.
    out = np.empty((shots.size, g.nt), dtype=np.float32)
    for shot in np.unique(shots):
        take = np.flatnonzero(shots == shot)
        block = g.shot(int(shot))
        if block.ndim > 2:                       # 3-D shot: flatten the patch
            block = block.reshape(-1, block.shape[-1])
        out[take] = np.asarray(block[recs[take]], dtype=np.float32)

    return CmpGather(out, offsets.astype(np.float32), shots, recs,
                     int(ix), int(iy), float(x), float(y))


def best_bin(grid) -> tuple:
    """``(ix, iy)`` of the highest-fold bin -- the one worth opening first.

    A velocity panel is only as good as the offset range behind it, so the
    sensible default bin to land on is the best-illuminated one rather than
    the middle of the map, which on a tapered survey is often near the edge
    of the live area.
    """
    if grid.counts.max() <= 0:
        raise ValueError("no live bins in this grid")
    iy, ix = np.unravel_index(int(np.argmax(grid.counts)), grid.counts.shape)
    return int(ix), int(iy)
