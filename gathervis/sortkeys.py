"""Trace orderings for a single shot record.

A 3-D shot is a patch of several receiver lines. Processing shops do not look
at it as a cuboid -- every trace of the shot goes onto one ``(trace, time)``
panel, and what you change is the *order* the traces are laid out in:

``as recorded``
    Acquisition order ``(line, station)``: the receiver lines end to end, so
    the record reads as N nested hyperbolas whose apices step with cross-line
    distance. This is the display that geometry errors, dead channels,
    reversed lines and polarity flips jump out of.
``receiver line``
    One line at a time, i.e. a plain 2-D shot record -- for detailed work on
    a line that looked wrong in the full patch.
``offset``
    Every trace by source-receiver distance, collapsing the whole patch onto
    one hyperbola: moveout, mute design, velocity.
``azimuth``
    By source-to-receiver azimuth, for azimuthal amplitude behaviour and for
    seeing how the patch covers azimuth at all.

Only ``as recorded`` works on any shot: ``offset`` and ``azimuth`` need
source and receiver coordinates, and ``receiver line`` needs a 4-D
``(shot, recy, recx, time)`` array, where the line structure is explicit.

Ordering is kept separate from processing on purpose: ``arrange`` returns the
raw traces in the requested order (a lazy view where it can be one), and the
filter/gain chain runs on the result exactly as it does for a 2-D gather.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

__all__ = ["SORTS", "AS_RECORDED", "RECEIVER_LINE", "OFFSET", "AZIMUTH",
           "ShotPanel", "available", "nlines", "line_length", "offsets",
           "azimuths", "arrange"]

AS_RECORDED = "as recorded"
RECEIVER_LINE = "receiver line"
OFFSET = "offset"
AZIMUTH = "azimuth"
SORTS = (AS_RECORDED, RECEIVER_LINE, OFFSET, AZIMUTH)


class ShotPanel(NamedTuple):
    """One shot laid out as a 2-D panel, plus what it takes to read the axis.

    ``order`` is the flat trace index of every displayed column, in the
    dataset's own ``(line, station)`` numbering. It is what makes a sort
    reversible: the pick tool stores picks against traces and uses ``order``
    to find the column each one currently sits in.
    """

    data: np.ndarray                 # (ntrace, nt)
    order: np.ndarray                # (ntrace,) original flat trace index
    xlabel: str
    key: Optional[np.ndarray]        # sort value per column (m / degrees)
    bounds: np.ndarray               # x of the receiver-line separators


def nlines(g) -> int:
    """Receiver lines per shot: >1 only when the line structure is explicit."""
    return int(g.data.shape[1]) if g.data.ndim == 4 else 1


def line_length(g) -> int:
    """Traces per receiver line."""
    return int(g.data.shape[2]) if g.data.ndim == 4 else int(g.data.shape[1])


def available(g) -> tuple:
    """The sorts this dataset can actually offer, in menu order."""
    if "shot" not in g.axes:
        return ()
    out = [AS_RECORDED]
    if nlines(g) > 1:
        out.append(RECEIVER_LINE)
    if g.geometry is not None:
        out += [OFFSET, AZIMUTH]
    return tuple(out)


def _vectors(g, ishot: int) -> np.ndarray:
    """(ntrace, 2) source-to-receiver vectors in map coordinates."""
    geo = g.geometry
    if geo is None:
        raise ValueError("offset / azimuth sorting needs geometry "
                         "(src= and rec= coordinates)")
    return geo.rec_for(ishot)[:, :2] - geo.src[ishot][:2]


def offsets(g, ishot: int) -> np.ndarray:
    """Horizontal source-receiver distance per trace, in map units."""
    d = _vectors(g, ishot)
    return np.hypot(d[:, 0], d[:, 1])


def azimuths(g, ishot: int) -> np.ndarray:
    """Source-to-receiver azimuth per trace, degrees clockwise from +y.

    The survey convention (north = 0, east = 90), not the mathematical one,
    so the numbers match what an observer's report would say.
    """
    d = _vectors(g, ishot)
    return np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0


def _flat(g, ishot: int) -> np.ndarray:
    """The shot as (ntrace, nt). A view on the memmap wherever possible."""
    a = g.shot(ishot)
    return a.reshape(-1, a.shape[-1]) if a.ndim > 2 else a


def _span(key: np.ndarray, unit: str) -> str:
    return f"{key.min():.0f}-{key.max():.0f} {unit}"


def arrange(g, ishot: int, sort: str = AS_RECORDED, line: int = 0) -> ShotPanel:
    """Lay shot ``ishot`` out as a 2-D panel under the requested ordering.

    ``as recorded`` and ``receiver line`` stay lazy (a reshape / a slice of
    the memmap); ``offset`` and ``azimuth`` have to materialise the shot,
    which is one shot's worth of bytes, not the file's.
    """
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {SORTS}, got {sort!r}")
    flat = _flat(g, ishot)
    ntr = flat.shape[0]
    nl, nper = nlines(g), line_length(g)
    none = np.empty(0)

    if sort == RECEIVER_LINE:
        if nl == 1:
            raise ValueError("this dataset has one receiver line per shot; "
                             "'receiver line' needs 4-D (shot, recy, recx, time)")
        line = int(np.clip(line, 0, nl - 1))
        idx = np.arange(line * nper, (line + 1) * nper)
        return ShotPanel(g.shot(ishot)[line], idx,          # lazy: one line
                         f"station  (receiver line {line} of {nl})",
                         None, none)

    if sort == AS_RECORDED:
        order = np.arange(ntr)
        bounds = (np.arange(1, nl) * nper - 0.5) if nl > 1 else none
        xlabel = ("trace  (line, station)" if nl > 1 else "trace")
        return ShotPanel(flat, order, xlabel, None, bounds)

    key = offsets(g, ishot) if sort == OFFSET else azimuths(g, ishot)
    if key.shape[0] != ntr:                 # geometry/data mismatch, defensively
        raise ValueError(f"geometry has {key.shape[0]} receivers for shot "
                         f"{ishot}, data has {ntr} traces")
    order = np.argsort(key, kind="stable")
    unit = "m" if sort == OFFSET else "deg"
    xlabel = f"trace  (sorted by {sort}, {_span(key, unit)})"
    return ShotPanel(np.asarray(flat)[order], order, xlabel, key[order], none)