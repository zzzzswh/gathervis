"""Top and bottom mute lines: draw them once, carry them across gathers.

A mute is a traveltime line and a side: everything before the **top** line
is zeroed, everything after the **bottom** line is zeroed. Between them the
data is kept. That is the whole idea; the work is in where the line lives
and how it survives being carried from one gather to the next.

**The line is a function of offset, not of trace number.** This is the one
design decision everything else follows from. A mute is physically an
offset-dependent thing -- the direct wave arrives at ``offset / v``, the
stretch-damaged zone widens with offset -- so a line stored against offset
means the same thing on the next gather even though that gather has a
different number of traces in a different order. Stored against trace
number it would mean nothing at all as soon as anything changed. Without
geometry there are no offsets, so the line falls back to trace index and
stops travelling between gathers of different length; that is a real
limitation and :attr:`Mutes.domain` says which case you are in.

**One default, with per-gather overrides.** Most gathers want the same mute,
so the default is what a new gather gets. Some gathers do not -- a bad
shot, a patch of ground roll that arrives early -- so any gather can carry
its own line that shadows the default. :meth:`Mutes.line` resolves the two;
:meth:`Mutes.is_override` says which one you are looking at. Nothing is
interpolated between gathers: a gather either has its own line or uses the
default, because a viewer should not quietly invent a mute for a gather
nobody looked at.

Lines are polylines: ``(n, 2)`` arrays of ``(x, time)`` nodes, linear
between nodes and **held flat beyond the ends** rather than extrapolated.
Holding flat is the safe choice -- extrapolating a two-node line to zero
offset will happily produce a negative time.
"""
from __future__ import annotations

import json

import numpy as np

__all__ = ["TOP", "BOTTOM", "KINDS", "Mutes", "apply", "evaluate",
           "linear_line", "mute_shot"]

TOP = "top"
BOTTOM = "bottom"
KINDS = (TOP, BOTTOM)

DEFAULT_KEY = None          # the line every gather gets unless it has its own


def _nodes(line, name="line") -> np.ndarray:
    """Validate and normalise a polyline to a sorted ``(n, 2)`` float array."""
    a = np.asarray(line, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"{name} must be (n, 2) of (x, time), got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a[np.argsort(a[:, 0], kind="stable")]


def evaluate(line, x) -> np.ndarray:
    """The line's time at each ``x``; flat beyond its ends.

    An empty line evaluates to NaN, which :func:`apply` reads as "no mute on
    this side" -- so a gather with only a top line is not accidentally
    bottom-muted at time zero.
    """
    nodes = _nodes(line)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if nodes.shape[0] == 0:
        return np.full(x.shape, np.nan)
    return np.interp(x, nodes[:, 0], nodes[:, 1],
                     left=nodes[0, 1], right=nodes[-1, 1])


def linear_line(v, x_max, x_min=0.0, pad=0.0) -> np.ndarray:
    """A two-node line at ``x / v + pad`` -- a straight moveout cut.

    The usual starting point for a top mute: the direct wave and the
    first-arrival refractions are straight lines through the origin, so one
    velocity and one delay describe the cut you actually want, and you drag
    the nodes from there.
    """
    if float(v) <= 0:
        raise ValueError(f"mute velocity must be positive, got {v}")
    return np.array([[float(x_min), float(x_min) / float(v) + float(pad)],
                     [float(x_max), float(x_max) / float(v) + float(pad)]])


def apply(gather, x, dt, top=None, bottom=None, t0=0.0, taper=0.04):
    """Mute a gather with a top line, a bottom line, or both.

    Parameters
    ----------
    gather : (ntrace, nt) array
        Time is the last axis, as everywhere in gathervis.
    x : (ntrace,) array
        The coordinate the lines are defined against -- offsets, or trace
        indices when there is no geometry. One value per trace.
    dt : float
        Sample interval, seconds.
    top, bottom : (n, 2) arrays, optional
        Polylines of ``(x, time)``. ``None`` or empty means no mute on that
        side.
    t0 : float, optional
        Time of the first sample, seconds.
    taper : float, optional
        Length of the cosine ramp, seconds, applied on the live side of each
        cut. 0 for a hard edge.

        Not a nicety: a hard mute leaves a step running along the cut, and a
        step is broadband energy on a coherent trajectory -- it shows up in
        spectra and stacks as an event that was never in the ground.

    Returns
    -------
    (ntrace, nt) float32
    """
    a = np.asarray(gather, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"gather must be 2-D (ntrace, nt), got shape {a.shape}")
    ntr, nt = a.shape
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != ntr:
        raise ValueError(f"x has {x.size} entries but gather has {ntr} traces")
    if float(dt) <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    tau = float(t0) + np.arange(nt, dtype=np.float64) * float(dt)
    weight = np.ones((ntr, nt), dtype=np.float32)
    ramp = max(float(taper), 0.0)

    t_top = evaluate(top, x) if top is not None else np.full(ntr, np.nan)
    live = ~np.isnan(t_top)
    if live.any():
        if ramp > 0:                       # 0 at the cut, 1 one taper later
            w = np.clip((tau[None, :] - t_top[live, None]) / ramp, 0.0, 1.0)
            w = 0.5 - 0.5 * np.cos(np.pi * w)
        else:
            w = (tau[None, :] >= t_top[live, None]).astype(np.float64)
        weight[live] *= w.astype(np.float32)

    t_bot = evaluate(bottom, x) if bottom is not None else np.full(ntr, np.nan)
    live = ~np.isnan(t_bot)
    if live.any():
        if ramp > 0:                       # 1 one taper before the cut, 0 at it
            w = np.clip((t_bot[live, None] - tau[None, :]) / ramp, 0.0, 1.0)
            w = 0.5 - 0.5 * np.cos(np.pi * w)
        else:
            w = (tau[None, :] <= t_bot[live, None]).astype(np.float64)
        weight[live] *= w.astype(np.float32)

    return (a * weight).astype(np.float32)


class Mutes:
    """A default top/bottom mute plus the gathers that differ from it.

    ``key`` is whatever identifies a gather -- a shot index, a CDP number.
    ``None`` is the default that every other gather inherits.

        m = Mutes(domain="offset")
        m.set(None, TOP, [[0, 0.05], [3000, 1.2]])   # every gather
        m.set(37, TOP, [[0, 0.09], [3000, 1.3]])     # except shot 37
        m.line(12, TOP)                              # -> the default
        m.line(37, TOP)                              # -> shot 37's own
    """

    def __init__(self, domain="offset"):
        if domain not in ("offset", "trace"):
            raise ValueError(f"domain must be 'offset' or 'trace', got {domain!r}")
        self.domain = domain
        self._lines = {}                  # key -> {kind: (n, 2) array}

    # -- reading ---------------------------------------------------------
    def line(self, key, kind) -> np.ndarray:
        """The line in force for ``key``: its own if it has one, else the
        default, else empty."""
        self._check_kind(kind)
        own = self._lines.get(key, {}).get(kind)
        if own is not None:
            return own
        return self._lines.get(DEFAULT_KEY, {}).get(kind, np.zeros((0, 2)))

    def is_override(self, key, kind=None) -> bool:
        """Whether ``key`` has its own line (for ``kind``, or for either)."""
        own = self._lines.get(key, {})
        if key is DEFAULT_KEY or not own:
            return False
        return bool(own) if kind is None else kind in own

    def overrides(self) -> list:
        """The keys that differ from the default, in order."""
        return sorted(k for k, v in self._lines.items()
                      if k is not DEFAULT_KEY and v)

    def __len__(self):
        return len(self.overrides())

    def __repr__(self):
        have = [k for k in KINDS
                if self._lines.get(DEFAULT_KEY, {}).get(k) is not None]
        return (f"Mutes(domain={self.domain!r}, default={have or 'none'}, "
                f"{len(self)} overrides)")

    # -- writing ---------------------------------------------------------
    def set(self, key, kind, line):
        """Set the line for ``key`` (``None`` = the default for all gathers)."""
        self._check_kind(kind)
        nodes = _nodes(line, f"{kind} mute")
        if nodes.shape[0] == 0:
            self.clear(key, kind)
            return
        self._lines.setdefault(key, {})[kind] = nodes

    def clear(self, key, kind=None):
        """Drop ``key``'s own line, so it goes back to inheriting the default.

        On the default itself this removes the mute everywhere.
        """
        if key not in self._lines:
            return
        if kind is None:
            del self._lines[key]
            return
        self._check_kind(kind)
        self._lines[key].pop(kind, None)
        if not self._lines[key]:
            del self._lines[key]

    def promote(self, key):
        """Make ``key``'s lines the default, and drop it as an override.

        What you do when you have tuned a mute on one gather and decide it is
        the one you want everywhere.
        """
        own = self._lines.get(key)
        if not own:
            return
        self._lines[DEFAULT_KEY] = dict(own)
        if key is not DEFAULT_KEY:
            del self._lines[key]

    @staticmethod
    def _check_kind(kind):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    # -- applying --------------------------------------------------------
    def apply(self, gather, x, dt, key=None, t0=0.0, taper=0.04):
        """Mute ``gather`` with whichever lines are in force for ``key``."""
        return apply(gather, x, dt, top=self.line(key, TOP),
                     bottom=self.line(key, BOTTOM), t0=t0, taper=taper)

    # -- files -----------------------------------------------------------
    def to_json(self, indent=1) -> str:
        """Serialise to JSON: ``domain``, ``default``, and ``overrides``.

        Keys are written as strings because JSON has no others; integer-like
        keys are read back as integers, which is what a shot index is.
        """
        def block(d):
            return {k: [[round(float(x), 6), round(float(t), 6)] for x, t in v]
                    for k, v in d.items()}
        payload = {
            "gathervis_mutes": 1,
            "domain": self.domain,
            "default": block(self._lines.get(DEFAULT_KEY, {})),
            "overrides": {str(k): block(v) for k, v in self._lines.items()
                          if k is not DEFAULT_KEY and v},
        }
        return json.dumps(payload, indent=indent)

    @classmethod
    def from_json(cls, text) -> "Mutes":
        payload = json.loads(text)
        if "gathervis_mutes" not in payload:
            raise ValueError("not a gathervis mute file")
        out = cls(domain=payload.get("domain", "offset"))
        for kind, nodes in payload.get("default", {}).items():
            out.set(DEFAULT_KEY, kind, nodes)
        for key, block in payload.get("overrides", {}).items():
            key = int(key) if str(key).lstrip("-").isdigit() else key
            for kind, nodes in block.items():
                out.set(key, kind, nodes)
        return out


def mute_shot(g, ishot: int, mutes, taper: float = 0.04) -> np.ndarray:
    """Apply saved mute lines to one shot of a dataset.

    The other end of the tool: draw the lines in the viewer, export the
    JSON, then use it from a script without opening anything.

        from gathervis.mute import Mutes, mute_shot
        m = Mutes.from_json(open("mutes.json").read())
        clean = mute_shot(ds, 7, m)

    The traces come back in the dataset's own order, not in whatever order
    the panel happened to be sorted in when the lines were drawn -- the
    lines are stored against offset precisely so that this does not matter.

    Parameters
    ----------
    g : Gathers
    ishot : int
        Which shot. It is also the key looked up in ``mutes``, so a gather
        that was given its own line in the viewer gets it here too.
    mutes : Mutes
    taper : float, optional
        Cosine ramp length, seconds.

    Returns
    -------
    (ntrace, nt) float32
    """
    from . import sortkeys as _sort

    data = np.asarray(g.shot(int(ishot)))
    if data.ndim > 2:                              # a 3-D patch: flatten it
        data = data.reshape(-1, data.shape[-1])
    if mutes.domain == "offset":
        if g.geometry is None:
            raise ValueError("these mutes are stored against offset, but this "
                             "dataset has no geometry to compute offsets from")
        x = _sort.offsets(g, int(ishot))
    else:
        x = np.arange(data.shape[0], dtype=float)
    return mutes.apply(data, x, g.dt, key=int(ishot), t0=g.t0, taper=taper)
