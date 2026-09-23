"""Shared state, a staged processing chain, and the views built on them.

This is the layer ``Workspace`` grew into and outgrew. Three pieces:

``Display``
    Everything that decides how data *looks* -- filter, gain, polarity, clip
    percentile, colormap, display mode -- in one observable object. Changing
    one announces which **stage** of the chain it invalidated, and nothing
    above that stage is recomputed.

``Chain``
    ``raw -> filtered -> gained -> clim -> wire``, with every stage cached.
    This is where the measured win is. The chain costs about 47 ms on a
    480x3000 shot, and 36 ms of that is bandpass plus AGC -- so recomputing
    it because somebody moved the clip-percentile slider, which only affects
    the last two stages, wastes 36 ms for nothing. With the cache a clip
    change costs about 2 ms, a gain change skips the filter, and only a new
    shot or a new filter pays in full.

``Session``
    The dataset, one ``Display``, one shared ``Cursor`` (which shot, which
    bin), and a list of views. Views register themselves; the session does
    not know what any of them are. That is the other half of the problem
    ``Workspace`` had -- it knew the class, constructor, tab label, redraw
    method and wiring of every view it could show, so adding one meant
    editing it in four places.

Nothing here imports panel or bokeh: this is the state and the arithmetic,
testable on its own. The views that draw it live in ``viewer``.
"""
from __future__ import annotations

from collections.abc import MutableMapping

import numpy as np

from .process import agc, bandpass, robust_clim, trace_balance

__all__ = ["DATA", "FILTER", "GAIN", "SCALE", "STYLE", "STAGES",
           "Display", "DisplayState", "Chain", "Cursor", "Session", "View"]

# The stages of the chain, in order. A change at stage N invalidates N and
# everything after it, and leaves everything before it alone. They are plain
# ordered integers so "is this change cheaper than that one" is a comparison.
DATA = 0      # the traces themselves changed -- a new shot, a new gather
FILTER = 1    # bandpass corners
GAIN = 2      # agc / trace balance / polarity
SCALE = 3     # clip percentile -> clim
STYLE = 4     # colormap, wiggle vs density: no arithmetic at all
STAGES = (DATA, FILTER, GAIN, SCALE, STYLE)

_STAGE_NAMES = {DATA: "data", FILTER: "filter", GAIN: "gain",
                SCALE: "scale", STYLE: "style"}


class Display:
    """The display chain's settings, as one observable object.

    Every setter announces the stage it dirtied, so a listener knows how
    much to redo. Setting a value to what it already is announces nothing,
    which keeps a slider that fires on every pixel from causing work when it
    lands back where it started.
    """

    def __init__(self, cmap="gray", perc=98.0, symmetric=True):
        self._filter = None            # None or dict(f1=,f2=,f3=,f4=)
        self._gain = None              # None or ("agc", seconds) or ("balance",)
        self._flip = False             # polarity
        self._perc = float(perc)
        self._cmap = str(cmap)
        self._mode = "density"         # or "wiggle"
        self.symmetric = bool(symmetric)
        self._listeners = []           # [(fn, min_stage)]

    # -- observation ------------------------------------------------------
    def on_change(self, fn, stages=STYLE):
        """Call ``fn(stage)`` when something at or before ``stages`` changes.

        The default listens to everything. A view that only draws from the
        quantized image can pass ``stages=SCALE`` and never be woken by a
        colormap change it handles itself.
        """
        self._listeners.append((fn, stages))
        return fn

    def _announce(self, stage):
        for fn, upto in self._listeners:
            if stage <= upto:
                fn(stage)

    # -- settings ---------------------------------------------------------
    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, value):
        if value == self._filter:
            return
        self._filter = value
        self._announce(FILTER)

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        value = tuple(value) if value else None
        if value == self._gain:
            return
        self._gain = value
        self._announce(GAIN)

    @property
    def flip(self):
        return self._flip

    @flip.setter
    def flip(self, value):
        value = bool(value)
        if value == self._flip:
            return
        self._flip = value
        self._announce(GAIN)        # polarity rides with gain: same stage

    @property
    def perc(self):
        return self._perc

    @perc.setter
    def perc(self, value):
        value = float(value)
        if value == self._perc:
            return
        self._perc = value
        self._announce(SCALE)

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, value):
        if value == self._cmap:
            return
        self._cmap = str(value)
        self._announce(STYLE)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value == self._mode:
            return
        self._mode = str(value)
        self._announce(STYLE)

    def __repr__(self):
        bits = [f"cmap={self._cmap!r}", f"perc={self._perc}"]
        if self._filter:
            bits.append(f"filter={self._filter}")
        if self._gain:
            bits.append(f"gain={self._gain}")
        if self._flip:
            bits.append("flipped")
        return f"Display({', '.join(bits)})"


class DisplayState(MutableMapping):
    """A dict interface onto a :class:`Display`, plus the sample and clim.

    The viewer was written around a plain ``state`` dict passed to every
    panel, and a good deal of it -- and of the test suite -- reads and
    writes that dict directly. Rewriting all of it at once to take a
    ``Display`` would mean changing the code and the tests that check it in
    the same move, which is how a refactor turns into a rewrite with nothing
    left holding it honest.

    So the dict stays, and becomes a view. ``state["gain"] = ("agc", 0.5)``
    still works and still reads back; underneath it sets the property on the
    ``Display``, which announces the stage it dirtied, which is what lets
    the caches in :class:`Chain` know what to keep. Callers can be moved off
    it one at a time, and until the last one is gone nothing is broken.
    """

    _PROPS = ("filter", "gain", "flip", "perc", "cmap", "mode", "symmetric")

    def __init__(self, display: Display, sample=None, clim=(-1.0, 1.0)):
        self.display = display
        self._extra = {"sample": sample, "clim": tuple(clim)}

    def __getitem__(self, key):
        if key in self._PROPS:
            return getattr(self.display, key)
        return self._extra[key]

    def __setitem__(self, key, value):
        if key in self._PROPS:
            setattr(self.display, key, value)
        else:
            self._extra[key] = value

    def __delitem__(self, key):
        del self._extra[key]

    def __iter__(self):
        return iter(tuple(self._PROPS) + tuple(self._extra))

    def __len__(self):
        return len(self._PROPS) + len(self._extra)

    def __repr__(self):
        return f"DisplayState({dict(self)})"


class Chain:
    """``raw -> filtered -> gained -> clim``, cached per stage.

    One chain per panel that draws processed traces. Hand it the raw array
    and a key identifying it (a shot index, say); it returns the processed
    array and the colour limits to draw it with, recomputing only the stages
    whose inputs actually changed.

        chain = Chain(display)
        arr, clim = chain.process(raw, dt, key=ishot)

    The cache is one deep on purpose -- it holds the current gather, not a
    history. Stepping through shots is a linear walk, so a history would
    spend memory on gathers nobody is coming back to; the stages that matter
    are re-earned by the 1 ms it takes to read the next shot, not by keeping
    the last twenty.
    """

    def __init__(self, display: Display):
        self.display = display
        self._key = object()          # a key nothing will equal
        self._raw = None
        self._filtered = None
        self._gained = None
        self._clim = None
        self.stats = {"data": 0, "filter": 0, "gain": 0, "scale": 0, "hit": 0}
        display.on_change(self.invalidate, stages=SCALE)

    def invalidate(self, stage=DATA):
        """Drop every cached stage from ``stage`` onwards."""
        if stage <= DATA:
            self._raw = None
        if stage <= FILTER:
            self._filtered = None
        if stage <= GAIN:
            self._gained = None
        if stage <= SCALE:
            self._clim = None

    def process(self, raw, dt, key=None):
        """The processed array and its clim, recomputing only what changed."""
        d = self.display
        if key != self._key or self._raw is None:
            self._key = key
            self._raw = raw
            self._filtered = self._gained = self._clim = None
            self.stats["data"] += 1
        else:
            self.stats["hit"] += 1

        if self._filtered is None:
            self.stats["filter"] += 1
            self._filtered = (bandpass(self._raw, dt, **d.filter)
                              if d.filter else self._raw)

        if self._gained is None:
            self.stats["gain"] += 1
            arr = self._filtered
            if d.gain:
                arr = (agc(arr, dt, window=d.gain[1]) if d.gain[0] == "agc"
                       else trace_balance(arr))
            self._gained = arr

        if self._clim is None:
            self.stats["scale"] += 1
            # With gain on, the raw global clim no longer matches the ~unit
            # output, so it is recomputed from the gained gather. Without
            # gain, the caller's global clim is the right one, set through
            # `set_global_clim`.
            self._clim = (robust_clim(self._gained, d.perc,
                                      symmetric=d.symmetric)
                          if d.gain else self._global_clim)

        # Polarity is applied *after* the clim is measured, not before. On a
        # one-sided volume (a property cube, where clim is not symmetric)
        # measuring the flipped array would hand back limits that do not
        # bracket what is drawn.
        out = -np.asarray(self._gained) if d.flip else self._gained
        return out, self._clim

    _global_clim = (-1.0, 1.0)

    def set_global_clim(self, clim):
        """The dataset-wide clim used when no gain is applied."""
        if clim != self._global_clim:
            self._global_clim = tuple(clim)
            self._clim = None

    def __repr__(self):
        n = self.stats
        return (f"Chain(data={n['data']}, filter={n['filter']}, "
                f"gain={n['gain']}, scale={n['scale']}, reuse={n['hit']})")


class Cursor:
    """What the views agree they are looking at: a shot, a bin.

    Selection is shared rather than owned by whichever view happens to offer
    the control, so tapping a source on the map and dragging the shot slider
    are two ways of saying the same thing and every view hears both.
    """

    def __init__(self, shot=0, bin=None):
        self._shot = int(shot)
        self._bin = bin
        self._listeners = []

    def on_change(self, fn):
        self._listeners.append(fn)
        return fn

    def _announce(self, what):
        for fn in self._listeners:
            fn(what)

    @property
    def shot(self):
        return self._shot

    @shot.setter
    def shot(self, i):
        i = int(i)
        if i == self._shot:
            return
        self._shot = i
        self._announce("shot")

    @property
    def bin(self):
        return self._bin

    @bin.setter
    def bin(self, ixy):
        ixy = None if ixy is None else (int(ixy[0]), int(ixy[1]))
        if ixy == self._bin:
            return
        self._bin = ixy
        self._announce("bin")

    def __repr__(self):
        return f"Cursor(shot={self._shot}, bin={self._bin})"


class View:
    """What a session needs from anything that occupies a tab.

    Subclasses set ``name`` and implement ``panel()``. ``attach`` is where a
    view subscribes to the session it has been added to -- the session never
    reaches into the view, so adding a new kind of view touches no existing
    code.
    """

    name = "view"

    def attach(self, session: "Session"):
        """Called once when added. Subscribe to display/cursor here."""
        self.session = session

    def panel(self):
        raise NotImplementedError

    def refresh(self, stage=DATA):
        """Redraw for a change at ``stage``. Default: nothing to do."""


class Session:
    """A dataset, one display chain, one cursor, and the views on them.

        s = Session(gathers)
        s.add(GatherView())
        s.add(GeometryView())

    The session does not know what a ``GatherView`` is; it holds the shared
    state they all read and the list of the ones that were added. That is
    the whole point -- views are a list, not a branch.
    """

    def __init__(self, data, cmap="gray", perc=98.0):
        self.data = data
        symmetric = getattr(data, "axes", ("time",))[-1] != "depth"
        self.display = Display(cmap=cmap, perc=perc, symmetric=symmetric)
        self.cursor = Cursor()
        self.views = []
        self._focus = []           # hooks that bring a view to the front
        self.sample = data.sample() if hasattr(data, "sample") else None
        clim = (robust_clim(self.sample, perc, symmetric=symmetric)
                if self.sample is not None else (-1.0, 1.0))
        self.state = DisplayState(self.display, self.sample, clim)
        self._chains = []
        self.display.on_change(self._on_display, stages=SCALE)

    @property
    def clim(self):
        """The dataset-wide colour limits at the current clip percentile."""
        return self.state["clim"]

    def _on_display(self, stage):
        if stage <= SCALE and self.sample is not None:
            self.state["clim"] = robust_clim(self.sample, self.display.perc,
                                             symmetric=self.display.symmetric)
            for chain in self._chains:
                chain.set_global_clim(self.state["clim"])

    def chain(self) -> Chain:
        """A new cached chain wired to this session's display."""
        c = Chain(self.display)
        c.set_global_clim(self.clim)
        self._chains.append(c)
        return c

    def add(self, view: View) -> View:
        """Register a view. Returns it, so calls can be chained or kept."""
        view.attach(self)
        self.views.append(view)
        return view

    def view(self, name):
        """The first view with this name, or None."""
        return next((v for v in self.views if v.name == name), None)

    def on_focus(self, fn):
        """Register something that can bring a view to the front by name."""
        self._focus.append(fn)
        return fn

    def focus(self, name):
        """Ask for the named view to be shown.

        A view that wants another one on screen -- the layout map, after a
        tap selects a shot -- says so by name rather than reaching for a tab
        index it would have to know. Whoever is presenting the views decides
        what showing one means, and if nobody is, nothing happens.
        """
        for fn in self._focus:
            fn(name)

    def __repr__(self):
        return (f"Session({getattr(self.data, 'name', None) or 'dataset'}, "
                f"views={[v.name for v in self.views]})")
