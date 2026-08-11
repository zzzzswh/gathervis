"""Data model: pre-stack gathers with semantic axes + optional acquisition geometry.

Design rules (keep it simple):
  * ``data`` is any numpy-like array (ndarray / memmap / zarr array). Never copied.
  * The last axis is always time.
  * When a ``shot`` axis exists it is always axis 0.
  * Default semantics: 2D -> (trace, time), 3D -> (shot, rec, time),
    4D -> (shot, recy, recx, time).
"""
from __future__ import annotations

import numpy as np

__all__ = ["Geometry", "Gathers", "from_array", "from_file"]

DEFAULT_AXES = {
    2: ("trace", "time"),
    3: ("shot", "rec", "time"),
    4: ("shot", "recy", "recx", "time"),
}
KNOWN_AXES = {"shot", "rec", "recx", "recy", "trace", "time", "cmp", "offset"}


def _xyz(a, name):
    """Coerce coordinates to float64 (..., 3), padding z=0 if only (x, y) given."""
    a = np.asarray(a, dtype=np.float64)
    if a.shape[-1] == 2:
        a = np.concatenate([a, np.zeros(a.shape[:-1] + (1,))], axis=-1)
    if a.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 2) or (..., 3), got {a.shape}")
    return a


class Geometry:
    """Acquisition geometry: source and receiver coordinates.

    Parameters
    ----------
    src : (ns, 2|3) array
        One XY(Z) position per shot.
    rec : (nr, 2|3) or (ns, nr, 2|3) array
        Receiver positions, either shared by all shots (fixed spread)
        or given per shot.
    """

    def __init__(self, src, rec):
        self.src = _xyz(src, "src")
        if self.src.ndim != 2:
            raise ValueError(f"src must be 2-D (ns, 2|3), got shape {self.src.shape}")
        rec = _xyz(rec, "rec")
        if rec.ndim == 2:
            self.rec, self.shared = rec, True
        elif rec.ndim == 3:
            if rec.shape[0] != self.ns:
                raise ValueError(
                    f"per-shot rec has {rec.shape[0]} shots but src has {self.ns}")
            self.rec, self.shared = rec, False
        else:
            raise ValueError(f"rec must be (nr, 2|3) or (ns, nr, 2|3), got {rec.shape}")

    @property
    def ns(self) -> int:
        return self.src.shape[0]

    @property
    def nr(self) -> int:
        return self.rec.shape[0] if self.shared else self.rec.shape[1]

    def rec_for(self, ishot: int) -> np.ndarray:
        """(nr, 3) receiver positions active for shot ``ishot``."""
        return self.rec if self.shared else self.rec[ishot]

    def __repr__(self):
        kind = "shared spread" if self.shared else "per-shot"
        return f"Geometry(ns={self.ns}, nr={self.nr}, {kind})"


class Gathers:
    """A pre-stack dataset: data + semantic axes (+ optional geometry).

    ``data`` is kept by reference (memmap-friendly); slicing a shot only
    touches the bytes of that shot.
    """

    def __init__(self, data, axes=None, dt=1.0, t0=0.0, geometry=None):
        if data.ndim not in DEFAULT_AXES and data.ndim != 2:
            raise ValueError(f"expected 2-D/3-D/4-D data, got {data.ndim}-D")
        axes = tuple(axes) if axes is not None else DEFAULT_AXES[data.ndim]
        if len(axes) != data.ndim:
            raise ValueError(f"axes {axes} does not match data ndim {data.ndim}")
        unknown = set(axes) - KNOWN_AXES
        if unknown:
            raise ValueError(f"unknown axis names {unknown}; known: {sorted(KNOWN_AXES)}")
        if axes[-1] != "time":
            raise ValueError(f"last axis must be 'time', got {axes}")
        if "shot" in axes and axes[0] != "shot":
            raise ValueError("'shot' axis must be axis 0")

        self.data = data
        self.axes = axes
        self.dt = float(dt)
        self.t0 = float(t0)
        self.geometry = geometry

        if geometry is not None:
            if "shot" not in axes:
                raise ValueError("geometry given but data has no 'shot' axis")
            if geometry.ns != data.shape[0]:
                raise ValueError(
                    f"geometry has {geometry.ns} shots, data has {data.shape[0]}")
            ntr = int(np.prod(data.shape[1:-1]))
            if geometry.nr != ntr:
                raise ValueError(
                    f"geometry has {geometry.nr} receivers, data has {ntr} traces/shot")

    # -- basic properties -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def nt(self) -> int:
        return self.data.shape[-1]

    @property
    def nshot(self) -> int:
        if "shot" not in self.axes:
            raise ValueError("dataset has no 'shot' axis")
        return self.data.shape[0]

    @property
    def times(self) -> np.ndarray:
        return self.t0 + self.dt * np.arange(self.nt)

    # -- access ------------------------------------------------------------
    def shot(self, i: int):
        """Data of shot ``i``: (rec, time) for a 2D line, (recy, recx, time) for 3D."""
        return self.data[int(i)]

    def sample(self, max_samples: int = 200_000, chunks: int = 32) -> np.ndarray:
        """A flat subsample for global amplitude statistics.

        Reads a few evenly spaced *contiguous* blocks rather than a fine
        stride: a stride touches every 4 KB disk page (i.e. reads the whole
        file), while contiguous blocks read only ~``max_samples`` values no
        matter how large the memmap is.
        """
        flat = self.data.reshape(-1)
        n = flat.shape[0]
        if n <= max_samples:
            return np.asarray(flat)
        per = max_samples // chunks
        starts = np.linspace(0, n - per, chunks).astype(np.int64)
        return np.concatenate([np.asarray(flat[s:s + per]) for s in starts])

    def __repr__(self):
        geo = f", geometry={self.geometry!r}" if self.geometry is not None else ""
        return (f"Gathers(shape={tuple(self.shape)}, axes={self.axes}, "
                f"dt={self.dt}{geo})")


def from_array(data, src=None, rec=None, axes=None, dt=1.0, t0=0.0) -> Gathers:
    """Wrap an in-memory / memmapped array (optionally with geometry)."""
    geometry = None
    if src is not None or rec is not None:
        if src is None or rec is None:
            raise ValueError("both src and rec are required for geometry")
        geometry = Geometry(src, rec)
    return Gathers(data, axes=axes, dt=dt, t0=t0, geometry=geometry)


def from_file(path, shape, dtype="float32", axes=None, dt=1.0, t0=0.0,
              src=None, rec=None, offset=0) -> Gathers:
    """Lazily open a raw binary file as a read-only memmap."""
    data = np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape), offset=offset)
    return from_array(data, src=src, rec=rec, axes=axes, dt=dt, t0=t0)
