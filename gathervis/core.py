"""Data model: pre-stack gathers with semantic axes + optional acquisition geometry.

Design rules (keep it simple):
  * ``data`` is any numpy-like array (ndarray / memmap / zarr array). Never copied.
  * The last axis is always the vertical one: time (data) or depth
    (property volumes such as velocity models).
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
KNOWN_AXES = {"shot", "rec", "recx", "recy", "trace", "time", "cmp", "offset",
              "x", "y", "depth"}   # x/y/depth: property volumes (e.g. velocity)
VERTICAL_AXES = ("time", "depth")


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

    def unique_receivers(self) -> np.ndarray:
        """Every receiver position once, ``(n, 3)``.

        A per-shot geometry repeats a spread once per shot that uses it --
        15 2-D lines of 1109 receivers over 8385 shots are 9.3 M rows for
        16635 positions, and drawing all of them would stall the browser.
        Consecutive shots usually share their spread, so a shot whose spread
        equals the last one kept is skipped first (cheap), and exact
        duplicates among the rest go after.
        """
        if self.shared:
            return self.rec
        kept, last = [], None
        for r in self.rec:
            if last is None or not np.array_equal(r, last):
                kept.append(r)
                last = r
        return np.unique(np.concatenate(kept), axis=0)

    def __repr__(self):
        kind = "shared spread" if self.shared else "per-shot"
        return f"Geometry(ns={self.ns}, nr={self.nr}, {kind})"


class Gathers:
    """A pre-stack dataset: data + semantic axes (+ optional geometry).

    ``data`` is kept by reference (memmap-friendly); slicing a shot only
    touches the bytes of that shot.
    """

    def __init__(self, data, axes=None, dt=1.0, t0=0.0, geometry=None, name=None):
        if data.ndim not in DEFAULT_AXES and data.ndim != 2:
            raise ValueError(f"expected 2-D/3-D/4-D data, got {data.ndim}-D")
        axes = tuple(axes) if axes is not None else DEFAULT_AXES[data.ndim]
        if len(axes) != data.ndim:
            raise ValueError(f"axes {axes} does not match data ndim {data.ndim}")
        unknown = set(axes) - KNOWN_AXES
        if unknown:
            raise ValueError(f"unknown axis names {unknown}; known: {sorted(KNOWN_AXES)}")
        if axes[-1] not in VERTICAL_AXES:
            raise ValueError(
                f"last axis must be one of {VERTICAL_AXES}, got {axes}")
        if "shot" in axes and axes[0] != "shot":
            raise ValueError("'shot' axis must be axis 0")

        self.data = data
        self.axes = axes
        self.dt = float(dt)
        self.t0 = float(t0)
        self.geometry = geometry
        self.name = name

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

        ``data`` need not be a numpy array. Anything with ``shape``,
        ``ndim``, ``dtype`` and indexing along axis 0 works -- a lazy
        concatenation of several memmaps, say -- and is sampled through that
        indexing, still in contiguous blocks. So is any array that is not
        C-contiguous (a transposed or strided memmap view, a broadcast
        array): flattening one of those copies all of it.
        """
        flags = getattr(self.data, "flags", None)
        if (not hasattr(self.data, "reshape") or flags is None
                or not flags["C_CONTIGUOUS"]):
            return self._sample_indexed(max_samples, chunks)
        flat = self.data.reshape(-1)
        n = flat.shape[0]
        if n <= max_samples:
            return np.asarray(flat)
        per = max_samples // chunks
        starts = np.linspace(0, n - per, chunks).astype(np.int64)
        return np.concatenate([np.asarray(flat[s:s + per]) for s in starts])

    def _sample_indexed(self, max_samples, chunks):
        """``sample`` for an array that only supports indexing: a block of
        consecutive traces from each of up to ``chunks`` evenly spaced
        gathers, read as ``data[i, r0:r1]``."""
        shape = tuple(int(n) for n in self.data.shape)
        if len(shape) < 3:                      # (trace, time): one block
            per_row = int(np.prod(shape[1:]))
            n = max(1, min(shape[0], -(-max_samples // max(per_row, 1))))
            r0 = (shape[0] - n) // 2
            return np.asarray(self.data[r0:r0 + n], dtype=np.float64).reshape(-1)
        picks = np.unique(np.linspace(0, shape[0] - 1,
                                      min(chunks, shape[0])).astype(np.int64))
        per = max(1, max_samples // picks.size)
        per_row = int(np.prod(shape[2:]))       # one trace (or one line)
        n = max(1, min(shape[1], -(-per // max(per_row, 1))))
        r0 = (shape[1] - n) // 2
        return np.concatenate([
            np.asarray(self.data[int(i), r0:r0 + n]).reshape(-1)
            for i in picks])

    def __repr__(self):
        geo = f", geometry={self.geometry!r}" if self.geometry is not None else ""
        return (f"Gathers(shape={tuple(self.shape)}, axes={self.axes}, "
                f"dt={self.dt}{geo})")


def from_array(data, src=None, rec=None, axes=None, dt=1.0, t0=0.0,
               name=None) -> Gathers:
    """Wrap an in-memory / memmapped array (optionally with geometry)."""
    geometry = None
    if src is not None or rec is not None:
        if src is None or rec is None:
            raise ValueError("both src and rec are required for geometry")
        geometry = Geometry(src, rec)
    return Gathers(data, axes=axes, dt=dt, t0=t0, geometry=geometry, name=name)


def from_file(path, shape, dtype="float32", axes=None, dt=1.0, t0=0.0,
              src=None, rec=None, offset=0, name=None) -> Gathers:
    """Lazily open a raw binary file as a read-only memmap."""
    data = np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape), offset=offset)
    return from_array(data, src=src, rec=rec, axes=axes, dt=dt, t0=t0, name=name)


def from_segy(path, max_gb: float = 8.0, name=None) -> Gathers:
    """Open a SEG-Y file: shots grouped by FFID, geometry from trace headers.

    Reads dt from the binary header, source/receiver x-y from the standard
    trace-header words (SourceX/Y, GroupX/Y, honoring the coordinate scalar),
    and groups traces into shots by field record number (FFID). Ragged shots
    (unequal trace counts) are zero-padded to the largest and a warning is
    printed. Data is loaded into memory; files estimated above ``max_gb``
    are refused with advice (lazy SEG-Y is on the roadmap -- convert to .npy
    for now). Requires the optional ``segyio`` package.
    """
    import warnings
    try:
        import segyio
    except ImportError as e:
        raise ImportError("from_segy requires segyio: pip install segyio") from e

    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        ntr, nt = f.tracecount, len(f.samples)
        est_gb = ntr * nt * 4 / 1e9
        if est_gb > max_gb:
            raise MemoryError(
                f"{path}: ~{est_gb:.1f} GB > max_gb={max_gb}. Lazy SEG-Y is "
                "on the roadmap; for now convert to .npy and use from_file.")
        dt = segyio.tools.dt(f) / 1e6                    # us -> s
        H = segyio.TraceField
        ffid = f.attributes(H.FieldRecord)[:]
        sx = f.attributes(H.SourceX)[:].astype(float)
        sy = f.attributes(H.SourceY)[:].astype(float)
        gx = f.attributes(H.GroupX)[:].astype(float)
        gy = f.attributes(H.GroupY)[:].astype(float)
        sc = f.attributes(H.SourceGroupScalar)[:].astype(float)
        sc[sc == 0] = 1.0
        scale = np.where(sc > 0, sc, 1.0 / np.abs(sc))   # SEG-Y scalar rule
        sx *= scale; sy *= scale; gx *= scale; gy *= scale

        shots, order = np.unique(ffid, return_index=True)
        shots = shots[np.argsort(order)]                 # keep file order
        ns = len(shots)
        groups = [np.flatnonzero(ffid == s) for s in shots]
        nr = max(len(g) for g in groups)
        if min(len(g) for g in groups) != nr:
            warnings.warn(f"{path}: ragged shots "
                          f"({min(len(g) for g in groups)}..{nr} traces); "
                          "zero-padding to the largest")
        data = np.zeros((ns, nr, nt), dtype=np.float32)
        src = np.zeros((ns, 2))
        rec = np.zeros((ns, nr, 2))
        for i, g in enumerate(groups):
            for j, tr in enumerate(g):
                data[i, j] = f.trace[tr]
            src[i] = sx[g[0]], sy[g[0]]
            rec[i, :len(g), 0] = gx[g]
            rec[i, :len(g), 1] = gy[g]
            if len(g) < nr:                              # pad geometry too
                rec[i, len(g):] = rec[i, len(g) - 1]
    return from_array(data, src=src, rec=rec, dt=dt,
                      name=name or str(path))
