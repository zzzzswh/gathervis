"""Command-line viewer: look at saved data without paying the torch import.

    gathervis line2d_data.npy --geom line2d_geom.npz
    gathervis shots.bin --shape 60 192 1200 --dt 0.002
    gathervis vol.npy --axes recy recx time            # slice view
"""
from __future__ import annotations

import argparse

import numpy as np

from .core import Gathers, from_array, from_file


def _parser():
    p = argparse.ArgumentParser(
        prog="gathervis",
        description="Interactive pre-stack gather viewer (browser-based).")
    p.add_argument("data", help=".npy file (lazily memmapped) or raw binary")
    p.add_argument("--shape", type=int, nargs="+",
                   help="required for raw binary, e.g. --shape 60 192 1200")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--dt", type=float, default=None, help="sample interval (s)")
    p.add_argument("--axes", nargs="+", help="e.g. --axes shot rec time")
    p.add_argument("--view", choices=["browse", "slices"], default=None)
    p.add_argument("--geom", help=".npz with 'src', 'rec' (and optionally 'dt')")
    p.add_argument("--name", default=None,
                   help="description shown in the info card")
    p.add_argument("--cmap", default="gray")
    p.add_argument("--port", type=int, default=8080,
                   help="0 = auto-pick a free port")
    p.add_argument("--address", default="127.0.0.1")
    return p


def _load(args) -> Gathers:
    src = rec = None
    dt = args.dt
    if args.geom:
        g = np.load(args.geom)
        src, rec = g["src"], g["rec"]
        if dt is None and "dt" in g:
            dt = float(g["dt"])
    dt = 1.0 if dt is None else dt
    axes = tuple(args.axes) if args.axes else None

    if args.data.lower().endswith((".sgy", ".segy")):
        from .core import from_segy
        return from_segy(args.data)            # dt + geometry from headers
    if args.data.endswith(".npy"):
        data = np.load(args.data, mmap_mode="r")   # lazy
        return from_array(data, src=src, rec=rec, axes=axes, dt=dt,
                          name=args.name)
    if not args.shape:
        raise SystemExit("--shape is required for raw binary files")
    return from_file(args.data, tuple(args.shape), dtype=args.dtype,
                     axes=axes, dt=dt, src=src, rec=rec, name=args.name)


def main(argv=None):
    args = _parser().parse_args(argv)
    g = _load(args)
    print(g)
    from .viewer import show
    show(g, view=args.view, cmap=args.cmap, port=args.port,
         address=args.address)


if __name__ == "__main__":
    main()
