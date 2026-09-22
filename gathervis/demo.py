"""Tiny analytic synthetic (hyperbolic events, Ricker wavelet).

Dependency-free stand-in for real modelling; unit tests and the quickstart use
this so the package works without torch/deepwave. Real demo data should be
generated with deepwave (see examples/deepwave_line2d.py).
"""
from __future__ import annotations

import numpy as np

from .core import Gathers, from_array

__all__ = ["synthetic_line", "synthetic_patch"]


def _ricker(tau, f):
    x = (np.pi * f * tau) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def synthetic_line(ns=32, nr=96, nt=751, dt=0.002, f=20.0,
                   d_src=50.0, d_rec=25.0,
                   layers=((300.0, 1800.0), (700.0, 2300.0)),
                   noise=0.02, seed=0) -> Gathers:
    """A 2-D line with a fixed receiver spread and shots rolling along it.

    Returns a ``Gathers`` of shape (ns, nr, nt) with geometry attached.
    """
    rng = np.random.default_rng(seed)
    rec_x = np.arange(nr) * d_rec
    src_x = np.linspace(rec_x[0], rec_x[-1], ns)
    src = np.stack([src_x, np.zeros(ns)], axis=1)
    rec = np.stack([rec_x, np.zeros(nr)], axis=1)

    off = np.abs(src_x[:, None] - rec_x[None, :])          # (ns, nr)
    t = np.arange(nt, dtype=np.float32) * dt               # (nt,)
    data = np.zeros((ns, nr, nt), dtype=np.float32)

    # direct wave
    tau = t[None, None, :] - (off / 1500.0)[:, :, None]
    data += 0.4 * _ricker(tau, f).astype(np.float32)

    # reflections: hyperbola per layer, mild geometric spreading
    for depth, vel in layers:
        t_ref = np.sqrt((2.0 * depth / vel) ** 2 + (off / vel) ** 2)
        tau = t[None, None, :] - t_ref[:, :, None]
        data += (_ricker(tau, f) / (1.0 + t_ref[:, :, None])).astype(np.float32)

    if noise:
        data += (noise * data.std() *
                 rng.standard_normal(data.shape)).astype(np.float32)
    return from_array(data, src=src, rec=rec, dt=dt)


def synthetic_patch(ns=6, nly=8, nlx=48, nt=501, dt=0.002, f=20.0,
                    d_rec=25.0, d_line=200.0, d_src=100.0, src_lines=1,
                    layers=((400.0, 2000.0), (900.0, 2600.0)),
                    noise=0.02, seed=0) -> Gathers:
    """A 3-D patch: ``nly`` receiver lines of ``nlx`` stations, and ``ns``
    shots.

    With ``src_lines=1`` (the default) the shots roll along a single line at
    y = 0. With ``src_lines > 1`` they sit on that many shot lines running
    cross-line, i.e. orthogonal to the receiver lines -- the standard land
    3-D template, and the layout whose CMP fold map actually looks like one
    (``ns`` must divide evenly among the shot lines).

    Returns a ``Gathers`` of shape ``(ns, nly, nlx, nt)`` -- the 4-D layout
    the viewer reads as one 3-D shot record per shot. Because every trace's
    traveltime follows its own source-receiver offset, the acquisition-order
    panel shows the ``nly`` nested hyperbolas a real 3-D shot shows, with
    apices stepping as the cross-line distance grows.
    """
    rng = np.random.default_rng(seed)
    rec_x = np.arange(nlx) * d_rec
    line_y = (np.arange(nly) - (nly - 1) / 2.0) * d_line
    rx = np.repeat(rec_x[None, :], nly, axis=0)                 # (nly, nlx)
    ry = np.repeat(line_y[:, None], nlx, axis=1)
    rec = np.stack([rx.ravel(), ry.ravel()], axis=1)            # (nly*nlx, 2)

    if src_lines <= 1:
        src_x = rec_x[0] + d_src * np.arange(ns)
        src = np.stack([src_x, np.zeros(ns)], axis=1)           # on y = 0
    else:
        if ns % src_lines:
            raise ValueError(f"ns={ns} does not divide into "
                             f"src_lines={src_lines} shot lines")
        per = ns // src_lines
        # shot lines across the receiver patch, shots stepping along them
        line_x = np.linspace(rec_x[0], rec_x[-1], src_lines)
        shot_y = (np.arange(per) - (per - 1) / 2.0) * d_src
        sx, sy = np.meshgrid(line_x, shot_y, indexing="ij")
        src = np.stack([sx.ravel(), sy.ravel()], axis=1)

    # (ns, nly*nlx) source-receiver offsets, then the usual hyperbolae
    off = np.hypot(rec[None, :, 0] - src[:, None, 0],
                   rec[None, :, 1] - src[:, None, 1]).astype(np.float32)
    t = np.arange(nt, dtype=np.float32) * dt
    data = np.zeros((ns, nly * nlx, nt), dtype=np.float32)

    tau = t[None, None, :] - (off / 1500.0)[:, :, None]         # direct wave
    data += 0.4 * _ricker(tau, f).astype(np.float32)
    for depth, vel in layers:
        t_ref = np.sqrt((2.0 * depth / vel) ** 2 + (off / vel) ** 2)
        tau = t[None, None, :] - t_ref[:, :, None]
        data += (_ricker(tau, f) / (1.0 + t_ref[:, :, None])).astype(np.float32)
    if noise:
        data += (noise * data.std() *
                 rng.standard_normal(data.shape)).astype(np.float32)

    return from_array(data.reshape(ns, nly, nlx, nt), src=src, rec=rec, dt=dt,
                      name=f"synthetic 3-D patch ({nly} lines x {nlx} stations)")
