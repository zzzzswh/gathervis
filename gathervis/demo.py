"""Tiny analytic synthetic (hyperbolic events, Ricker wavelet).

Dependency-free stand-in for real modelling; unit tests and the quickstart use
this so the package works without torch/deepwave. Real demo data should be
generated with deepwave (see examples/deepwave_line2d.py).
"""
from __future__ import annotations

import numpy as np

from .core import Gathers, from_array

__all__ = ["synthetic_line"]


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
