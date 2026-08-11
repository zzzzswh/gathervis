"""Amplitude/display helpers. This is the per-frame hot path: keep it flat and vectorized.

Pipeline per displayed panel:  decimate -> quantize(uint8) -> ship to browser.
uint8 shipping cuts the wire payload to 1/4 of float32 before compression.
"""
from __future__ import annotations

import numpy as np

__all__ = ["robust_clim", "decimate", "quantize"]


def robust_clim(a, perc: float = 98.0, max_samples: int = 200_000):
    """Symmetric color limits (-v, v) from a percentile of |amplitude|.

    ``a`` may be a full array or an already-subsampled 1-D sample.
    Large inputs are strided-subsampled, so this is cheap even on memmaps.
    """
    a = a.reshape(-1)
    step = max(1, a.shape[0] // max_samples)
    sample = np.asarray(a[::step], dtype=np.float32)
    v = float(np.percentile(np.abs(sample), perc))
    if not np.isfinite(v) or v <= 0.0:
        v = 1.0
    return (-v, v)


def decimate(img: np.ndarray, max_px=(1600, 1600)) -> np.ndarray:
    """Stride-decimate a 2-D panel so no dimension exceeds ``max_px``.

    Pure striding: zero-copy on ndarrays, and on memmaps it prevents ever
    reading the skipped bytes. Anti-aliasing (LOD pyramids) is an M2 topic.
    """
    s0 = -(-img.shape[0] // max_px[0])  # ceil div
    s1 = -(-img.shape[1] // max_px[1])
    return img[::max(s0, 1), ::max(s1, 1)]


def quantize(img: np.ndarray, clim) -> np.ndarray:
    """Map float amplitudes to uint8 [0, 255] for cheap transport."""
    lo, hi = clim
    q = (np.asarray(img, dtype=np.float32) - lo) * (255.0 / (hi - lo))
    np.clip(q, 0.0, 255.0, out=q)
    return q.astype(np.uint8)
