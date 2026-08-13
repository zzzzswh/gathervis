"""Amplitude/display helpers. This is the per-frame hot path: keep it flat and vectorized.

Pipeline per displayed panel:  decimate -> quantize(uint8) -> ship to browser.
uint8 shipping cuts the wire payload to 1/4 of float32 before compression.
"""
from __future__ import annotations

import numpy as np

__all__ = ["robust_clim", "decimate", "quantize", "bandpass", "agc",
           "trace_balance", "sta_lta", "pick_first_breaks"]


def _xp(a):
    """Array-module dispatch: returns cupy if ``a`` is a CuPy array (data
    already on the GPU), else numpy. Lets bandpass/agc/trace_balance run
    GPU-side transparently on GPU servers -- no code changes needed."""
    try:
        import cupy
        if isinstance(a, cupy.ndarray):
            return cupy
    except ImportError:
        pass
    return np


def robust_clim(a, perc: float = 98.0, max_samples: int = 200_000,
                symmetric: bool = True):
    """Robust color limits from percentiles.

    ``symmetric=True`` (wavefields, zero-mean): (-v, v) with v the ``perc``
    percentile of |amplitude|. ``symmetric=False`` (property volumes such as
    velocity): the (100-perc, perc) percentiles of the values themselves.

    ``a`` may be a full array or an already-subsampled 1-D sample.
    Large inputs are strided-subsampled, so this is cheap even on memmaps.
    """
    a = a.reshape(-1)
    step = max(1, a.shape[0] // max_samples)
    sample = np.asarray(a[::step], dtype=np.float32)
    if symmetric:
        v = float(np.percentile(np.abs(sample), perc))
        if not np.isfinite(v) or v <= 0.0:
            v = 1.0
        return (-v, v)
    lo, hi = (float(x) for x in np.percentile(sample, [100.0 - perc, perc]))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return (-1.0, 1.0)
    return (lo, hi)


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


def bandpass(a, dt: float, f1=None, f2=None, f3=None, f4=None) -> np.ndarray:
    """Zero-phase trapezoid (Ormsby-style) filter along the last (time) axis.

    The amplitude response ramps linearly 0->1 over (f1, f2) -- the low-cut /
    high-pass side -- and 1->0 over (f3, f4) -- the high-cut / low-pass side.
    Either side may be None to pass everything on that side:

      high-pass: f1, f2         low-pass: f3, f4         band-pass: all four

    Applied in the frequency domain (rfft), so it is exactly zero-phase and
    costs one FFT round trip over just the displayed gather.
    """
    xp = _xp(a)
    a = xp.asarray(a, dtype=xp.float32)
    nt = a.shape[-1]
    f = xp.fft.rfftfreq(nt, float(dt)).astype(xp.float32)
    h = xp.ones_like(f)
    if f1 is not None and f2 is not None and f2 > f1 >= 0:
        h *= xp.clip((f - f1) / (f2 - f1), 0.0, 1.0)
    if f3 is not None and f4 is not None and f4 > f3 >= 0:
        h *= xp.clip((f4 - f) / (f4 - f3), 0.0, 1.0)
    spec = xp.fft.rfft(a, axis=-1)
    spec *= h
    return xp.fft.irfft(spec, n=nt, axis=-1).astype(xp.float32)


def agc(a, dt: float, window: float = 0.5) -> np.ndarray:
    """Automatic gain control: divide by the sliding-window RMS along the
    last (time) axis. ``window`` is the full window length in seconds; edges
    use the true (shrinking) window. Zero-padded regions stay zero.

    Runs on the GPU transparently when ``a`` is a CuPy array.
    """
    xp = _xp(a)
    a = xp.asarray(a, dtype=xp.float32)
    nt = a.shape[-1]
    n = max(3, int(round(window / float(dt))))
    h = n // 2
    c = xp.cumsum(a * a, axis=-1, dtype=xp.float64)
    c = xp.concatenate([xp.zeros_like(c[..., :1]), c], axis=-1)   # len nt+1
    idx = xp.arange(nt)
    i0 = xp.clip(idx - h, 0, nt)
    i1 = xp.clip(idx + h + 1, 0, nt)
    rms = xp.sqrt((c[..., i1] - c[..., i0]) / (i1 - i0)).astype(xp.float32)
    eps = 1e-4 * float(rms.max()) + 1e-30       # keeps dead zones quiet
    return a / (rms + eps)


def trace_balance(a) -> np.ndarray:
    """Equalize traces: divide each trace by its own RMS (last axis = time).

    Runs on the GPU transparently when ``a`` is a CuPy array.
    """
    xp = _xp(a)
    a = xp.asarray(a, dtype=xp.float32)
    rms = xp.sqrt((a * a).mean(axis=-1, keepdims=True))
    eps = 1e-4 * float(rms.max()) + 1e-30
    return a / (rms + eps)


def sta_lta(a, dt: float, sta: float = 0.02, lta: float = 0.2) -> np.ndarray:
    """Causal, *gapped* STA/LTA energy ratio along the last (time) axis: the
    LTA window ends where the STA window starts, so an onset drives STA up
    while LTA still holds pre-onset noise -- early first breaks right after
    the record start can still trigger. The ratio is zeroed where the LTA
    window holds fewer than max(4, sta-samples) samples.

    Runs on the GPU transparently when ``a`` is a CuPy array.
    """
    xp = _xp(a)
    a = xp.asarray(a, dtype=xp.float32)
    e = (a * a).astype(xp.float64)
    nt = e.shape[-1]
    ns_ = max(1, int(round(sta / float(dt))))
    nl_ = max(2, int(round(lta / float(dt))))
    c = xp.cumsum(e, axis=-1)
    c = xp.concatenate([xp.zeros_like(c[..., :1]), c], axis=-1)
    idx = xp.arange(nt)
    s_i0 = xp.clip(idx - ns_ + 1, 0, nt)
    sta_v = (c[..., idx + 1] - c[..., s_i0]) / (idx + 1 - s_i0)
    l_i1 = xp.clip(idx - ns_ + 1, 0, nt)       # LTA ends at the STA start
    l_i0 = xp.clip(idx - ns_ - nl_ + 1, 0, nt)
    cnt = xp.maximum(l_i1 - l_i0, 1)
    lta_v = (c[..., l_i1] - c[..., l_i0]) / cnt
    r = (sta_v / (lta_v + 1e-30)).astype(xp.float32)
    r[..., (l_i1 - l_i0) < max(4, ns_)] = 0.0  # not enough pre-window yet
    return r


def pick_first_breaks(a, dt: float, sta: float = 0.02, lta: float = 0.2,
                      thresh: float = 4.0) -> np.ndarray:
    """First-break times (s, relative to the first sample) per trace: the
    first gapped-STA/LTA excursion that *stays* above ``thresh`` for at least
    sta/2 (anti-spike criterion, rejects single-sample noise triggers). NaN
    where a trace never triggers. The pick sits at the trigger, near the
    energy onset.
    """
    xp = _xp(a)
    r = sta_lta(a, dt, sta, lta)
    nt = r.shape[-1]
    ns_ = max(1, int(round(sta / float(dt))))
    trig = r > thresh
    w = max(2, ns_ // 2)                       # must hold for ~sta/2
    ci = xp.cumsum(trig.astype(xp.int32), axis=-1)
    ci = xp.concatenate([xp.zeros_like(ci[..., :1]), ci], axis=-1)
    idx = xp.arange(nt)
    i1 = xp.clip(idx + w, 0, nt)
    sustained = (ci[..., i1] - ci[..., idx]) == (i1 - idx)
    sustained &= trig
    k = xp.argmax(sustained, axis=-1)
    return xp.where(sustained.any(axis=-1), k * float(dt),
                    xp.nan).astype(xp.float32)
