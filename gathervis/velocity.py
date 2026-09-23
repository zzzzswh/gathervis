"""NMO correction and velocity spectra.

The chain a velocity analysis actually runs through, one function per step::

    nmo               moveout removal for a trial v, with stretch mute
    front_mute        remove the direct wave before analysis
    semblance         coherency of an NMO-corrected gather, per time sample
    velocity_spectrum semblance over a grid of trial velocities  ->  (v, t) panel
    nmo_stack         v(t) -> the stacked trace
    dix               v_rms(t) -> interval velocity

Every panel here keeps gathervis's array convention: **time is the last
axis**. A gather is ``(ntrace, nt)``, and a velocity spectrum is ``(nv, nt)``
-- which is what makes the spectrum drawable by the same ImagePane as the
gather it came from, with velocity where trace number usually sits.

The whole module is vectorized over traces and samples, because it sits on
the interactive path: a velocity slider re-runs :func:`nmo` on every drag,
and a spectrum re-runs it once per trial velocity. It is CuPy-transparent in
the same way ``process.py`` is -- hand it arrays that are already on the GPU
and the arithmetic stays there.

Picking itself is not here. These are the numbers a velocity panel is drawn
from; deciding where the velocity function goes is a person's job, and the
panel that lets them do it is still to be built (see ``docs/todo.md``).
"""
from __future__ import annotations

import numpy as np

__all__ = ["nmo", "front_mute", "nmo_stack", "semblance",
           "velocity_spectrum", "velocity_axis", "dix"]


def _xp(a):
    """Array-module dispatch: cupy for arrays already on the GPU, else numpy."""
    try:
        import cupy
        if isinstance(a, cupy.ndarray):
            return cupy
    except ImportError:
        pass
    return np


def _asnumpy(a):
    """Host copy, whichever backend ``a`` lives on."""
    return a.get() if _xp(a) is not np else np.asarray(a)


# =============================================================================
# Moveout
# =============================================================================
def nmo(gather, offsets, v, dt, t0=0.0, stretch_mute=0.5):
    """Normal-moveout correction of one gather.

    Each sample at zero-offset time ``tau`` is read from the hyperbola
    ``t(h) = sqrt(tau^2 + h^2 / v(tau)^2)``, so a flat reflector's events
    line up horizontally when ``v`` is right, smile when it is too high and
    frown when it is too low. That is the whole basis of velocity analysis,
    and of the spectrum built in :func:`velocity_spectrum`.

    Parameters
    ----------
    gather : (ntrace, nt) array
        Traces to correct. Time is the last axis.
    offsets : (ntrace,) array
        Source-receiver distance per trace, in the same length unit as ``v``.
        Sign is irrelevant -- only ``h^2`` enters.
    v : float or (nt,) array
        NMO velocity. A scalar applies one velocity to the whole trace (what
        a spectrum's trial velocities are); an array is v(tau), one value per
        zero-offset time sample, which is what a pick gives you.
    dt : float
        Sample interval, seconds.
    t0 : float, optional
        Time of the *first sample*, seconds (``Gathers.t0``). Moveout is
        computed in absolute traveltime, so this matters whenever the record
        does not start at the shot instant.
    stretch_mute : float or None, optional
        Zero every sample stretched by more than this fraction
        (``(t - tau) / tau``). 0.5 is the usual 50% limit; ``None`` disables
        the mute. Stretch is worst at long offset and shallow time, where NMO
        smears the wavelet down to lower frequencies -- left in, it is what
        makes a stack go dull at the top.

    Returns
    -------
    (ntrace, nt) float32
        Moveout-corrected gather. Samples whose hyperbola falls off the end
        of the record, and muted samples, come back as zero.
    """
    xp = _xp(gather)
    a = xp.asarray(gather, dtype=xp.float32)
    if a.ndim != 2:
        raise ValueError(f"gather must be 2-D (ntrace, nt), got shape {a.shape}")
    ntr, nt = a.shape

    off = xp.asarray(offsets, dtype=xp.float32).reshape(-1)
    if off.size != ntr:
        raise ValueError(f"offsets has {off.size} entries but gather has {ntr} traces")

    dt = float(dt)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    tau = xp.asarray(t0, dtype=xp.float32) + xp.arange(nt, dtype=xp.float32) * dt

    vt = xp.asarray(v, dtype=xp.float32)
    if vt.ndim == 0:
        vt = xp.full(nt, vt, dtype=xp.float32)
    elif vt.shape != (nt,):
        raise ValueError(f"v must be a scalar or (nt,)={nt}, got shape {vt.shape}")
    if float(vt.min()) <= 0:
        raise ValueError("NMO velocities must be positive")

    # The hyperbola, then linear interpolation along it. take_along_axis does
    # per-trace gathering in one shot, so there is no Python loop over traces.
    # Moveout only ever moves a sample later (t_h >= tau), so the read
    # position cannot go off the *start* of the record -- only off the end.
    t_h = xp.sqrt(tau ** 2 + (off[:, None] / vt[None, :]) ** 2)
    pos = (t_h - float(t0)) / dt
    base = xp.floor(pos)
    frac = (pos - base).astype(xp.float32)
    idx = base.astype(xp.int64)
    inside = idx < nt - 1
    idx = xp.minimum(idx, nt - 2)
    lo = xp.take_along_axis(a, idx, axis=1)
    hi = xp.take_along_axis(a, idx + 1, axis=1)
    out = xp.where(inside, (1.0 - frac) * lo + frac * hi, xp.float32(0.0))

    if stretch_mute is not None and stretch_mute > 0:
        safe = xp.where(tau > 0, tau, xp.float32(1.0))     # avoids 0/0 at tau=0
        stretch = xp.where(tau > 0, (t_h - tau) / safe, xp.float32(0.0))
        out = xp.where(stretch > float(stretch_mute), xp.float32(0.0), out)

    return out.astype(xp.float32)


def front_mute(gather, offsets, dt, v, t0=0.0, pad=0.0, taper=0.04):
    """Zero everything arriving before a straight-line moveout, with a taper.

    Velocity analysis is about reflections, and the direct wave is not one:
    it is a straight line through the origin at the near-surface velocity,
    strong, and perfectly coherent, so semblance loves it. Left in, it puts
    a bright low-velocity smear across the shallow part of every spectrum
    drawn from the gather.

    The taper is not a nicety. A hard mute leaves a sharp edge running along
    a straight moveout trajectory, and that edge is itself coherent energy
    with its own apparent velocity, which puts back a weaker version of the
    thing the mute was meant to remove.

    Parameters
    ----------
    gather : (ntrace, nt) array
    offsets : (ntrace,) array
    dt : float
        Sample interval, seconds.
    v : float
        Mute velocity: the slope of the cut, in length units per second.
        Somewhere near the direct-wave / refractor velocity.
    t0 : float, optional
        Time of the first sample, seconds.
    pad : float, optional
        Extra time added to the cut, seconds -- how far below the direct
        wave to start keeping data.
    taper : float, optional
        Length of the cosine ramp from 0 to 1, seconds. 0 for a hard cut.

    Returns
    -------
    (ntrace, nt) float32
    """
    xp = _xp(gather)
    a = xp.asarray(gather, dtype=xp.float32)
    if a.ndim != 2:
        raise ValueError(f"gather must be 2-D (ntrace, nt), got shape {a.shape}")
    off = xp.abs(xp.asarray(offsets, dtype=xp.float32).reshape(-1))
    if off.size != a.shape[0]:
        raise ValueError(f"offsets has {off.size} entries but gather has "
                         f"{a.shape[0]} traces")
    if float(v) <= 0:
        raise ValueError(f"mute velocity must be positive, got {v}")

    tau = xp.asarray(t0, dtype=xp.float32) + xp.arange(a.shape[1], dtype=xp.float32) * float(dt)
    onset = off[:, None] / float(v) + float(pad)
    if taper and taper > 0:
        ramp = xp.clip((tau[None, :] - onset) / float(taper), 0.0, 1.0)
        weight = 0.5 - 0.5 * xp.cos(xp.float32(np.pi) * ramp)
    else:
        weight = (tau[None, :] >= onset).astype(xp.float32)
    return (a * weight).astype(xp.float32)


def nmo_stack(gather, offsets, v, dt, t0=0.0, stretch_mute=0.5,
              normalize=True):
    """NMO a gather and sum it into one trace.

    With ``normalize`` on, each sample is divided by the number of live
    traces contributing to it rather than by the nominal fold. Mutes and
    off-record samples thin the fold out with depth and offset, and dividing
    by a constant instead would make the shallow, heavily muted part of the
    stack fade for no physical reason.

    Returns ``(nt,) float32``.
    """
    xp = _xp(gather)
    corrected = nmo(gather, offsets, v, dt, t0=t0, stretch_mute=stretch_mute)
    total = corrected.sum(axis=0)
    if not normalize:
        return total.astype(xp.float32)
    live = (xp.abs(corrected) > 1e-12).sum(axis=0).astype(xp.float32)
    return (total / xp.maximum(live, xp.float32(1.0))).astype(xp.float32)


# =============================================================================
# Coherency
# =============================================================================
def semblance(corrected, window=21, min_eff_fold=4.0):
    """Sliding-window semblance of an NMO-corrected gather, per time sample.

    Semblance is the energy of the stack over the stacked energy,
    ``sum_win (sum_k a_k)^2 / (N * sum_win sum_k a_k^2)``, which is 1 where
    the traces are identical and ~1/N where they are noise. ``N`` is taken as
    the *live* fold in the window, not the nominal one, so a time gate that
    the stretch mute has eaten into is still scored fairly.

    Parameters
    ----------
    corrected : (ntrace, nt) array
        Output of :func:`nmo`.
    window : int, optional
        Window length in samples; forced odd.
    min_eff_fold : float, optional
        Below this live fold the window is scored 0 rather than given the
        flattering semblance that two surviving traces would always earn.

    Returns
    -------
    (nt,) float32 in [0, 1]
    """
    xp = _xp(corrected)
    a = xp.asarray(corrected, dtype=xp.float32)
    if a.ndim != 2:
        raise ValueError(f"expected (ntrace, nt), got shape {a.shape}")
    w = int(window) | 1                                   # 20 -> 21, 21 -> 21
    if w < 1:
        raise ValueError(f"window must be positive, got {window}")

    live = (xp.abs(a) > 1e-12).astype(xp.float32)
    kernel = xp.ones(w, dtype=xp.float32)
    num = xp.convolve((a.sum(axis=0)) ** 2, kernel, mode="same")
    den_energy = xp.convolve((a * a).sum(axis=0), kernel, mode="same")
    fold = xp.convolve(live.sum(axis=0), kernel, mode="same") / w

    s = xp.where(fold >= float(min_eff_fold),
                 num / (fold * den_energy + 1e-12), xp.float32(0.0))
    return xp.clip(s, 0.0, 1.0).astype(xp.float32)


# =============================================================================
# Velocity spectrum
# =============================================================================
def velocity_axis(vmin=1400.0, vmax=4500.0, nv=100) -> np.ndarray:
    """The trial-velocity grid a spectrum is scanned over, ``(nv,) float32``."""
    if nv < 2:
        raise ValueError(f"nv must be at least 2, got {nv}")
    if not vmax > vmin:
        raise ValueError(f"need vmax > vmin, got vmin={vmin}, vmax={vmax}")
    return np.linspace(float(vmin), float(vmax), int(nv)).astype(np.float32)


def velocity_spectrum(gather, offsets, dt, v_grid=None, t0=0.0,
                      vmin=1400.0, vmax=4500.0, nv=100,
                      window=21, stretch_mute=0.5, min_eff_fold=4.0,
                      normalize_per_t=False):
    """Semblance over a grid of trial velocities: the panel you pick on.

    One NMO plus one semblance per trial velocity. Cost is
    ``nv * ntrace * nt``, all vectorized -- a 120-trace, 1500-sample CMP over
    100 velocities is a fraction of a second, which is what makes it usable
    as something you re-run when the mute or the window changes.

    Parameters
    ----------
    gather : (ntrace, nt) array
    offsets : (ntrace,) array
    dt : float
        Sample interval, seconds.
    v_grid : (nv,) array, optional
        Trial velocities. Built from ``vmin/vmax/nv`` when omitted.
    t0 : float, optional
        Time of the first sample, seconds.
    window, stretch_mute, min_eff_fold
        Passed through to :func:`nmo` and :func:`semblance`.
    normalize_per_t : bool, optional
        Divide each time row by its own (smoothed) maximum. This is a display
        choice: it makes weak deep events as bright as the strong shallow
        ones, at the cost of raising the noise floor wherever there is no
        real event at all.

    Returns
    -------
    spec : (nv, nt) float32 in [0, 1]
    v_grid : (nv,) float32
    """
    xp = _xp(gather)
    if v_grid is None:
        v_grid = velocity_axis(vmin, vmax, nv)
    v_grid = xp.asarray(v_grid, dtype=xp.float32).reshape(-1)

    nt = xp.asarray(gather).shape[1]
    spec = xp.empty((v_grid.size, nt), dtype=xp.float32)
    for j in range(int(v_grid.size)):
        corrected = nmo(gather, offsets, v_grid[j], dt, t0=t0,
                        stretch_mute=stretch_mute)
        spec[j] = semblance(corrected, window=window,
                            min_eff_fold=min_eff_fold)

    if normalize_per_t:
        peak = spec.max(axis=0)
        kernel = xp.ones(int(window) | 1, dtype=xp.float32) / (int(window) | 1)
        peak = xp.convolve(peak, kernel, mode="same")
        floor = max(0.05, float(peak.max()) * 0.05)
        spec = xp.clip(spec / xp.where(peak > floor, peak, xp.inf), 0.0, 1.0)

    return spec.astype(xp.float32), v_grid


# =============================================================================
# Dix
# =============================================================================
def dix(v_rms, t, smooth=0, floor=500.0):
    """Interval velocity from an RMS (stacking) velocity function.

    Dix's formula, ``v_int(i)^2 = (v^2 t |_i - v^2 t |_{i-1}) / (t_i -
    t_{i-1})``. It differentiates, so it amplifies every wobble in the pick:
    a pick that looks smooth enough to stack with can still produce interval
    velocities that swing by a kilometre a second between layers. Hence the
    two guards -- ``smooth`` samples of moving average on the input, and a
    ``floor`` under the result where the difference comes out negative.

    Parameters
    ----------
    v_rms : (nt,) array
        RMS velocity per zero-offset time, as picked.
    t : (nt,) array
        Zero-offset time, seconds, increasing.
    smooth : int, optional
        Moving-average length in samples applied to ``v_rms`` first.
    floor : float, optional
        Lower bound on the output, used wherever Dix would give a negative
        or unphysically small square.

    Returns
    -------
    (nt,) float32 interval velocity.
    """
    v = _asnumpy(v_rms).astype(np.float64).reshape(-1)
    t = _asnumpy(t).astype(np.float64).reshape(-1)
    if v.size != t.size:
        raise ValueError(f"v_rms has {v.size} entries but t has {t.size}")
    if v.size < 2:
        return v.astype(np.float32)
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")

    if smooth and smooth > 1:
        k = np.ones(int(smooth)) / int(smooth)
        v = np.convolve(v, k, mode="same")
        edge = int(smooth) // 2                 # convolve fades the ends: hold them
        if edge:
            v[:edge], v[-edge:] = v[edge], v[-edge - 1]

    num = np.diff(v ** 2 * t)
    den = np.diff(t)
    vi = np.empty_like(v)
    vi[0] = v[0]
    vi[1:] = np.sqrt(np.maximum(num / den, float(floor) ** 2))
    return vi.astype(np.float32)
