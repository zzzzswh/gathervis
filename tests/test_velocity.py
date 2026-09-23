"""NMO, velocity spectra and automatic picking (gathervis.velocity)."""
import numpy as np
import pytest

from gathervis import velocity as V

DT = 0.002
REFLECTORS = ((0.30, 1800.0), (0.60, 2200.0), (0.95, 2600.0), (1.40, 3100.0))


def _ricker(freq=25.0, half=60, dt=DT):
    t = (np.arange(-half, half + 1)) * dt
    a = (np.pi * freq * t) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


def synthetic_cmp(nt=1000, fold=48, dx=50.0, reflectors=REFLECTORS,
                  noise=0.05, seed=0):
    """A CMP gather with known (t0, v_rms) hyperbolas: (ntrace, nt), offsets."""
    rng = np.random.default_rng(seed)
    offsets = ((np.arange(fold) + 1) * dx).astype(np.float32)
    w, half = _ricker(), 60
    g = np.zeros((fold, nt), dtype=np.float32)
    for t0, v in reflectors:
        for k, t_h in enumerate(np.sqrt(t0 ** 2 + (offsets / v) ** 2)):
            i = int(round(t_h / DT))
            lo, hi = max(i - half, 0), min(i + half + 1, nt)
            g[k, lo:hi] += w[lo - (i - half):hi - (i - half)]
    if noise:
        g += (noise * rng.standard_normal(g.shape)).astype(np.float32)
    return g, offsets


@pytest.fixture(scope="module")
def cmp_gather():
    return synthetic_cmp()


@pytest.fixture(scope="module")
def spectrum(cmp_gather):
    g, offsets = cmp_gather
    spec, v_grid = V.velocity_spectrum(g, offsets, DT, vmin=1400, vmax=4000,
                                       nv=120, normalize_per_t=True)
    return spec, v_grid, (np.arange(g.shape[1]) * DT).astype(np.float32)


# ---------------- NMO ----------------
def test_nmo_flattens_an_event_at_the_right_velocity():
    """The whole point: at the true velocity the hyperbola becomes a line."""
    g, offsets = synthetic_cmp(reflectors=((0.60, 2200.0),), noise=0.0)
    flat = V.nmo(g, offsets, 2200.0, DT, stretch_mute=None)
    assert np.ptp(flat.argmax(axis=1)) <= 2       # aligned to within a sample

    # too slow -> over-correction, the event arches up towards short time
    slow = V.nmo(g, offsets, 1800.0, DT, stretch_mute=None)
    assert slow.argmax(axis=1)[-1] < slow.argmax(axis=1)[0]


def test_nmo_leaves_zero_offset_untouched():
    g, _ = synthetic_cmp(fold=4, noise=0.0)
    out = V.nmo(g, np.zeros(4, np.float32), 2000.0, DT, stretch_mute=None)
    assert np.allclose(out, g, atol=1e-5)


def test_nmo_accepts_a_velocity_function():
    g, offsets = synthetic_cmp(noise=0.0)
    t = np.arange(g.shape[1]) * DT
    v_t = np.interp(t, [r[0] for r in REFLECTORS],
                    [r[1] for r in REFLECTORS]).astype(np.float32)
    # With the mute off, the shallow event is coherent in *time* but not in
    # waveform: at 2400 m and 1800 m/s its moveout is over a second, and the
    # stretch that removing it causes smears the far traces beyond
    # recognition. Muting the stretched samples is what makes the gather
    # stack, which is the whole reason the mute is on by default.
    flat = V.nmo(g, offsets, v_t, DT, stretch_mute=0.5)
    coherency = V.semblance(flat, window=21)
    for t0, _ in REFLECTORS:                      # every event flat at once
        assert coherency[int(t0 / DT)] > 0.8
    assert V.semblance(V.nmo(g, offsets, v_t, DT, stretch_mute=None))[150] < 0.8


def test_stretch_mute_zeroes_the_far_shallow_corner():
    g, offsets = synthetic_cmp(noise=0.0)
    muted = V.nmo(g, offsets, 1600.0, DT, stretch_mute=0.5)
    kept = V.nmo(g, offsets, 1600.0, DT, stretch_mute=None)
    assert np.all(muted[-1, 1:100] == 0)          # far offset, shallow
    assert muted[0, 0] == kept[0, 0]              # tau=0: no stretch to define
    assert np.any(kept[-1, :100] != 0)
    assert np.count_nonzero(muted) < np.count_nonzero(kept)


def test_nmo_rejects_bad_input():
    g = np.zeros((8, 50), np.float32)
    offsets = ((np.arange(8) + 1) * 50.0).astype(np.float32)
    with pytest.raises(ValueError, match="offsets"):
        V.nmo(g, offsets[:4], 2000.0, DT)
    with pytest.raises(ValueError, match="positive"):
        V.nmo(g, offsets, -1.0, DT)
    with pytest.raises(ValueError, match="scalar"):
        V.nmo(g, offsets, np.ones(7, np.float32), DT)
    with pytest.raises(ValueError, match="2-D"):
        V.nmo(g[None], offsets, 2000.0, DT)


def test_t0_shifts_the_hyperbola():
    """A record that starts late must not be treated as starting at zero."""
    g, offsets = synthetic_cmp(reflectors=((0.60, 2200.0),), noise=0.0)
    late = V.nmo(g[:, 100:], offsets, 2200.0, DT, t0=0.2, stretch_mute=None)
    assert np.ptp(late.argmax(axis=1)) <= 2
    wrong = V.nmo(g[:, 100:], offsets, 2200.0, DT, t0=0.0, stretch_mute=None)
    assert np.ptp(wrong.argmax(axis=1)) > 2


# ---------------- semblance ----------------
def test_semblance_is_one_for_identical_traces_and_low_for_noise():
    trace = _ricker()[None].repeat(12, axis=0)
    coherent = np.zeros((12, 400), np.float32)
    coherent[:, 180:180 + trace.shape[1]] = trace
    assert V.semblance(coherent, window=21).max() > 0.99

    rng = np.random.default_rng(3)
    noise = rng.standard_normal((12, 400)).astype(np.float32)
    assert V.semblance(noise, window=21).mean() < 0.3


def test_semblance_respects_min_eff_fold():
    g = np.zeros((20, 300), np.float32)
    g[:3, 120:161] = _ricker(half=20)[None, :]
    assert V.semblance(g, min_eff_fold=10.0).max() == 0.0   # only 3 live
    assert V.semblance(g, min_eff_fold=2.0).max() > 0.0


def test_semblance_forces_an_odd_window():
    g, offsets = synthetic_cmp(noise=0.0)
    corrected = V.nmo(g, offsets, 2200.0, DT)
    assert np.array_equal(V.semblance(corrected, window=20),
                          V.semblance(corrected, window=21))


# ---------------- spectrum ----------------
def test_spectrum_peaks_at_the_true_velocities(spectrum):
    spec, v_grid, t = spectrum
    assert spec.shape == (v_grid.size, t.size)
    assert 0.0 <= spec.min() and spec.max() <= 1.0
    for t0, v_true in REFLECTORS:
        column = spec[:, int(t0 / DT)]
        assert abs(v_grid[column.argmax()] - v_true) / v_true < 0.06


def test_spectrum_axis_order_is_velocity_then_time(cmp_gather):
    """(nv, nt) -- time last, like every other panel in gathervis."""
    g, offsets = cmp_gather
    spec, v_grid = V.velocity_spectrum(g, offsets, DT, nv=17)
    assert spec.shape == (17, g.shape[1]) == (v_grid.size, g.shape[1])


def test_velocity_axis_validates():
    assert V.velocity_axis(1500, 3000, 4).tolist() == [1500, 2000, 2500, 3000]
    with pytest.raises(ValueError):
        V.velocity_axis(3000, 1500, 10)
    with pytest.raises(ValueError):
        V.velocity_axis(1500, 3000, 1)


# ---------------- stack ----------------
def test_stack_puts_energy_at_the_zero_offset_times(cmp_gather):
    g, offsets = cmp_gather
    t = np.arange(g.shape[1]) * DT
    v_t = np.interp(t, [r[0] for r in REFLECTORS],
                    [r[1] for r in REFLECTORS]).astype(np.float32)
    stacked = V.nmo_stack(g, offsets, v_t, DT)
    assert stacked.shape == (g.shape[1],)
    for t0, _ in REFLECTORS:
        gate = slice(int((t0 - 0.03) / DT), int((t0 + 0.03) / DT))
        assert np.abs(stacked[gate]).max() > 0.4 * np.abs(stacked).max()


def test_stack_normalisation_divides_by_the_live_fold():
    """Not by the nominal fold -- the mute thins it out with time."""
    g, offsets = synthetic_cmp(noise=0.0)
    v = np.full(g.shape[1], 1700.0, np.float32)
    corrected = V.nmo(g, offsets, v, DT, stretch_mute=0.3)
    live = np.maximum((np.abs(corrected) > 1e-12).sum(axis=0), 1)
    assert live.min() < live.max()                # the mute really does thin it
    assert np.allclose(V.nmo_stack(g, offsets, v, DT, stretch_mute=0.3),
                       corrected.sum(axis=0) / live, atol=1e-5)
    assert np.allclose(V.nmo_stack(g, offsets, v, DT, stretch_mute=0.3,
                                   normalize=False),
                       corrected.sum(axis=0), atol=1e-4)


# ---------------- Dix ----------------
def test_dix_inverts_a_two_layer_rms_function():
    """Build v_rms from known intervals, then get the intervals back."""
    v_int = np.array([1800.0, 2600.0], np.float32)
    t_top, t_base = 0.4, 1.0
    v_rms_base = np.sqrt((v_int[0] ** 2 * t_top
                          + v_int[1] ** 2 * (t_base - t_top)) / t_base)
    t = np.array([t_top, t_base], np.float32)
    out = V.dix(np.array([v_int[0], v_rms_base], np.float32), t)
    assert abs(out[0] - v_int[0]) < 1.0
    assert abs(out[1] - v_int[1]) < 1.0


def test_dix_floors_an_inversion_instead_of_taking_a_negative_root():
    t = np.array([0.5, 1.0], np.float32)
    out = V.dix(np.array([3000.0, 1500.0], np.float32), t, floor=600.0)
    assert out[1] == pytest.approx(600.0)


def test_dix_validates():
    with pytest.raises(ValueError, match="increasing"):
        V.dix(np.ones(3, np.float32), np.array([0.0, 0.5, 0.2], np.float32))
    with pytest.raises(ValueError, match="entries"):
        V.dix(np.ones(3, np.float32), np.ones(4, np.float32))


# ---------------- front mute ----------------
def test_front_mute_removes_a_linear_arrival():
    """A direct wave is a straight line; the mute is shaped to follow it."""
    fold, nt, v_direct = 24, 600, 1500.0
    offsets = ((np.arange(fold) + 1) * 40.0).astype(np.float32)
    g = np.zeros((fold, nt), np.float32)
    w, half = _ricker(half=20), 20
    for k, h in enumerate(offsets):                 # the direct wave itself
        i = int(round(h / v_direct / DT))
        lo, hi = max(i - half, 0), min(i + half + 1, nt)
        g[k, lo:hi] = w[lo - (i - half):hi - (i - half)]
    muted = V.front_mute(g, offsets, DT, v_direct, pad=0.05, taper=0.04)
    assert np.abs(muted).max() < 0.05 * np.abs(g).max()
    assert np.abs(g).max() > 0.9                    # it really was there


def test_front_mute_keeps_what_is_below_the_cut():
    g, offsets = synthetic_cmp(reflectors=((1.40, 3100.0),), noise=0.0)
    muted = V.front_mute(g, offsets, DT, 1500.0, pad=0.05)
    assert np.allclose(muted[:, 900:], g[:, 900:], atol=1e-5)


def test_front_mute_taper_is_gradual():
    g = np.ones((4, 400), np.float32)
    offsets = np.full(4, 400.0, np.float32)
    soft = V.front_mute(g, offsets, DT, 2000.0, taper=0.08)
    hard = V.front_mute(g, offsets, DT, 2000.0, taper=0.0)
    edge = soft[0]
    assert np.unique(hard[0]).tolist() == [0.0, 1.0]        # a step
    assert ((edge > 0.01) & (edge < 0.99)).sum() > 10       # a ramp
    assert np.all(np.diff(edge) >= -1e-6)                   # monotone


def test_front_mute_validates():
    g, offsets = synthetic_cmp(fold=6, noise=0.0)
    with pytest.raises(ValueError, match="positive"):
        V.front_mute(g, offsets, DT, 0.0)
    with pytest.raises(ValueError, match="offsets"):
        V.front_mute(g, offsets[:2], DT, 1500.0)
