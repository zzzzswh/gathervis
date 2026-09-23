"""Hand-picked velocity analysis on one CMP gather (viewer.VelocityPanel)."""
import numpy as np
import pytest

import gathervis as gv
from gathervis import cmp, survey
from gathervis.demo import synthetic_line
from gathervis.viewer import VelocityPanel, velocity_analysis

LAYERS = ((300.0, 1800.0), (700.0, 2300.0), (1150.0, 2900.0))
TRUTH = [(2 * d / v, v) for d, v in LAYERS]


@pytest.fixture(scope="module")
def line():
    return synthetic_line(ns=24, nr=64, nt=600, layers=LAYERS, noise=0.04)


@pytest.fixture(scope="module")
def cg(line):
    grid = survey.fold(line.geometry)
    return cmp.gather_in_bin(line, grid, *cmp.best_bin(grid))


@pytest.fixture
def vp(cg, line):
    return VelocityPanel(cg.data, cg.offsets, line.dt,
                         label=f"bin {cg.ix},{cg.iy}", x=cg.x, y=cg.y)


def _pick_the_peaks(vp, dt):
    """What a person does: read the brightest velocity off each event."""
    vs = [float(vp._vgrid[vp._spec[:, int(round(t / dt))].argmax()])
          for t, _ in TRUTH]
    vp.cds.data = dict(x=vs, y=[t for t, _ in TRUTH])
    return vs


# ---------------- the input is a gather that was handed in ----------------
def test_it_takes_a_plain_array_with_offsets(line):
    data = np.zeros((8, 100), np.float32)
    vp = VelocityPanel(data, np.arange(8) * 100.0, line.dt)
    assert vp._spec is not None


def test_a_cmp_gather_supplies_its_own_offsets_and_location(cg, line):
    app = velocity_analysis(cg, dt=line.dt)
    assert app is not None                        # a Panel app, not served


def test_a_plain_array_needs_offsets_and_dt():
    with pytest.raises(ValueError, match="offsets"):
        velocity_analysis(np.zeros((4, 50), "f4"), dt=0.002)
    with pytest.raises(ValueError, match="dt"):
        velocity_analysis(np.zeros((4, 50), "f4"), offsets=np.arange(4.0))


def test_it_rejects_a_gather_that_is_not_2d(line):
    with pytest.raises(ValueError, match="2-D"):
        VelocityPanel(np.zeros((2, 4, 50), "f4"), np.arange(4.0), line.dt)
    with pytest.raises(ValueError, match="offsets has"):
        VelocityPanel(np.zeros((4, 50), "f4"), np.arange(9.0), line.dt)


# ---------------- the spectrum ----------------
def test_the_spectrum_peaks_at_the_true_velocities(vp, line):
    for t0, v_true in TRUTH:
        column = vp._spec[:, int(round(t0 / line.dt))]
        assert abs(vp._vgrid[column.argmax()] - v_true) / v_true < 0.06


def test_changing_a_spectrum_control_recomputes(vp):
    vp.w_nv.value = 40
    assert vp._spec.shape[0] == 40


def test_a_bad_velocity_range_is_reported_not_raised(vp):
    vp.w_vmax.value = 100.0
    assert "v max" in vp.note.object


def test_the_front_mute_changes_the_gather_it_analyses(vp):
    vp.w_mute_on.value = True
    muted = np.count_nonzero(vp._prepared())
    vp.w_mute_on.value = False
    assert np.count_nonzero(vp._prepared()) > muted


# ---------------- picking, by hand ----------------
def test_with_no_picks_the_curve_is_flat_not_empty(vp):
    v_t = vp.velocity_function()
    assert v_t.shape == vp.times.shape
    assert len(np.unique(v_t)) == 1


def test_picks_interpolate_and_hold_flat_beyond_the_ends(vp):
    vp.cds.data = dict(x=[1800.0, 2400.0], y=[0.30, 0.90])
    v_t = vp.velocity_function()
    assert float(np.interp(0.60, vp.times, v_t)) == pytest.approx(2100.0, abs=30)
    assert v_t[0] == pytest.approx(1800.0, abs=5)       # held, not extrapolated
    assert v_t[-1] == pytest.approx(2400.0, abs=5)


def test_moving_a_pick_redraws_the_moveout_panel(vp):
    vp.cds.data = dict(x=[2000.0], y=[0.5])
    first = np.asarray(vp.pane_nmo._last[0]).copy()
    vp.cds.data = dict(x=[3000.0], y=[0.5])
    assert not np.allclose(first, np.asarray(vp.pane_nmo._last[0]))


def test_the_pick_flattens_the_events_and_the_stack_agrees(vp, line):
    """The loop the panel exists for: pick the peaks, the stack lands on t0."""
    _pick_the_peaks(vp, line.dt)
    stacked = np.asarray(vp._cds_stack.data["x"])
    assert stacked.size == vp.times.size
    for t0, _ in TRUTH:
        gate = slice(int((t0 - 0.03) / line.dt), int((t0 + 0.03) / line.dt))
        assert np.abs(stacked[gate]).max() > 0.4 * np.abs(stacked).max()


def test_a_wrong_pick_does_not_flatten_them(vp, line):
    """The stack is drawn normalised, so its peak is always 1 -- what tells a
    good pick from a bad one is how much of the energy lands on the events."""
    def concentration():
        st = np.abs(np.asarray(vp._cds_stack.data["x"]))
        gates = np.zeros(st.size, bool)
        for t0, _ in TRUTH:
            gates[int((t0 - 0.03) / line.dt):int((t0 + 0.03) / line.dt)] = True
        return st[gates].sum() / (st.sum() + 1e-12)

    _pick_the_peaks(vp, line.dt)
    good = concentration()
    vp.cds.data = dict(x=[1400.0, 1400.0], y=[0.1, 1.0])     # far too slow
    assert concentration() < good


def test_clear_removes_the_picks(vp):
    vp.cds.data = dict(x=[2000.0], y=[0.5])
    vp.clear()
    assert vp.picks()[0].size == 0
    assert "tap the spectrum" in vp.note.object


def test_picks_come_back_in_time_order(vp):
    vp.cds.data = dict(x=[2400.0, 1800.0], y=[0.9, 0.3])
    ts, vs = vp.picks()
    assert list(ts) == [0.3, 0.9] and list(vs) == [1800.0, 2400.0]


def test_there_is_no_automatic_picker(vp):
    """Deliberate: the panel shows the answer, it does not decide it."""
    assert not [n for n in dir(vp) if "auto" in n.lower()]


# ---------------- the velocity file ----------------
def test_the_velocity_file_carries_the_picks_and_the_location(vp, line):
    _pick_the_peaks(vp, line.dt)
    text = vp._export().read()
    assert "time_s,velocity" in text
    assert "# x: " in text and "# y: " in text and "bin " in text
    rows = [r for r in text.splitlines() if r and not r.startswith("#")][1:]
    assert len(rows) == len(TRUTH)


def test_a_velocity_file_reads_back_in(vp, line):
    want = _pick_the_peaks(vp, line.dt)
    text = vp._export().read()
    vp.clear()
    vp._import(type("E", (), {"new": text.encode()}))
    ts, vs = vp.picks()
    assert np.allclose(vs, want, atol=0.01)
    assert np.allclose(ts, [t for t, _ in TRUTH], atol=1e-5)


def test_whitespace_separated_files_are_accepted_too(vp):
    vp._import(type("E", (), {"new": b"0.30 1800\n0.90 2400\n"}))
    assert vp.picks()[1].tolist() == [1800.0, 2400.0]


def test_a_file_with_nothing_usable_is_reported(vp):
    vp._import(type("E", (), {"new": b"# just a comment\n"}))
    assert "no time,velocity rows" in vp.note.object


# ---------------- layout ----------------
def test_the_panel_offers_its_three_views_and_its_cards(vp):
    import panel as pn
    titles = {c.title for c in pn.Column(vp.panel()).select(pn.Card)}
    assert {"Front mute", "Velocity spectrum", "Velocity picks"} <= titles
    assert isinstance(vp.panel(), pn.viewable.Viewable)
