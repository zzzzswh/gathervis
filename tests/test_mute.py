"""Top/bottom mute lines and how they carry across gathers (gathervis.mute)."""
import numpy as np
import pytest

from gathervis import mute as M

DT = 0.002
NT = 1000
OFFSETS = np.array([100.0, 300.0, 600.0, 900.0, 1400.0, 2000.0])


@pytest.fixture
def ones():
    return np.ones((OFFSETS.size, NT), np.float32)


@pytest.fixture
def top():
    return M.linear_line(v=1500.0, x_max=2000.0, pad=0.05)


# ---------------- the line itself ----------------
def test_a_line_is_linear_between_nodes_and_flat_beyond():
    line = [[500.0, 0.4], [1500.0, 0.8]]
    assert M.evaluate(line, [1000.0])[0] == pytest.approx(0.6)
    assert M.evaluate(line, [0.0])[0] == pytest.approx(0.4)      # held, not
    assert M.evaluate(line, [9000.0])[0] == pytest.approx(0.8)   # extrapolated


def test_nodes_are_sorted_by_x_however_they_arrive():
    jumbled = [[1500.0, 0.8], [500.0, 0.4], [1000.0, 0.6]]
    assert M.evaluate(jumbled, [750.0])[0] == pytest.approx(0.5)


def test_an_empty_line_is_no_mute_not_a_mute_at_zero(ones):
    assert np.all(np.isnan(M.evaluate([], [0.0, 1.0])))
    assert np.allclose(M.apply(ones, OFFSETS, DT), ones)
    assert np.allclose(M.apply(ones, OFFSETS, DT, top=[], bottom=[]), ones)


def test_linear_line_follows_a_straight_moveout():
    line = M.linear_line(v=1500.0, x_max=3000.0, pad=0.1)
    assert M.evaluate(line, [1500.0])[0] == pytest.approx(1.0 + 0.1)
    with pytest.raises(ValueError, match="positive"):
        M.linear_line(v=0.0, x_max=1000.0)


def test_a_malformed_line_is_rejected():
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        M.evaluate([[1.0, 2.0, 3.0]], [0.0])
    with pytest.raises(ValueError, match="non-finite"):
        M.evaluate([[0.0, np.nan]], [0.0])


# ---------------- applying ----------------
def test_top_mute_cuts_at_the_line(ones, top):
    out = M.apply(ones, OFFSETS, DT, top=top, taper=0.0)
    onset = out.argmax(axis=1) * DT
    assert np.allclose(onset, OFFSETS / 1500.0 + 0.05, atol=DT)
    for k, t_cut in enumerate(OFFSETS / 1500.0 + 0.05):
        assert np.all(out[k, :int(t_cut / DT)] == 0)


def test_bottom_mute_cuts_the_other_way(ones):
    bottom = [[0.0, 1.0], [2000.0, 1.4]]
    out = M.apply(ones, OFFSETS, DT, bottom=bottom, taper=0.0)
    for k, x in enumerate(OFFSETS):
        cut = int(M.evaluate(bottom, [x])[0] / DT)
        assert out[k, cut - 5] != 0 and np.all(out[k, cut + 1:] == 0)


def test_both_lines_leave_a_window(ones, top):
    out = M.apply(ones, OFFSETS, DT, top=top,
                  bottom=[[0.0, 1.2], [2000.0, 1.6]], taper=0.0)
    for k, x in enumerate(OFFSETS):
        live = np.flatnonzero(out[k])
        assert live[0] * DT == pytest.approx(x / 1500.0 + 0.05, abs=2 * DT)
        assert live[-1] * DT == pytest.approx(1.2 + x / 2000.0 * 0.4, abs=2 * DT)


def test_crossed_lines_mute_the_trace_entirely(ones):
    """Bottom above top leaves nothing; that is the honest answer, not an error."""
    out = M.apply(ones, OFFSETS, DT, top=[[0.0, 1.0], [2000.0, 1.0]],
                  bottom=[[0.0, 0.3], [2000.0, 0.3]], taper=0.0)
    assert np.count_nonzero(out) == 0


def test_the_taper_is_a_ramp_not_a_step(ones, top):
    hard = M.apply(ones, OFFSETS, DT, top=top, taper=0.0)
    soft = M.apply(ones, OFFSETS, DT, top=top, taper=0.06)
    assert np.unique(hard[2]).tolist() == [0.0, 1.0]
    partial = (soft[2] > 0.01) & (soft[2] < 0.99)
    assert partial.sum() > 10
    assert np.all(np.diff(soft[2][:500]) >= -1e-6)      # monotone through the cut


def test_apply_validates(ones, top):
    with pytest.raises(ValueError, match="2-D"):
        M.apply(ones[None], OFFSETS, DT, top=top)
    with pytest.raises(ValueError, match="x has"):
        M.apply(ones, OFFSETS[:2], DT, top=top)
    with pytest.raises(ValueError, match="dt must be positive"):
        M.apply(ones, OFFSETS, 0.0, top=top)


# ---------------- default and overrides ----------------
def test_a_gather_without_its_own_line_inherits_the_default(top):
    m = M.Mutes()
    m.set(None, M.TOP, top)
    assert np.allclose(m.line(12, M.TOP), top)
    assert np.allclose(m.line(999, M.TOP), top)
    assert not m.is_override(12)
    assert m.overrides() == []


def test_an_override_shadows_the_default_for_that_gather_only(top):
    m = M.Mutes()
    m.set(None, M.TOP, top)
    own = [[0.0, 0.2], [2000.0, 1.5]]
    m.set(37, M.TOP, own)
    assert np.allclose(m.line(37, M.TOP), own)
    assert np.allclose(m.line(36, M.TOP), top)          # neighbours untouched
    assert m.is_override(37) and m.is_override(37, M.TOP)
    assert not m.is_override(37, M.BOTTOM)              # only the top differs
    assert m.overrides() == [37] and len(m) == 1


def test_an_override_on_one_side_still_inherits_the_other(top):
    m = M.Mutes()
    m.set(None, M.TOP, top)
    m.set(None, M.BOTTOM, [[0.0, 1.5], [2000.0, 1.8]])
    m.set(37, M.TOP, [[0.0, 0.3], [2000.0, 1.0]])
    assert np.allclose(m.line(37, M.BOTTOM), m.line(1, M.BOTTOM))


def test_clearing_an_override_returns_the_gather_to_the_default(top):
    m = M.Mutes()
    m.set(None, M.TOP, top)
    m.set(37, M.TOP, [[0.0, 0.3], [2000.0, 1.0]])
    m.clear(37)
    assert np.allclose(m.line(37, M.TOP), top)
    assert m.overrides() == []


def test_setting_an_empty_line_clears_it(top):
    m = M.Mutes()
    m.set(None, M.TOP, top)
    m.set(37, M.TOP, [])
    assert np.allclose(m.line(37, M.TOP), top)          # back to the default


def test_promote_makes_one_gathers_line_the_default(top):
    """Tuned a mute on one gather and want it everywhere."""
    m = M.Mutes()
    m.set(None, M.TOP, top)
    tuned = [[0.0, 0.12], [2000.0, 1.45]]
    m.set(37, M.TOP, tuned)
    m.promote(37)
    assert np.allclose(m.line(5, M.TOP), tuned)
    assert m.overrides() == []                          # 37 is no longer special


def test_clearing_the_default_removes_the_mute_everywhere(top):
    m = M.Mutes()
    m.set(None, M.TOP, top)
    m.clear(None)
    assert m.line(3, M.TOP).shape == (0, 2)


def test_mutes_validates():
    with pytest.raises(ValueError, match="domain"):
        M.Mutes(domain="depth")
    m = M.Mutes()
    with pytest.raises(ValueError, match="kind"):
        m.set(None, "middle", [[0.0, 1.0]])
    with pytest.raises(ValueError, match="kind"):
        m.line(None, "sideways")


def test_mutes_apply_uses_the_right_line_per_gather(ones, top):
    m = M.Mutes()
    m.set(None, M.TOP, top)
    m.set(37, M.TOP, [[0.0, 0.8], [2000.0, 0.8]])
    default = m.apply(ones, OFFSETS, DT, key=12, taper=0.0)
    special = m.apply(ones, OFFSETS, DT, key=37, taper=0.0)
    assert default.argmax(axis=1)[0] * DT == pytest.approx(0.117, abs=2 * DT)
    assert np.allclose(special.argmax(axis=1) * DT, 0.8, atol=2 * DT)


# ---------------- files ----------------
def test_json_round_trip_keeps_default_and_overrides(top):
    m = M.Mutes(domain="offset")
    m.set(None, M.TOP, top)
    m.set(None, M.BOTTOM, [[0.0, 1.5], [2000.0, 1.9]])
    m.set(37, M.TOP, [[0.0, 0.3], [2000.0, 1.1]])
    back = M.Mutes.from_json(m.to_json())
    assert back.domain == "offset"
    assert back.overrides() == [37]                     # int key, not "37"
    for key in (None, 12, 37):
        for kind in M.KINDS:
            assert np.allclose(back.line(key, kind), m.line(key, kind))


def test_a_trace_domain_file_says_so():
    m = M.Mutes(domain="trace")
    m.set(None, M.TOP, [[0, 0.1], [95, 0.5]])
    assert M.Mutes.from_json(m.to_json()).domain == "trace"


def test_a_foreign_file_is_refused():
    with pytest.raises(ValueError, match="not a gathervis mute file"):
        M.Mutes.from_json('{"picks": []}')


def test_repr_says_what_is_in_there(top):
    m = M.Mutes()
    assert "none" in repr(m)
    m.set(None, M.TOP, top)
    m.set(37, M.TOP, top)
    assert "top" in repr(m) and "1 overrides" in repr(m)


# ---------------- using a saved file from a script ----------------
def test_mute_shot_applies_a_saved_file_without_the_viewer():
    """The other end of the tool: draw in the app, export, use in a script."""
    from gathervis import sortkeys as S
    from gathervis.demo import synthetic_line

    g = synthetic_line(ns=8, nr=40, nt=400)
    m = M.Mutes("offset")
    m.set(None, M.TOP, M.linear_line(v=1500.0, x_max=1200.0, pad=0.05))
    m.set(5, M.TOP, [[0.0, 0.25], [1200.0, 1.05]])          # one exception
    m = M.Mutes.from_json(m.to_json())                      # via the file

    for shot in (0, 5):
        out = M.mute_shot(g, shot, m, taper=0.0)
        assert out.shape == g.shot(shot).shape
        onset = np.array([np.flatnonzero(r)[0] if np.any(r) else -1
                          for r in out]) * g.dt
        expected = M.evaluate(m.line(shot, M.TOP), S.offsets(g, shot))
        assert np.abs(onset - expected).max() <= g.dt       # within a sample


def test_mute_shot_uses_the_gathers_own_line_when_it_has_one():
    from gathervis.demo import synthetic_line
    g = synthetic_line(ns=6, nr=24, nt=300)
    m = M.Mutes("offset")
    m.set(None, M.TOP, [[0.0, 0.1], [1000.0, 0.1]])
    m.set(3, M.TOP, [[0.0, 0.4], [1000.0, 0.4]])
    assert np.count_nonzero(M.mute_shot(g, 3, m, taper=0.0)) < \
           np.count_nonzero(M.mute_shot(g, 2, m, taper=0.0))


def test_mute_shot_says_so_when_it_cannot_compute_offsets():
    import gathervis as gv
    m = M.Mutes("offset")
    m.set(None, M.TOP, [[0.0, 0.1], [1000.0, 0.5]])
    bare = gv.from_array(np.zeros((2, 4, 10), "f4"), dt=0.002)
    with pytest.raises(ValueError, match="no geometry"):
        M.mute_shot(bare, 0, m)


def test_mute_shot_falls_back_to_trace_index_for_a_trace_domain_file():
    import gathervis as gv
    m = M.Mutes("trace")
    m.set(None, M.TOP, [[0, 0.02], [3, 0.02]])
    bare = gv.from_array(np.ones((2, 4, 50), "f4"), dt=0.002)
    out = M.mute_shot(bare, 0, m, taper=0.0)
    assert np.all(out[:, :10] == 0) and np.all(out[:, 12:] != 0)


def test_mute_shot_flattens_a_3d_patch():
    from gathervis.demo import synthetic_patch
    patch = synthetic_patch(ns=3, nly=3, nlx=8, nt=60)
    m = M.Mutes("offset")
    m.set(None, M.TOP, [[0.0, 0.02], [5000.0, 0.06]])
    out = M.mute_shot(patch, 0, m)
    assert out.shape == (3 * 8, 60)
