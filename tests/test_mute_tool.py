"""The mute tool on the gather panel (viewer.MuteTool)."""
import numpy as np
import panel as pn
import pytest

import gathervis as gv
from gathervis import mute as M
from gathervis import sortkeys as S
from gathervis.demo import synthetic_line
from gathervis.viewer import Workspace


@pytest.fixture
def ws():
    return Workspace(synthetic_line(ns=12, nr=48, nt=400))


@pytest.fixture
def mt(ws):
    tool = ws.browser.mt
    tool.w_cut_v.value, tool.w_cut_pad.value = 1500.0, 0.05
    tool.seed_linear()
    return tool


def _nudge_time(tool, kind, dt):
    d = dict(tool.cds[kind].data)
    d["y"] = [t + dt for t in d["y"]]
    tool.cds[kind].data = d


# ---------------- what domain the lines live in ----------------
def test_lines_are_stored_against_offset_when_there_is_geometry(mt):
    assert mt.mutes.domain == "offset"
    assert mt._x.max() > 100                       # metres, not column numbers


def test_without_geometry_the_lines_fall_back_to_trace_index():
    bare = gv.from_array(np.zeros((4, 20, 100), "f4"), dt=0.002)
    tool = Workspace(bare).browser.mt
    assert tool.mutes.domain == "trace"
    assert np.array_equal(tool._x, np.arange(20))


def test_seeding_builds_a_straight_cut_across_the_spread(mt):
    line = mt.mutes.line(None, M.TOP)
    assert len(line) == 2
    assert line[0, 1] == pytest.approx(mt._x.min() / 1500.0 + 0.05)
    assert line[-1, 1] == pytest.approx(mt._x.max() / 1500.0 + 0.05)


def test_the_cut_is_drawn_across_every_trace_not_just_the_nodes(mt):
    assert len(mt._cds_line[M.TOP].data["x"]) == mt._x.size
    assert len(mt.cds[M.TOP].data["x"]) == 2       # two handles


# ---------------- carrying across gathers ----------------
def test_the_line_survives_switching_gathers(ws, mt):
    before = list(mt._cds_line[M.TOP].data["y"])
    ws.browser.set_shot(7)
    assert len(mt._cds_line[M.TOP].data["y"]) == mt._x.size
    assert not mt.mutes.is_override(7)             # still on the default
    ws.browser.set_shot(0)
    assert np.allclose(mt._cds_line[M.TOP].data["y"], before)


def test_the_line_survives_re_sorting_the_panel(ws, mt):
    line = mt.mutes.line(0, M.TOP).copy()
    ws.browser.w_sort.value = S.OFFSET
    assert np.allclose(mt.mutes.line(0, M.TOP), line)
    drawn = np.asarray(mt._cds_line[M.TOP].data["y"])
    assert np.all(np.diff(drawn) >= -1e-9)         # sorted by offset: monotone
    ws.browser.w_sort.value = S.AS_RECORDED
    assert np.allclose(mt.mutes.line(0, M.TOP), line)


def test_a_time_only_edit_does_not_move_the_line_in_offset(ws, mt):
    """The trap: a node snapped to the nearest column of *this* gather must
    not write that column's offset back when only its time was dragged."""
    ws.browser.set_shot(7)
    x_before = mt.mutes.line(7, M.TOP)[:, 0].copy()
    mt.w_scope.value = "this gather"
    _nudge_time(mt, M.TOP, 0.12)
    after = mt.mutes.line(7, M.TOP)
    assert np.allclose(after[:, 0], x_before)      # offsets untouched
    assert np.allclose(after[:, 1], mt.mutes.line(3, M.TOP)[:, 1] + 0.12)


def test_dragging_a_node_sideways_does_move_it(ws, mt):
    ws.browser.set_shot(7)
    mt.w_scope.value = "this gather"
    d = dict(mt.cds[M.TOP].data)
    d["x"] = [d["x"][0], 30.0]
    mt.cds[M.TOP].data = d
    assert mt.mutes.line(7, M.TOP)[:, 0].max() == pytest.approx(mt._x[30])


def test_walking_the_shots_leaves_the_lines_alone(ws, mt):
    ws.browser.set_shot(7)
    mt.w_scope.value = "this gather"
    _nudge_time(mt, M.TOP, 0.1)
    expected = mt.mutes.line(7, M.TOP).copy()
    for shot in (0, 5, 11, 3, 7):
        ws.browser.set_shot(shot)
    assert np.allclose(mt.mutes.line(7, M.TOP), expected)
    assert mt.mutes.overrides() == [7]


# ---------------- default vs this gather ----------------
def test_editing_with_the_default_scope_changes_every_gather(ws, mt):
    ws.browser.set_shot(4)
    assert mt.w_scope.value == "all gathers"
    _nudge_time(mt, M.TOP, 0.2)
    assert mt.mutes.overrides() == []
    assert np.allclose(mt.mutes.line(9, M.TOP), mt.mutes.line(4, M.TOP))


def test_editing_with_this_gather_scope_makes_an_exception(ws, mt):
    ws.browser.set_shot(4)
    default = mt.mutes.line(None, M.TOP).copy()
    mt.w_scope.value = "this gather"
    _nudge_time(mt, M.TOP, 0.2)
    assert mt.mutes.is_override(4)
    assert np.allclose(mt.mutes.line(9, M.TOP), default)   # nobody else moved


def test_revert_puts_a_gather_back_on_the_default(ws, mt):
    ws.browser.set_shot(4)
    default = mt.mutes.line(None, M.TOP).copy()
    mt.w_scope.value = "this gather"
    _nudge_time(mt, M.TOP, 0.2)
    mt.revert()
    assert not mt.mutes.is_override(4)
    assert np.allclose(mt.mutes.line(4, M.TOP), default)


def test_promote_pushes_this_gathers_line_to_everyone(ws, mt):
    ws.browser.set_shot(4)
    mt.w_scope.value = "this gather"
    _nudge_time(mt, M.TOP, 0.2)
    tuned = mt.mutes.line(4, M.TOP).copy()
    mt.promote()
    assert np.allclose(mt.mutes.line(9, M.TOP), tuned)
    assert mt.mutes.overrides() == []
    assert mt.w_scope.value == "all gathers"       # the switch follows


def test_the_status_line_says_which_line_is_in_force(ws, mt):
    ws.browser.set_shot(4)
    assert "default" in mt.status.object
    mt.w_scope.value = "this gather"
    _nudge_time(mt, M.TOP, 0.2)
    assert "its own line" in mt.status.object
    assert "1 gather" in mt.status.object
    assert "offset" in mt.status.object


# ---------------- applying ----------------
def test_apply_mutes_the_displayed_gather(ws, mt):
    clean = np.asarray(ws.browser.pane._last[0]).copy()
    mt.w_apply.value = True
    muted = np.asarray(ws.browser.pane._last[0])
    assert np.count_nonzero(muted == 0) > np.count_nonzero(clean == 0)
    mt.w_apply.value = False
    assert np.allclose(np.asarray(ws.browser.pane._last[0]), clean)


def test_apply_cuts_each_trace_at_its_own_offset(ws, mt):
    mt.w_taper.value = 0.0
    mt.w_apply.value = True
    ws.browser.w_sort.value = S.OFFSET
    arr = np.asarray(ws.browser.pane._last[0])
    onset = np.array([np.flatnonzero(r)[0] if np.any(r) else len(r) for r in arr])
    assert np.all(np.diff(onset) >= 0)             # later cut at longer offset


def test_the_bottom_line_cuts_the_other_way(ws, mt):
    mt.w_taper.value = 0.0
    mt.mutes.set(None, M.TOP, [])
    mt.mutes.set(None, M.BOTTOM, [[0.0, 0.3], [2000.0, 0.3]])
    mt._load()
    mt.w_apply.value = True
    arr = np.asarray(ws.browser.pane._last[0])
    cut = int(0.3 / ws.browser.g.dt)
    assert np.any(arr[:, cut - 5] != 0)
    assert np.all(arr[:, cut + 2:] == 0)


def test_an_edit_redraws_the_muted_display(ws, mt):
    mt.w_apply.value = True
    before = np.asarray(ws.browser.pane._last[0]).copy()
    _nudge_time(mt, M.TOP, 0.2)
    assert not np.allclose(before, np.asarray(ws.browser.pane._last[0]))


# ---------------- files ----------------
def test_export_import_round_trip(ws, mt):
    ws.browser.set_shot(7)
    mt.w_scope.value = "this gather"
    _nudge_time(mt, M.TOP, 0.15)
    text = mt._export().read()
    expected = {k: mt.mutes.line(7, k).copy() for k in M.KINDS}

    mt.mutes = M.Mutes("offset")                   # wipe it
    mt._load()
    assert mt.mutes.line(7, M.TOP).shape == (0, 2)

    mt._import(type("E", (), {"new": text.encode()}))
    assert mt.mutes.overrides() == [7]
    for kind in M.KINDS:
        assert np.allclose(mt.mutes.line(7, kind), expected[kind])


def test_a_bad_file_is_reported_not_raised(mt):
    mt._import(type("E", (), {"new": b'{"nope": 1}'}))
    assert "could not read" in mt.status.object


def test_clear_removes_both_lines(mt):
    mt.mutes.set(None, M.BOTTOM, [[0.0, 1.0], [2000.0, 1.0]])
    mt._load()
    mt.clear()
    assert all(mt.mutes.line(0, k).shape == (0, 2) for k in M.KINDS)
    assert mt._cds_line[M.TOP].data["x"] == []


# ---------------- it is a tool, not a tab ----------------
def test_the_mute_card_sits_with_the_other_gather_tools(ws):
    titles = {c.title for c in pn.Column(*ws.sidebar()).select(pn.Card)}
    assert "Mute" in titles
    assert "Mute" not in list(ws.tabs._names)
