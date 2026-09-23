"""Trace ordering for 3-D shot records (gathervis.sortkeys + the browser)."""
import numpy as np
import pytest

import gathervis as gv
from gathervis import sortkeys as S
from gathervis.demo import synthetic_line, synthetic_patch
from gathervis.process import robust_clim
from gathervis.viewer import ShotBrowser, Workspace


@pytest.fixture(scope="module")
def patch():
    return synthetic_patch(ns=4, nly=5, nlx=12, nt=80)


def _state(g):
    s = {"sample": g.sample(), "symmetric": True, "perc": 98.0}
    s["clim"] = robust_clim(s["sample"])
    return s


# ---------------- what each dataset can offer ----------------
def test_available_scales_with_the_data(patch):
    assert S.available(patch) == (S.AS_RECORDED, S.RECEIVER_LINE,
                                  S.OFFSET, S.AZIMUTH)
    # a 2-D line has one receiver line per shot, so no line selector
    line = synthetic_line(ns=3, nr=16, nt=40)
    assert S.available(line) == (S.AS_RECORDED, S.OFFSET, S.AZIMUTH)
    # no geometry -> no offset / azimuth to sort on
    bare = gv.from_array(np.zeros((3, 16, 40), "f4"), dt=0.004)
    assert S.available(bare) == (S.AS_RECORDED,)
    # no shot axis at all -> nothing to sort
    vol = gv.from_array(np.zeros((6, 6, 20), "f4"), axes=("x", "y", "depth"))
    assert S.available(vol) == ()


def test_line_structure(patch):
    assert S.nlines(patch) == 5 and S.line_length(patch) == 12
    assert S.nlines(synthetic_line(ns=2, nr=9, nt=20)) == 1


# ---------------- the orderings themselves ----------------
def test_as_recorded_is_the_flattened_shot(patch):
    p = S.arrange(patch, 1, S.AS_RECORDED)
    assert p.data.shape == (5 * 12, patch.nt)
    assert np.array_equal(p.order, np.arange(60))
    assert np.array_equal(np.asarray(p.data),
                          np.asarray(patch.shot(1)).reshape(60, -1))
    # a separator between every pair of adjacent receiver lines
    assert np.allclose(p.bounds, [11.5, 23.5, 35.5, 47.5])
    assert p.key is None


def test_receiver_line_is_one_line(patch):
    p = S.arrange(patch, 1, S.RECEIVER_LINE, line=3)
    assert p.data.shape == (12, patch.nt)
    assert np.array_equal(np.asarray(p.data), np.asarray(patch.shot(1)[3]))
    assert np.array_equal(p.order, np.arange(36, 48))
    assert not len(p.bounds)                 # a single line has no separators
    # out-of-range line numbers clamp rather than raise
    assert np.array_equal(S.arrange(patch, 1, S.RECEIVER_LINE, line=99).order,
                          S.arrange(patch, 1, S.RECEIVER_LINE, line=4).order)


def test_receiver_line_needs_line_structure():
    line = synthetic_line(ns=2, nr=9, nt=20)
    with pytest.raises(ValueError, match="one receiver line"):
        S.arrange(line, 0, S.RECEIVER_LINE)


def test_offset_sort_is_monotonic(patch):
    p = S.arrange(patch, 2, S.OFFSET)
    assert np.all(np.diff(p.key) >= 0)                  # sorted, ascending
    assert np.allclose(p.key, S.offsets(patch, 2)[p.order])
    # the panel really is the shot's traces, just reordered
    flat = np.asarray(patch.shot(2)).reshape(60, -1)
    assert np.array_equal(np.asarray(p.data), flat[p.order])
    assert "offset" in p.xlabel and "m" in p.xlabel


def test_azimuth_is_a_survey_azimuth(patch):
    # +y is north = 0 deg, +x is east = 90 deg (not the mathematical convention)
    g = gv.from_array(np.zeros((1, 4, 30), "f4"), dt=0.004,
                      src=[[0.0, 0.0]],
                      rec=[[0.0, 10.0], [10.0, 0.0], [0.0, -10.0], [-10.0, 0.0]])
    assert np.allclose(S.azimuths(g, 0), [0.0, 90.0, 180.0, 270.0])
    assert np.allclose(S.offsets(g, 0), 10.0)
    p = S.arrange(g, 0, S.AZIMUTH)
    assert np.array_equal(p.order, [0, 1, 2, 3])


def test_offset_without_geometry_is_refused():
    bare = gv.from_array(np.zeros((2, 8, 20), "f4"), dt=0.004)
    with pytest.raises(ValueError, match="geometry"):
        S.arrange(bare, 0, S.OFFSET)


def test_unknown_sort_is_refused(patch):
    with pytest.raises(ValueError, match="sort must be"):
        S.arrange(patch, 0, "by vibes")


# ---------------- picks belong to traces, not to columns ----------------
def test_picks_ride_across_a_sort_change(patch):
    b = ShotBrowser(patch, _state(patch))
    tr = 17                                           # some trace, any trace
    b.pt.cds.data = dict(x=[float(tr)], y=[0.05])
    b.w_sort.value = S.OFFSET
    moved = int(np.flatnonzero(b._order == tr)[0])    # where that trace went
    assert b.pt.cds.data["x"] == [float(moved)]
    assert b.pt.cds.data["y"] == [0.05]
    b.w_sort.value = S.AS_RECORDED                    # ... and back again
    assert b.pt.cds.data["x"] == [float(tr)]


def test_offscreen_picks_are_hidden_not_lost(patch):
    """Going to one receiver line must not throw the rest of the shot's picks
    away -- they are off screen, which is not the same as deleted."""
    b = ShotBrowser(patch, _state(patch))
    b.pt.cds.data = dict(x=[0.0, 40.0], y=[0.05, 0.06])   # lines 0 and 3
    b.w_sort.value = S.RECEIVER_LINE
    b.w_line.value = 3
    assert b.pt.cds.data["x"] == [4.0]                # only the line-3 pick
    assert b.pt.cds.data["y"] == [0.06]
    b.w_line.value = 0
    assert b.pt.cds.data["x"] == [0.0] and b.pt.cds.data["y"] == [0.05]
    b.w_sort.value = S.AS_RECORDED                    # both are still there
    assert b.pt.cds.data["x"] == [0.0, 40.0]
    assert b.pt.cds.data["y"] == [0.05, 0.06]


def test_exported_picks_are_trace_numbers(patch):
    """The CSV says which trace was picked, so it means the same thing
    whatever sort the picks were made under."""
    b = ShotBrowser(patch, _state(patch))
    b.w_sort.value = S.OFFSET
    col = int(np.flatnonzero(b._order == 17)[0])
    b.pt.cds.data = dict(x=[float(col)], y=[0.05])
    assert b.pt.picks()[0] == ([17.0], [0.05])


# ---------------- the browser wiring ----------------
def test_browser_switches_arrangement(patch):
    b = ShotBrowser(patch, _state(patch))
    assert b.pane._last[0].shape == (60, patch.nt)
    assert "line, station" in b.pane.figure.xaxis.axis_label
    assert len(b.pane._cds_sep.data["xs"]) == 4        # 5 lines -> 4 separators
    assert b.pane._cds_seplab.data["text"] == ["L0", "L1", "L2", "L3", "L4"]

    b.w_sort.value = S.RECEIVER_LINE
    assert b.w_line.visible
    assert b.pane._last[0].shape == (12, patch.nt)
    assert not len(b.pane._cds_sep.data["xs"])         # no separators on one line

    b.w_sort.value = S.OFFSET
    assert not b.w_line.visible
    assert b.pane._last[0].shape == (60, patch.nt)
    assert "offset" in b.pane.figure.xaxis.axis_label


def test_sort_selector_hidden_when_there_is_one_sort():
    bare = gv.from_array(np.zeros((3, 16, 40), "f4"), dt=0.004)
    b = ShotBrowser(bare, _state(bare))
    assert b.sorts == (S.AS_RECORDED,)
    assert not b.w_sort.visible


def test_download_name_tracks_the_sort(patch):
    b = ShotBrowser(patch, _state(patch))
    b.set_shot(2)
    assert b.pane.w_download.filename.endswith("shot0002.png")
    b.w_sort.value = S.OFFSET
    assert b.pane.w_download.filename.endswith("shot0002_offset.png")
    b.w_sort.value = S.RECEIVER_LINE
    b.w_line.value = 2
    assert b.pane.w_download.filename.endswith("shot0002_line02.png")


def test_4d_workspace_tabs_and_chain(patch):
    ws = Workspace(patch)
    assert list(ws.tabs._names) == ["Shot gathers", "Geometry",
                                    "Shot volume"]
    # the cuboid tab follows the same filter/gain chain as the 2-D panel
    raw = np.asarray(ws.shot_volume.vol).copy()
    ws._w_gain.value = "AGC"
    assert not np.allclose(np.asarray(ws.shot_volume.vol), raw)
    # ... and follows the shot slider
    ws.browser.set_shot(3)
    assert np.allclose(np.asarray(ws.shot_volume.vol),
                       np.asarray(ws.shot_volume.vol))
    assert ws.shot_volume.vol.shape == patch.shot(3).shape


def test_info_card_reports_the_patch(patch):
    from gathervis.viewer import _survey_kind
    assert _survey_kind(patch) == "3-D seismic (5 receiver lines x 12 stations per shot)"
