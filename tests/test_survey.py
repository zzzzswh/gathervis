"""CMP binning, fold, and the Fold tab (gathervis.survey + viewer.FoldMap)."""
import numpy as np
import pytest

import gathervis as gv
from gathervis import survey as SV
from gathervis.demo import synthetic_line, synthetic_patch
from gathervis.viewer import FoldMap, Workspace


@pytest.fixture(scope="module")
def patch():
    return synthetic_patch(ns=8, nly=6, nlx=24, nt=40)


# ---------------- midpoints ----------------
def test_midpoint_is_halfway():
    g = gv.from_array(np.zeros((2, 3, 10), "f4"),
                      src=[[0.0, 0.0], [100.0, 0.0]],
                      rec=[[0.0, 0.0], [50.0, 0.0], [100.0, 200.0]])
    assert np.allclose(SV.midpoints(g.geometry, 0),
                       [[0, 0], [25, 0], [50, 100]])
    assert np.allclose(SV.midpoints(g.geometry, 1),
                       [[50, 0], [75, 0], [100, 100]])
    # the whole survey is every shot's midpoints, shot-major
    every = SV.midpoints(g.geometry)
    assert every.shape == (6, 2)
    assert np.allclose(every[:3], SV.midpoints(g.geometry, 0))


def test_midpoints_handle_per_shot_spreads():
    rec = np.zeros((2, 3, 2))
    rec[0] = [[0, 0], [10, 0], [20, 0]]
    rec[1] = [[0, 50], [10, 50], [20, 50]]      # the spread moved
    g = gv.from_array(np.zeros((2, 3, 10), "f4"),
                      src=[[0.0, 0.0], [0.0, 50.0]], rec=rec)
    assert not g.geometry.shared
    assert np.allclose(SV.midpoints(g.geometry, 1), [[0, 50], [5, 50], [10, 50]])


# ---------------- bin size ----------------
def test_default_bin_is_half_the_station_spacing(patch):
    # synthetic_patch: 25 m stations, 200 m line spacing
    assert SV.default_bin(patch.geometry) == (12.5, 100.0)


def test_default_bin_survives_degenerate_layouts():
    # a single straight line has no cross-line spacing: borrow the in-line one
    line = synthetic_line(ns=3, nr=8, nt=20, d_rec=40.0)
    assert SV.default_bin(line.geometry) == (20.0, 20.0)
    # one receiver, no spacing anywhere: fall back to a unit bin
    g = gv.from_array(np.zeros((1, 1, 10), "f4"), src=[[0.0, 0.0]],
                      rec=[[5.0, 5.0]])
    assert SV.default_bin(g.geometry) == (1.0, 1.0)


def test_absurd_bin_sizes_are_refused(patch):
    with pytest.raises(ValueError, match="coarser bin"):
        SV.fold(patch.geometry, (0.001, 0.001))
    with pytest.raises(ValueError, match="must be positive"):
        SV.fold(patch.geometry, (0.0, 10.0))


# ---------------- fold ----------------
def test_fold_conserves_traces(patch):
    geo = patch.geometry
    grid = SV.fold(geo)
    assert grid.counts.sum() == geo.ns * geo.nr   # every trace lands somewhere
    assert grid.counts.dtype == np.int32
    assert grid.nlive == np.count_nonzero(grid.counts)
    assert grid.counts.min() >= 0


def test_fold_is_the_count_per_bin():
    # four traces, two of which share a midpoint -> one bin of fold 2
    g = gv.from_array(np.zeros((2, 2, 10), "f4"),
                      src=[[0.0, 0.0], [20.0, 0.0]],
                      rec=[[20.0, 0.0], [40.0, 0.0]])
    # midpoints: (10,0) (20,0) | (20,0) (30,0)
    grid = SV.fold(g.geometry, (10.0, 10.0))
    assert grid.counts.sum() == 4
    assert grid.counts.max() == 2                 # (20, 0) is hit twice


def test_coarser_bins_raise_the_fold(patch):
    fine = SV.fold(patch.geometry, (12.5, 100.0))
    coarse = SV.fold(patch.geometry, (25.0, 100.0))
    assert coarse.counts.sum() == fine.counts.sum()
    assert coarse.counts.max() > fine.counts.max()
    assert coarse.nlive < fine.nlive


def test_grid_geometry_round_trips(patch):
    grid = SV.fold(patch.geometry)
    ny, nx = grid.shape
    x0, y0, w, h = grid.extent
    assert (w, h) == (nx * grid.dx, ny * grid.dy)
    # the centre of bin (3, 2) maps back to bin (3, 2)
    assert grid.bin_of(x0 + 3.5 * grid.dx, y0 + 2.5 * grid.dy) == (3, 2)
    assert grid.bin_of(x0 - 1.0, y0) is None       # outside the grid
    assert grid.bin_of(x0 + w + 1.0, y0) is None


# ---------------- per-bin offset / azimuth ----------------
def test_bin_contents_match_the_fold(patch):
    grid = SV.fold(patch.geometry)
    iy, ix = np.unravel_index(int(grid.counts.argmax()), grid.shape)
    offs, azis, shots = SV.offsets_azimuths_in_bin(patch.geometry, grid, ix, iy)
    assert offs.size == grid.counts[iy, ix]        # exactly that bin's traces
    assert offs.size == azis.size == shots.size
    assert np.all(offs >= 0) and np.all((azis >= 0) & (azis < 360))
    assert set(np.unique(shots)) <= set(range(patch.geometry.ns))


def test_empty_bin_gives_empty_arrays():
    """A survey with a hole in it: the hole's bins report nothing, which is
    what the rose diagram for an unilluminated bin has to draw."""
    g = gv.from_array(np.zeros((2, 2, 10), "f4"),
                      src=[[0.0, 0.0], [0.0, 0.0]],
                      rec=[[0.0, 0.0], [100.0, 0.0]])   # midpoints 0 and 50
    grid = SV.fold(g.geometry, (10.0, 10.0))
    assert grid.counts[0, 1] == 0                       # the gap between them
    offs, azis, shots = SV.offsets_azimuths_in_bin(g.geometry, grid, 1, 0)
    assert offs.size == azis.size == shots.size == 0


# ---------------- the panel ----------------
def test_fold_panel_reports_the_grid(patch):
    f = FoldMap(patch.geometry)
    assert f.grid.shape == SV.fold(patch.geometry).shape
    assert "live" in f.summary.object.lower()
    assert "fold" in f.summary.object.lower()
    img = f.cds.data["image"][0]
    # empty bins go to NaN so they draw as background, not as fold 1
    assert np.isnan(img).sum() == img.size - f.grid.nlive
    assert f.mapper.low == 1 and f.mapper.high == int(f.grid.counts.max())


def test_fold_panel_rebins_and_resets(patch):
    f = FoldMap(patch.geometry)
    fine = f.grid.shape
    f.w_dx.value = 50.0
    assert f.grid.shape[1] < fine[1]              # fewer, wider bins
    assert f.grid.counts.sum() == patch.geometry.ns * patch.geometry.nr
    f.w_reset.clicks += 1
    assert (f.w_dx.value, f.w_dy.value) == SV.default_bin(patch.geometry)
    assert f.grid.shape == fine


def test_fold_panel_reports_a_bad_bin_instead_of_raising(patch):
    f = FoldMap(patch.geometry)
    good = f.grid
    f.w_dx.value = 0.0001
    assert "coarser bin" in f.summary.object
    assert f.grid is good                          # the last good map stays up


def test_tapping_a_bin_selects_it(patch):
    f = FoldMap(patch.geometry)
    picked = []
    f.on_pick = lambda ix, iy: picked.append((ix, iy))

    class _Tap:                                    # bokeh Tap event stand-in
        x = f.grid.x0 + 2.5 * f.grid.dx
        y = f.grid.y0 + 1.5 * f.grid.dy

    f._on_tap(_Tap())
    assert picked == [(2, 1)]
    assert f._cds_pick.data["x"] == [_Tap.x]       # outlined on the map
    _Tap.x = f.grid.x0 - 999.0                     # a tap off the grid
    f._on_tap(_Tap())
    assert picked == [(2, 1)]                      # ... is ignored


def test_the_fold_map_lives_under_the_layout_map(patch):
    """One tab, because a hole in the fold is a gap in the layout above it."""
    ws = Workspace(patch)
    assert "Fold" not in list(ws.tabs._names)          # not a tab of its own
    assert "Geometry" in list(ws.tabs._names)
    assert ws.fold is not None and ws.map is not None
    view = ws.session.view("geometry")
    assert view.fold is ws.fold and view.map is ws.map


def test_no_geometry_tab_at_all_without_geometry():
    bare = gv.from_array(np.zeros((3, 8, 20), "f4"), dt=0.004)
    ws = Workspace(bare)
    assert "Geometry" not in list(ws.tabs._names)
    assert ws.fold is None


def test_bins_are_centred_on_the_midpoints():
    """Regular midpoints land mid-bin, not on the edges between bins."""
    g = gv.from_array(np.zeros((1, 5, 10), "f4"), src=[[0.0, 0.0]],
                      rec=[[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [60.0, 0.0],
                           [80.0, 0.0]])            # midpoints every 10 m
    grid = SV.fold(g.geometry, (10.0, 10.0))
    assert grid.x0 == -5.0 and grid.y0 == -5.0
    assert grid.counts.tolist() == [[1, 1, 1, 1, 1]]


def test_geometry_is_one_map_with_layers(patch):
    ws = Workspace(patch)
    view = ws.session.view("geometry")
    assert ws.map.figure is ws.fold.figure           # one figure, not two
    view.w_layers.value = ["Sources"]
    assert not ws.fold.r_img.visible and not ws.map.r_rec.visible
    assert ws.map.r_src.visible
    view.w_layers.value = ["Fold", "Receivers", "Sources"]
    assert ws.fold.r_img.visible and ws.map.r_rec.visible


def test_a_tap_on_a_source_is_not_a_bin_pick(patch):
    ws = Workspace(patch)
    x, y = patch.geometry.src[0, :2]
    assert ws.map.near_source(float(x), float(y))
    picked = []
    ws.fold.on_pick = lambda ix, iy: picked.append((ix, iy))

    class _Tap:
        pass
    _Tap.x, _Tap.y = float(x), float(y)
    ws.fold._on_tap(_Tap)
    assert picked == []


def _two_d_lines(nlines=3, shots=8, nrec=30, nt=20):
    """Parallel 2-D lines: each line's shots record only that line."""
    src, rec = [], []
    for l in range(nlines):
        spread = np.stack([np.arange(nrec) * 10.0, np.full(nrec, 100.0 * l)], 1)
        for k in range(shots):
            src.append((15.0 + 35.0 * k, 100.0 * l))
            rec.append(spread)
    data = np.zeros((nlines * shots, nrec, nt), "f4")
    return gv.from_array(data, src=np.array(src), rec=np.array(rec), dt=0.004)


def test_unique_receivers_of_a_per_line_survey():
    g = _two_d_lines()
    assert not g.geometry.shared and g.geometry.rec.shape == (24, 30, 3)
    u = g.geometry.unique_receivers()
    assert u.shape == (90, 3)                        # 3 lines x 30, once each
    assert SV.default_bin(g.geometry) == (5.0, 50.0)


def test_the_map_draws_each_receiver_once():
    ws = Workspace(_two_d_lines())
    xs = ws.map.r_rec.data_source.data["x"]
    assert len(xs) == 90                              # not 24 shots x 30
    ws.browser.set_shot(20)                           # a shot on line 2
    assert set(ws.map._cds_arec.data["y"]) == {200.0}
