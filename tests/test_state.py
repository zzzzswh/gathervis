"""Shared state, staged caching and the view registry (gathervis.state)."""
import numpy as np
import pytest

import gathervis as gv
from gathervis.demo import synthetic_line
from gathervis.state import (DATA, FILTER, GAIN, SCALE, STYLE, Chain,
                               Cursor, Display, Session, View)


@pytest.fixture
def display():
    return Display()


@pytest.fixture
def gather():
    return synthetic_line(ns=4, nr=32, nt=400).shot(0)


# ---------------- what a change invalidates ----------------
def test_each_setting_announces_its_own_stage(display):
    seen = []
    display.on_change(seen.append)
    display.filter = dict(f1=5, f2=10, f3=60, f4=80)
    display.gain = ("agc", 0.5)
    display.flip = True
    display.perc = 95.0
    display.cmap = "seismic"
    display.mode = "wiggle"
    assert seen == [FILTER, GAIN, GAIN, SCALE, STYLE, STYLE]


def test_setting_a_value_to_what_it_already_is_announces_nothing(display):
    display.perc = 95.0
    seen = []
    display.on_change(seen.append)
    display.perc = 95.0
    display.cmap = "gray"
    display.gain = None
    assert seen == []


def test_a_listener_can_ignore_stages_it_handles_itself(display):
    """A view that owns its colormap should not be woken by one."""
    loud, quiet = [], []
    display.on_change(loud.append, stages=STYLE)
    display.on_change(quiet.append, stages=SCALE)
    display.cmap = "seismic"
    display.perc = 90.0
    assert loud == [STYLE, SCALE]
    assert quiet == [SCALE]


def test_polarity_rides_with_gain(display):
    """Same stage: both are applied after the filter, so both invalidate it."""
    seen = []
    display.on_change(seen.append)
    display.flip = True
    assert seen == [GAIN]


# ---------------- the cache ----------------
def test_nothing_is_recomputed_when_nothing_changed(display, gather):
    chain = Chain(display)
    display.filter = dict(f1=5, f2=10, f3=60, f4=80)
    display.gain = ("agc", 0.5)
    first = chain.process(gather, 0.002, key=0)[0]
    before = dict(chain.stats)
    again = chain.process(gather, 0.002, key=0)[0]
    assert chain.stats["filter"] == before["filter"]
    assert chain.stats["gain"] == before["gain"]
    assert chain.stats["hit"] == before["hit"] + 1
    assert again is first                      # the same array, not an equal one


def test_a_clip_change_reuses_the_filter_and_the_gain(display, gather):
    """The measured win: this used to re-run bandpass and AGC for nothing."""
    chain = Chain(display)
    display.filter = dict(f1=5, f2=10, f3=60, f4=80)
    display.gain = ("agc", 0.5)
    gained = chain.process(gather, 0.002, key=0)[0]
    before = dict(chain.stats)
    display.perc = 90.0
    again, _ = chain.process(gather, 0.002, key=0)
    assert chain.stats["filter"] == before["filter"]     # not recomputed
    assert chain.stats["gain"] == before["gain"]
    assert chain.stats["scale"] == before["scale"] + 1   # only this one
    assert again is gained


def test_a_gain_change_reuses_the_filter(display, gather):
    chain = Chain(display)
    display.filter = dict(f1=5, f2=10, f3=60, f4=80)
    display.gain = ("agc", 0.5)
    chain.process(gather, 0.002, key=0)
    before = dict(chain.stats)
    display.gain = ("agc", 0.2)
    chain.process(gather, 0.002, key=0)
    assert chain.stats["filter"] == before["filter"]
    assert chain.stats["gain"] == before["gain"] + 1


def test_a_filter_change_redoes_everything_after_it(display, gather):
    chain = Chain(display)
    display.filter = dict(f1=5, f2=10, f3=60, f4=80)
    display.gain = ("agc", 0.5)
    chain.process(gather, 0.002, key=0)
    before = dict(chain.stats)
    display.filter = dict(f1=5, f2=10, f3=50, f4=70)
    chain.process(gather, 0.002, key=0)
    assert chain.stats["filter"] == before["filter"] + 1
    assert chain.stats["gain"] == before["gain"] + 1


def test_a_style_change_costs_nothing(display, gather):
    chain = Chain(display)
    display.gain = ("agc", 0.5)
    chain.process(gather, 0.002, key=0)
    before = dict(chain.stats)
    display.cmap = "seismic"
    display.mode = "wiggle"
    chain.process(gather, 0.002, key=0)
    assert {k: chain.stats[k] for k in ("filter", "gain", "scale")} == \
           {k: before[k] for k in ("filter", "gain", "scale")}


def test_a_new_key_invalidates_the_whole_cache(display, gather):
    chain = Chain(display)
    display.gain = ("agc", 0.5)
    chain.process(gather, 0.002, key=0)
    before = dict(chain.stats)
    chain.process(gather * 2, 0.002, key=1)
    assert chain.stats["data"] == before["data"] + 1
    assert chain.stats["gain"] == before["gain"] + 1


def test_the_cache_still_returns_the_right_numbers(display, gather):
    """Caching must not change the answer, only how often it is computed."""
    from gathervis.process import agc, bandpass
    display.filter = dict(f1=5, f2=10, f3=60, f4=80)
    display.gain = ("agc", 0.5)
    display.flip = True
    arr, _ = Chain(display).process(gather, 0.002, key=0)
    expected = -agc(bandpass(gather, 0.002, f1=5, f2=10, f3=60, f4=80),
                    0.002, window=0.5)
    assert np.allclose(arr, expected)


def test_without_gain_the_global_clim_is_used(display, gather):
    chain = Chain(display)
    chain.set_global_clim((-3.0, 3.0))
    assert chain.process(gather, 0.002, key=0)[1] == (-3.0, 3.0)
    display.gain = ("agc", 0.5)                 # now it must be recomputed
    assert chain.process(gather, 0.002, key=0)[1] != (-3.0, 3.0)


# ---------------- the cursor ----------------
def test_the_cursor_is_shared_and_announces_once():
    cur = Cursor()
    seen = []
    cur.on_change(seen.append)
    cur.shot = 4
    cur.shot = 4                                # no change, no announcement
    cur.bin = (3, 7)
    assert seen == ["shot", "bin"]
    assert cur.shot == 4 and cur.bin == (3, 7)


# ---------------- the registry ----------------
def test_a_session_does_not_need_to_know_what_a_view_is():
    class Odd(View):
        name = "odd"

        def attach(self, session):
            self.session = session
            self.woken = []
            session.display.on_change(self.refresh)

        def panel(self):
            return "a panel"

        def refresh(self, stage=DATA):
            self.woken.append(stage)

    s = Session(synthetic_line(ns=3, nr=16, nt=80))
    view = s.add(Odd())
    assert s.views == [view] and s.view("odd") is view
    assert s.view("nope") is None
    s.display.cmap = "seismic"
    assert view.woken == [STYLE]


def test_session_recomputes_the_global_clim_on_a_clip_change():
    s = Session(synthetic_line(ns=3, nr=16, nt=80), perc=98.0)
    before = s.clim
    s.display.perc = 80.0
    assert s.clim != before


def test_a_depth_volume_is_not_symmetric():
    vol = gv.from_array(np.abs(np.random.default_rng(0).standard_normal((4, 4, 20))),
                        axes=("x", "y", "depth"))
    assert Session(vol).display.symmetric is False
    assert Session(synthetic_line(ns=2, nr=8, nt=40)).display.symmetric is True


def test_session_chains_share_the_display_but_not_the_cache():
    s = Session(synthetic_line(ns=3, nr=16, nt=80))
    a, b = s.chain(), s.chain()
    assert a.display is b.display is s.display
    assert a is not b
    s.display.gain = ("agc", 0.5)
    assert a._gained is None and b._gained is None


# ---------------- the adapter that keeps the old dict working ----------------
def test_the_state_dict_is_a_view_onto_the_display():
    """Old callers write a dict; the Display underneath hears about it."""
    from gathervis.state import DisplayState
    d = Display()
    state = DisplayState(d, sample=np.zeros(4), clim=(-1.0, 1.0))
    seen = []
    d.on_change(seen.append)
    state["gain"] = ("agc", 0.5)
    state["perc"] = 90.0
    assert seen == [GAIN, SCALE]
    assert state["gain"] == ("agc", 0.5) == d.gain
    assert state["perc"] == 90.0
    state["clim"] = (-2.0, 2.0)                 # plain storage still works
    assert state["clim"] == (-2.0, 2.0)
    assert set(state) >= {"filter", "gain", "flip", "perc", "sample", "clim"}


def test_a_workspace_state_change_reuses_what_it_can():
    """The measured win, through the real viewer rather than a Chain alone."""
    from gathervis.viewer import Workspace
    ws = Workspace(synthetic_line(ns=4, nr=32, nt=300))
    chain = ws.browser._chain
    assert chain is not None
    ws.state["filter"] = dict(f1=5, f2=10, f3=60, f4=80)
    ws.state["gain"] = ("agc", 0.3)
    ws.browser.redraw()

    before = dict(chain.stats)
    ws.state["perc"] = 90.0
    ws.browser.redraw()
    assert chain.stats["filter"] == before["filter"]     # filter reused
    assert chain.stats["gain"] == before["gain"]         # gain reused
    assert chain.stats["scale"] > before["scale"]        # only the clim redone


def test_the_cache_gives_the_same_picture_as_the_uncached_path():
    from gathervis import sortkeys as S
    from gathervis.viewer import Workspace, _process_gather
    g = synthetic_line(ns=4, nr=32, nt=300)
    ws = Workspace(g)
    for flt in (None, dict(f1=5, f2=10, f3=60, f4=80)):
        for gain in (None, ("agc", 0.3), ("balance",)):
            for flip in (False, True):
                ws.state["filter"], ws.state["gain"] = flt, gain
                ws.state["flip"] = flip
                ws.browser.redraw()
                panel = S.arrange(g, ws.browser.ishot, ws.browser.sort)
                expect, clim = _process_gather(panel.data, g.dt, dict(ws.state))
                assert np.allclose(ws.browser.pane._last[0], expect)
                assert np.allclose(ws.browser.pane._last[1], clim)


def test_polarity_is_applied_after_the_clim_is_measured():
    """On a one-sided volume, measuring the flipped array would hand back
    limits that do not bracket what is drawn."""
    d = Display()
    d.symmetric = False
    chain = Chain(d)
    arr = np.abs(np.random.default_rng(0).standard_normal((8, 200))).astype("f4")
    d.gain = ("balance",)
    plain, clim_plain = chain.process(arr, 0.002, key=0)
    d.flip = True
    flipped, clim_flipped = chain.process(arr, 0.002, key=0)
    assert np.allclose(flipped, -np.asarray(plain))
    assert clim_flipped == clim_plain                     # measured pre-flip


# ---------------- the registry ----------------
def test_views_declare_for_themselves_whether_they_fit():
    from gathervis.viewer import (VIEWS, GatherView, GeometryView,
                                  ShotVolumeView, SlicesView)
    line = synthetic_line(ns=4, nr=16, nt=80)
    bare = gv.from_array(np.zeros((6, 40), "f4"), dt=0.002)
    assert GatherView.applies(line) and not GatherView.applies(bare)
    assert GeometryView.applies(line) and not GeometryView.applies(bare)
    assert SlicesView.applies(line) and not ShotVolumeView.applies(line)
    assert set(VIEWS) == {GatherView, GeometryView, ShotVolumeView, SlicesView}


def test_the_workspace_builds_its_tabs_from_the_registry():
    from gathervis.viewer import Workspace
    ws = Workspace(synthetic_line(ns=4, nr=16, nt=80))
    assert [v.name for v in ws.session.views] == ["gather", "geometry",
                                                  "slices"]
    assert ws._view_tabs == {"gather": 0, "geometry": 1, "slices": 2}
    assert list(ws.tabs._names) == [v.title for v in ws.session.views]


def test_selection_goes_through_the_cursor_not_between_views():
    """The layout map moves the shot browser without either knowing the other."""
    from gathervis.viewer import Workspace
    ws = Workspace(synthetic_line(ns=8, nr=16, nt=80))
    ws.session.view("geometry")._picked(5)
    assert ws.session.cursor.shot == 5
    assert ws.browser.ishot == 5                      # heard it via the cursor
    assert ws.tabs._names[ws.tabs.active] == "Shot gathers"   # and focus()
    ws.browser.set_shot(2)                            # and the other way
    assert ws.session.cursor.shot == 2


def test_focus_by_name_does_nothing_when_nobody_is_presenting():
    s = Session(synthetic_line(ns=2, nr=8, nt=40))
    s.focus("gather")                                 # no hook registered
    seen = []
    s.on_focus(seen.append)
    s.focus("fold")
    assert seen == ["fold"]


# ---------------- the explicit layer ----------------
def test_a_composed_session_shows_only_what_was_added():
    g = synthetic_line(ns=4, nr=16, nt=80)
    comp = gv.session(g)
    comp.add(gv.gather())
    comp.add(gv.geometry())
    assert list(comp.build().tabs._names) == ["Shot gathers", "Geometry"]


def test_the_view_order_is_the_order_they_were_given():
    g = synthetic_line(ns=4, nr=16, nt=80)
    assert list(gv.session(g, views=[gv.geometry(), gv.gather()])
                .build().tabs._names) == ["Geometry", "Shot gathers"]


def test_asking_for_a_view_the_data_cannot_support_says_so():
    bare = gv.from_array(np.zeros((6, 40), "f4"), dt=0.002)
    with pytest.raises(ValueError, match="does not have"):
        gv.session(bare, views=[gv.geometry()]).build()


def test_show_and_session_agree_on_the_automatic_view_list():
    from gathervis.viewer import VIEWS, Workspace
    g = synthetic_line(ns=4, nr=16, nt=80)
    auto = Workspace(g)
    same = gv.session(g, views=[c() for c in VIEWS if c.applies(g)]).build()
    assert list(auto.tabs._names) == list(same.tabs._names)


def test_views_cannot_be_added_after_the_viewer_is_built():
    comp = gv.session(synthetic_line(ns=4, nr=16, nt=80), views=[gv.gather()])
    comp.build()
    with pytest.raises(RuntimeError, match="before"):
        comp.add(gv.geometry())


def test_gv_session_is_the_function_not_the_state_module():
    """They would collide if the state module were still called `session`:
    importing the viewer binds every submodule onto the package, which would
    shadow the lazy attribute -- and only after the first import, so it would
    work once and then stop."""
    import gathervis.viewer                            # binds the submodules
    assert callable(gv.session)
    assert gv.session.__module__ == "gathervis.viewer"
