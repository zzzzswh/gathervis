"""Pulling a CMP gather out of a bin (gathervis.cmp + survey.traces_in_bin)."""
import numpy as np
import pytest

import gathervis as gv
from gathervis import cmp as C
from gathervis import survey as S
from gathervis.demo import synthetic_line, synthetic_patch


@pytest.fixture(scope="module")
def line():
    return synthetic_line(ns=12, nr=32, nt=200)


@pytest.fixture(scope="module")
def grid(line):
    return S.fold(line.geometry)


# ---------------- the index ----------------
def test_traces_in_bin_agrees_with_the_fold_count(line, grid):
    for ix, iy in ((C.best_bin(grid)), (0, 0)):
        shots, recs, offsets = S.traces_in_bin(line.geometry, grid, ix, iy)
        assert shots.size == recs.size == offsets.size == grid.counts[iy, ix]


def test_traces_in_bin_really_points_at_that_bin(line, grid):
    geo = line.geometry
    ix, iy = C.best_bin(grid)
    shots, recs, offsets = S.traces_in_bin(geo, grid, ix, iy)
    mid = 0.5 * (geo.rec[recs][:, :2] + geo.src[shots][:, :2])
    assert np.all((mid[:, 0] >= grid.x0 + ix * grid.dx)
                  & (mid[:, 0] < grid.x0 + (ix + 1) * grid.dx))
    # and the offsets it reports are the real ones
    d = geo.rec[recs][:, :2] - geo.src[shots][:, :2]
    assert np.allclose(offsets, np.hypot(d[:, 0], d[:, 1]))


def test_traces_in_bin_comes_back_offset_sorted(line, grid):
    _, _, offsets = S.traces_in_bin(line.geometry, grid, *C.best_bin(grid))
    assert np.all(np.diff(offsets) >= 0)


def test_traces_in_bin_rejects_a_bin_off_the_grid(line, grid):
    ny, nx = grid.shape
    with pytest.raises(ValueError, match="outside"):
        S.traces_in_bin(line.geometry, grid, nx, 0)
    with pytest.raises(ValueError, match="outside"):
        S.traces_in_bin(line.geometry, grid, 0, -1)


def test_rose_diagram_still_works(line, grid):
    """offsets_azimuths_in_bin was refactored onto the same scan."""
    ix, iy = C.best_bin(grid)
    offs, azis, shots = S.offsets_azimuths_in_bin(line.geometry, grid, ix, iy)
    assert offs.size == azis.size == shots.size == grid.counts[iy, ix]
    assert np.all((azis >= 0) & (azis < 360))
    idx_offs = S.traces_in_bin(line.geometry, grid, ix, iy)[2]
    assert np.allclose(np.sort(offs), np.sort(idx_offs))


# ---------------- the gather ----------------
def test_gather_in_bin_pulls_the_right_traces(line, grid):
    ix, iy = C.best_bin(grid)
    cg = C.gather_in_bin(line, grid, ix, iy)
    assert cg.data.shape == (grid.counts[iy, ix], line.nt)
    assert cg.fold == grid.counts[iy, ix]
    for k in range(cg.fold):                       # every row is that trace
        assert np.allclose(cg.data[k], line.data[cg.shots[k], cg.recs[k]])


def test_gather_in_bin_is_offset_sorted(line, grid):
    cg = C.gather_in_bin(line, grid, *C.best_bin(grid))
    assert np.all(np.diff(cg.offsets) >= 0)


def test_gather_in_bin_reports_the_bin_centre(line, grid):
    ix, iy = C.best_bin(grid)
    cg = C.gather_in_bin(line, grid, ix, iy)
    assert cg.ix == ix and cg.iy == iy
    assert cg.x == pytest.approx(grid.x0 + (ix + 0.5) * grid.dx)
    assert cg.y == pytest.approx(grid.y0 + (iy + 0.5) * grid.dy)
    assert f"({ix}, {iy})" in repr(cg)


def test_an_empty_bin_gives_an_empty_gather_not_an_error(line):
    """Tapping a hole in the fold map is a normal thing to do."""
    grid = S.fold(line.geometry)
    holes = np.argwhere(grid.counts == 0)
    if not holes.size:                             # dense line: make a hole
        pytest.skip("this geometry has no empty bins")
    iy, ix = holes[0]
    cg = C.gather_in_bin(line, grid, int(ix), int(iy))
    assert cg.fold == 0
    assert cg.data.shape == (0, line.nt)
    assert "empty" in repr(cg)


def test_gather_in_bin_works_on_a_3d_patch():
    """A 3-D shot is a patch of lines; the receiver index is the flat one."""
    patch = synthetic_patch(ns=3, nly=4, nlx=10, nt=60)
    grid = S.fold(patch.geometry)
    cg = C.gather_in_bin(patch, grid, *C.best_bin(grid))
    assert cg.data.shape[1] == patch.nt
    flat = patch.data.reshape(patch.data.shape[0], -1, patch.nt)
    for k in range(cg.fold):
        assert np.allclose(cg.data[k], flat[cg.shots[k], cg.recs[k]])


def test_gather_in_bin_needs_geometry():
    grid = S.fold(synthetic_line(ns=4, nr=8, nt=40).geometry)
    bare = gv.from_array(np.zeros((4, 8, 40), "f4"), dt=0.004)
    with pytest.raises(ValueError, match="geometry"):
        C.gather_in_bin(bare, grid, 0, 0)


def test_geometry_implies_a_shot_axis():
    """Why gather_in_bin only has to check for geometry: Gathers enforces it."""
    coords = np.zeros((6, 2))
    with pytest.raises(ValueError, match="no 'shot' axis"):
        gv.from_array(np.zeros((6, 6, 20), "f4"), src=coords, rec=coords,
                      axes=("x", "y", "depth"))


def test_best_bin_is_the_highest_fold_one(grid):
    ix, iy = C.best_bin(grid)
    assert grid.counts[iy, ix] == grid.counts.max()
    empty = S.BinGrid(np.zeros((3, 3), np.int32), 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="no live bins"):
        C.best_bin(empty)


# ---------------- it reads only what it needs ----------------
def test_gather_in_bin_does_not_load_the_whole_file(tmp_path):
    """The point of the survey/cmp split: one bin, not the survey."""
    src = synthetic_line(ns=10, nr=24, nt=120)
    path = tmp_path / "line.npy"
    np.save(path, src.data)
    lazy = gv.from_array(np.load(path, mmap_mode="r"),
                         src=src.geometry.src, rec=src.geometry.rec, dt=src.dt)
    assert isinstance(lazy.data, np.memmap)
    grid = S.fold(lazy.geometry)
    cg = C.gather_in_bin(lazy, grid, *C.best_bin(grid))
    assert cg.fold > 0
    assert not isinstance(cg.data, np.memmap)      # materialised, and only this
    assert np.allclose(cg.data[0], src.data[cg.shots[0], cg.recs[0]])
