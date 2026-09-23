# TODO

Things agreed on but not built yet. Each entry says what it is, and — where
it matters — what it deliberately is *not*.

---

## Scope: what this project does and does not do

gathervis draws data and lets a person judge it by hand. The judgement is
the point; the tool only has to make the judgement possible — show the
gather, let someone draw on it, keep what they drew when they move to the
next gather, save it to a file.

Applying what was drawn counts as viewing. You cannot tell whether a mute
line is in the right place without seeing the gather with the mute on, so
the muted display belongs here.

Making the judgement *for* the person does not. Automatic picking is a
parameter-tuning problem wearing an algorithm's clothes: when the parameters
are wrong it produces a confident wrong answer, and a viewer should not be
in the business of producing those. The same goes for demultiple, denoise,
migration and anything else that belongs in a processing system.

---

---

## Spectrum performance

`velocity_spectrum` computes semblance at every time sample, which costs
about 3.9 s for a 480-trace, 3000-sample gather over 100 velocities.
Processing software computes a spectrum on a coarse time grid instead —
every 20–50 ms — and interpolates. A 4x decimation in time buys a 4x
speedup for no visible loss, but changes the function's return signature (it
would have to return its own time axis), so it is worth doing in one go with
the velocity UI above rather than on its own.

---

## Refactor: sessions and views

**Done, with one seam left.** `gathervis/state.py` holds `Display` (the
shared chain settings, announcing which *stage* each change dirties),
`Chain` (raw -> filtered -> gained -> clim, cached per stage), `Cursor` (the
shared selection), `DisplayState` (a dict view onto `Display`, so the old
`state` dict kept working while everything moved) and `Session`.

Views are a list, not a branch: each declares `applies(g)` for itself and
the workspace asks all of them, so adding one is adding a class to `VIEWS`.
Selection goes through the cursor, so no view holds a reference to another.
The explicit layer falls out of that — `gv.session(data)`, `gv.gather()`,
`gv.geometry()`, `gv.fold()`, `gv.shot_volume()`, `gv.slices()` — and
`gv.show()` is the same thing with the list filled in.

Two notes for whoever comes next:

* **The seam.** `ShotVolumeView.refresh` and `SlicesView.refresh` call a
  hook the workspace installs, because the redraw logic they need still
  lives there: it depends on the "volume too large to process" note, which
  is a sidebar widget, and on `PROC_MAX_BYTES`. Moving the note into the
  view would move it out of the sidebar, which is a layout decision, not a
  refactoring one. Everything else about those views is in the view.
* **The state dict is now an adapter, not the truth.** `ws.state["gain"]`
  still reads and writes, but it sets a property on `Display` underneath.
  Callers can be moved onto `session.display` one at a time; nothing breaks
  until the last one is gone.

Still open: draggable / rearrangeable tabs, now that tabs are a list. The
order is settled at construction (`gv.session(views=[...])`); rearranging in
the browser needs a drag library, which is the first external JS this would
pull in and so worth deciding on rather than drifting into.

---

## Not started

* Common-offset and time slicing (plan.md, M4).
* Offset-azimuth rose diagrams per bin — `survey.azimuth_sectors` computes
  them already; only the in-app plot is missing. It belongs under the fold
  map on the Geometry tab, for the bin that was tapped.
* `survey.default_bin` derives the bin from the receiver spacing, which is
  the textbook rule and correct when sources and receivers sit on one
  regular grid. When they do not, midpoints land off the bin grid and the
  fold map grows holes inside the live area. Nothing detects that today; a
  warning when the live bins average barely more than one midpoint would
  catch it.
* Synced comparison panels (plan.md, M2).

---

## Measured, for reference

On a 1.73 GB file (300 x 480 x 3000 float32, larger than RAM), one
480x3000 shot = 5.8 MB:

| step | cost |
| --- | --- |
| open (lazy memmap) | 0.9 ms |
| read one shot, cold cache | 8-19 ms (685 MB/s) |
| read one shot, page cache | 1 ms |
| decimate + quantize + clim (the wire path) | 2.7 ms |
| bandpass | 13 ms |
| AGC (after the interior shortcut; was 41 ms) | 21 ms |
| whole frame | 47 ms |

`plan.md` assumed the bottleneck order was IO > transfer > render >
arithmetic. It is not: IO and the wire together are under 4 ms, and the
arithmetic is 36 of the 47. Two things followed — the staged cache in
`Chain`, and the AGC rewrite. What that bought:

| the user does | before | after |
| --- | --- | --- |
| change colormap | 1 ms | 1 ms |
| move the clip percentile | 47 ms | 5.6 ms |
| change the AGC window | 47 ms | 32 ms |
| change a filter corner | 47 ms | 45 ms |
| step to the next shot | 47 ms | 42 ms |

Stepping shots is now the thing to attack next if anything: bandpass and
AGC dominate it and neither is cached across shots (correctly — the data
changed). Prefetching the next shot while the user looks at this one is the
obvious move, and costs nothing but a thread.
