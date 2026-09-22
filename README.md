<div align="center">

# gathervis

**English** | [简体中文](https://github.com/zzzzswh/gathervis/blob/main/README.zh-CN.md)

**A professional web viewer for pre-stack seismic data.**

*Open pre-stack data sitting on a remote server in your browser, and work it
the way you would in processing software: browse shots, filter, take spectra,
pick.*

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-147%20passing-brightgreen)

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/ui_gather.png" width="820" alt="Shot gathers tab: analysis window, its spectrum with the crosshair on, and the four tool blocks"/>
<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/ui_slice.png" width="820" alt="Volume slices tab: AGC-gained volume as a cuboid stretched 8x along the shot axis"/>

*2-D gather QC and 3-D volume slicing, in one browser tab.*

</div>

> **💬 Feature requests welcome!** If there is anything you wish it did — a
> display mode, a processing step, a file format — **please open an issue and
> tell me. I will implement it.** Thank you very much!
>
> **💬 欢迎提需求！** 如果你希望它能做什么 —— 显示方式、处理步骤、文件格式 ——
> **请务必开 issue 告诉我，我会实现的，非常感谢！**

---

## Why gathervis?

The reason is a practical one: not wanting to draw pre-stack data with
matplotlib one figure at a time. Change shot, redraw. Change a filter
parameter, redraw. Want the spectrum of one time gate and that is another
block of code. The figures come out fine, but the process is broken up, and
every look starts over. gathervis turns it into the kind of viewing you get in
processing software: the data is simply open in a browser, a slider steps
through shots, drawing a box gives you its spectrum, and changing the filter
updates the panel as you type.

* **Instant on huge data.** Files open as lazy memmaps — browsing a shot reads
  only that shot's bytes; slicing a volume reads only that slice. A 100 GB raw
  binary opens as fast as a 1 MB one.
* **Geometry-linked views.** Give it source/receiver coordinates and you get a
  Geometry tab and a Fold tab — tap any source point to jump to its shot
  gather, with the active spread highlighted; CMP fold is computed from
  coordinates alone.
* **QC tools built in.** Wiggle & variable-density display with industry
  colormaps, zero-phase Ormsby filters, AGC / trace balance, draw-a-window
  spectral analysis (amplitude spectra and f-k), and manual / STA-LTA
  first-break picking.
* **cigvis-style 3-D.** Volumes render as a rotatable cuboid with three live
  slice planes (WebGL).
* **Simple.** All it needs is a browser: one port, auto-forwarded by VS Code
  Remote-SSH, rendered inline in Jupyter, and reachable over a bare `ssh -L`.
  Four dependencies, installed with pip.

## Install

```bash
pip install -e .            # numpy + panel + bokeh + plotly
pip install deepwave        # optional, for the modelling examples (pulls torch)
pip install segyio          # optional, for SEG-Y import
```

## Quick start

```python
import gathervis as gv

# -- quick look, no geometry --
gv.show(gather_2d, dt=0.002, port=8080)          # a (trace, time) panel
gv.show(line_3d, dt=0.002, port=8080)            # default (shot, rec, time): browse shots
gv.show(line_3d, view='slices', port=8080)       # same array as a 3-D cuboid w/ slices
gv.show('shots.bin', shape=(60,192,1200),        # raw binary: lazy memmap, opens instantly
        dtype='float32', dt=0.002, port=8080)
gv.show(vel, axes=('x','y','depth'), dt=10.0)    # property volume (velocity QC)

# -- with acquisition geometry --
ds = gv.from_array(data, src=src_xyz, rec=rec_xyz, dt=0.002)
gv.show(ds, port=8080)   # + Geometry tab; tap a source -> its shot gather
```

Or from the shell, no code at all:

```bash
gathervis line2d_data.npy --geom line2d_geom.npz
gathervis shots.bin --shape 60 192 1200 --dt 0.002
gathervis field.sgy                                # SEG-Y: dt + geometry from headers
gathervis vel.npy --axes x y depth --dt 10 --cmap rainbow
```

Examples with synthetic data: `python examples/quickstart.py` serves an
analytic synthetic;
`python examples/deepwave_line2d.py` models a 2-D line with
[deepwave](https://github.com/ar4/deepwave); and
`python examples/deepwave_lines3d.py` shoots **five 2-D lines over a rich 3-D
velocity model** (dipping layers + anticline dome + low-velocity channel lens)
with deepwave's 3-D engine, then opens the result.

## Features

### Shot browsing and geometry

On the **Shot gathers** tab, the shot slider updates the panel while you drag,
**←** / **→** step one shot at a time, and *go to shot* jumps to a shot number.
Four tool blocks sit under the panel — *Window · Spectrum · Event picking ·
FB picking* — holding the controls for analysis windows, spectra, picks and
first-break parameters respectively.

The **Geometry** tab shows the source and receiver layout. Tapping a source
selects that shot and returns to the Shot gathers tab, with the active spread
highlighted on the map.

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/ui_geometry.png" width="820" alt="Geometry tab: five 2-D lines, active shot starred with its spread highlighted"/>

### Trace ordering for 3-D shots

A 3-D shot is a patch of several receiver lines. Processing practice puts the
whole patch on one `(trace, time)` panel and changes the order the traces are
laid out in; the **sort** selector on the Shot gathers tab provides four
orderings.

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/sorts.png" width="820" alt="the same 3-D shot sorted as recorded, by offset and by azimuth"/>

*The same shot in three orderings: as recorded (receiver-line breaks dashed in
orange), by offset, by azimuth.*

| sort | description |
| --- | --- |
| **as recorded** `(line, station)` | Acquisition order, receiver lines end to end, displayed as N nested hyperbolas whose apices step with cross-line distance. Used to check geometry errors, reversed lines, polarity flips and dead channels. Dashed separators mark the line boundaries. |
| **receiver line** | One receiver line at a time, i.e. a 2-D shot record. |
| **offset** | Ordered by source-receiver distance; the patch collapses onto one hyperbola. Used for moveout, mute design and velocity analysis. |
| **azimuth** | Ordered by source-to-receiver azimuth (survey convention, north = 0). Used for azimuthal behaviour and azimuth coverage. |

All four orderings are the same 2-D panel, so analysis windows, spectra,
picking and full-resolution export work identically on them. Picks are stored
against the **trace** rather than the screen column, so they follow a re-sort;
picks on traces outside a single-line view are hidden rather than deleted and
return when the view does.

`offset` and `azimuth` require geometry; `receiver line` requires 4-D
`(shot, recy, recx, time)` data, the shape in which the line structure is
explicit. The selector lists only the orderings the dataset supports, and is
hidden for a 2-D line with no geometry. The **Shot volume** tab renders the
current shot as a cuboid on the same filter/gain chain.

Example: `python examples/quickstart.py patch`.

### CMP fold map

The **Fold** tab bins the midpoint of every source-receiver pair on a
rectangular grid and counts them.

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/fold.png" width="520" alt="CMP fold map of an orthogonal land 3-D, full fold in the middle tapering to the edges"/>

*Fold of an orthogonal land 3-D: 25 m stations on 200 m receiver lines, 50 m
shot points on 200 m shot lines, fixed full spread.*

Bin size defaults to half the receiver station interval in-line and half the
line interval cross-line, the natural CMP sampling; both are editable on the
tab, and *default bin* restores them. Empty bins are drawn in the background
colour rather than at the bottom of the colour scale. The summary line gives
grid size, live bins, maximum and mean fold, and the trace total — which
should equal `sources × receivers` and serves as a geometry check.

The computation uses coordinates only and reads no trace data, so its cost is
independent of data volume and it works on a bare `Geometry`:

```python
from gathervis.survey import fold, default_bin
grid = fold(ds.geometry)                     # or fold(geo, (12.5, 100.0))
grid.counts                                  # (ny, nx) traces per bin
grid.nlive, grid.extent, grid.bin_of(x, y)
```

Tapping a bin selects and outlines it. The offset-azimuth distribution of a
bin is available from the same module; the in-app rose diagram is not yet
implemented:

```python
from gathervis.survey import offsets_azimuths_in_bin, azimuth_sectors
offs, azis, shots = offsets_azimuths_in_bin(ds.geometry, grid, ix, iy)
edges, counts = azimuth_sectors(azis, nsector=24)   # survey convention, N = 0
```

The number of non-empty sectors returned by `azimuth_sectors` is the bin's
azimuth coverage.

### Gestures & controls

| To do this | Do this |
| --- | --- |
| Draw a box window | select the box toolbar tool → **SHIFT+drag** (or click, move, click) |
| Draw a polygon window | select the polygon tool → click each vertex, **double-click or ESC** to finish |
| Move / delete a window | drag it; tap to select, then **BACKSPACE** (or *clear windows*) |
| Add / move / delete picks | point toolbar tool → tap adds, drag moves, tap + **BACKSPACE** deletes (or *clear shot* / *clear all*) |
| Auto first breaks | set STA / LTA / threshold → *auto pick*; refine with *snap* (peak / trough / \|max\|) or by dragging |
| Compare spectrum peaks | crosshair toolbar icon + the freq / dB readout under the panel |
| Compute an f-k spectrum | *f-k spectrum* in the Spectrum block, applied to the first window |
| Reuse a set of windows | *export JSON* / *import windows (.json)* in the Window block |
| Take picks out and back | *export CSV* / import in the picking block, `shot,trace,time_s` |
| Scale one axis | the x-only / y-only wheel-zoom toolbar tools |
| Re-sort a 3-D shot | the *sort* selector (as recorded / receiver line / offset / azimuth) |
| Re-bin the fold map | *bin x* / *bin y* on the Fold tab (*default bin* restores) |
| Select a CMP bin | tap it on the fold map |
| Step through receiver lines | *sort* = receiver line, then the line slider |
| Reshape the 3-D cuboid | one *stretch* slider per axis (1.0 centered, 0.125×–8×) |
| Jump to a shot | drag the slider, type in *go to shot*, or tap a source on the Geometry tab |
| Resize the spectrum panel | *size* (S / M / L) in the Spectrum block |
| Save the gather as an image | the **⤓ full image** button (full resolution) · the toolbar *save* icon (current view, screen resolution) |
| Step through shots | **←** / **→** |
| Share the current view | copy the URL: active tab, shot, sort, receiver line, colormap, clip, polarity, filter and gain (incl. AGC window) are all in the query string |

The same table is available in-app behind the *? gestures* button. The arrow
keys are the only shortcut; every other control is set once and stays on its
widget. Arrow keys are ignored while typing in a field, and
`gv.show(..., keys=False)` disables them.

### 3-D volume slices

Any 3-D array — a whole line, one 3-D shot gather, a velocity model — renders
as a wireframe cuboid with three axis-aligned slice planes. Sliders move the
planes, each step shipping only that decimated slice; the camera survives
subsequent updates after rotating or zooming; and one stretch slider per axis
(0.125×–8×, geometric steps, 1.0 centered) reshapes the cuboid. The second
screenshot at the top is a line stretched 8× along the shot axis with AGC
applied through the same filter/gain chain as the gather views.

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/velocity_qc.png" width="440" alt="velocity model QC with depth axis"/>

Above is the 3-D velocity model behind that modelling run, viewed with
`axes=('x','y','depth')`; property volumes get depth labels and min/max colour
limits automatically.

This tab offers one cuboid and three planes, for a quick look at the volume
the gathers live in rather than for volume interpretation. **If your data is a
volume in the first place, [cigvis](https://github.com/JintaoLee-Roger/cigvis)
is the recommended tool**: it visualizes 3-D seismic data together with
labels, faults, RGT, horizon surfaces, well-log trajectories and curves and
geological bodies, and supports 2-D and 1-D data as well, using vispy for 3-D,
matplotlib for 2-D/1-D and plotly in Jupyter, with optional viser and OpenVDS
backends. The slice view here follows its interaction style.

### Display modes and colormaps

Every 2-D panel switches between variable density and variable-area wiggle.
*flip polarity* negates the display, swapping wiggle fill lobes and density
colours together (SEG normal ↔ reverse). Colormaps are `gray` (the default),
`seismic`, `petrel` (anchors from cigvis) and `rainbow`, each with an `_r`
reversed variant. The *clip percentile* slider sets the colour limits and
doubles as wiggle gain.

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/display_modes.png" width="620" alt="variable density vs wiggle"/>

### Filters and gain

Trapezoid (Ormsby-style) **low-pass / high-pass / band-pass** filtering:
linear ramps over `f1–f2` (low cut) and `f3–f4` (high cut), zero phase. Gain
is either **AGC** (sliding-window RMS, window length editable) or **trace
balance**. The chain is filter → gain; with gain active the colour limits are
recomputed from the processed gather.

Processing is applied per displayed gather, so memmap laziness is preserved.
The Volume slices tab uses the same chain: volumes up to 512 MB are processed
whole on each change to the chain, keeping the three planes consistent, while
larger memmaps fall back to raw data with a note in the sidebar. If the input
is a CuPy array, filtering, gain and picking run on the GPU.

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/filters.png" width="620" alt="band-pass filtering a noisy gather"/>

### Analysis windows and amplitude spectra

Draw **rectangles** (SHIFT+drag, or click-move-click) or **polygons** (click
vertices, ESC to finish) on a gather, then press *compute spectrum*: the gated
samples go straight into `rfft` and `|X|` is plotted relative to the window's
own peak. No taper, no normalization, no smoothing.

`traces` decides how the window's traces reach the plot: **per trace** (each
one drawn), **mean** (magnitudes averaged) or **middle trace**. `y axis`
switches between linear **amplitude** (normalized so the loudest curve peaks at
1.0) and **dB**. The legend states what was plotted and the frequency spacing
`df = 1/(gate length)`.

Taking no taper has two consequences, both properties of the DFT of a
rectangularly cut record: the boxcar gate's sidelobes fall off only as 1/f, so
roughly 40 dB below the peak what is read is the window rather than the data;
and a single trace's periodogram is a 2-degree-of-freedom estimate whose
scatter does not shrink with record length. Averaging reduces that scatter by
`sqrt(N)`, but only for independent traces — on a coherent gather the traces
are near-copies and the mean differs little from any one of them. A taper
would be a convolution of the spectrum (periodic Hann is `(-1/4, 1/2, -1/4)`
on the complex spectrum), i.e. smoothing, so none is applied.

Windows export and import as JSON, holding trace/time coordinates and dt.

<img src="https://raw.githubusercontent.com/zzzzswh/gathervis/main/docs/images/windows_spectra.png" width="720" alt="analysis windows and their spectra"/>

### f-k spectrum

**f-k spectrum** in the *Spectrum* block transforms the **first** drawn window
in both directions: wavenumber across, frequency up, amplitude in dB relative
to the window's own peak over a fixed 60 dB range. A linear event maps to a ray
through the origin, so events of different apparent velocity separate, which is
what decides whether an apparent-velocity filter is needed and how to set it.

Convention: **positive k means events dipping toward increasing trace number**.
The axis is in cycles per trace, not per metre — trace spacing is not
necessarily uniform, least of all after an offset or azimuth sort. As with the
1-D spectrum, the transform uses the currently displayed gather, i.e. after the
filter/gain chain, and recomputes when the shot, sort or filter changes. The
window must cover at least 4 traces and 8 samples. Both spectrum panels share
the *size* selector (S / M / L) and both carry a crosshair and a `k` / `f`
readout.

### Full-resolution export

Panels are stride-decimated to a pixel budget before going on the wire, and
bokeh's save tool snapshots the canvas at its current on-screen size; neither
gives the data at its own resolution. The **⤓ full image** button next to
*fit* and *fullscreen* re-renders the gather at one pixel per trace and per
sample and downloads a lossless PNG, with colormap, display mode, clip
percentile, polarity, filter and gain as displayed. Wiggle mode draws every
trace, not the subset the browser receives. The PNG is written by numpy and
stdlib zlib, with no matplotlib or headless browser involved.

For a crop of the current view, use the toolbar's save icon.

The same from a script, for batch figures:

```python
from gathervis.process import bandpass, agc
from gathervis.render import save_png

a = agc(bandpass(g.shot(7), g.dt, f3=60, f4=80), g.dt)
save_png("shot007.png", a, cmap="gray")            # or display="wiggle"
```

### First-break and event picking

The toolbar's point tool adds a pick on tap, moves one on drag and deletes the
selected one on BACKSPACE. A dotted line connects the picks in trace order.
**auto pick** runs a gapped STA/LTA first-break picker on the currently
displayed (i.e. filtered) gather, with STA, LTA and threshold editable: the LTA
window ends where the STA window starts, so a break arriving early in the
record can still trigger, and a trigger must hold for about STA/2 to exclude
single-sample noise. **snap** moves a pick to the nearest peak, trough or
|max| of its own trace, searching ±30 ms.

Picks are stored per shot and follow the shot being browsed. They are keyed to
the trace rather than the screen column, so they still match after a re-sort.
Export and import as CSV, `shot,trace,time_s`.

### SEG-Y import

`gv.from_segy('field.sgy')`, or `gathervis field.sgy` from the shell. dt is
read from the binary header, traces are grouped into shots by field record
number (FFID), and source and receiver coordinates are read from the standard
trace-header words and scaled by the SEG-Y coordinate scalar rule, so the
Geometry and Fold tabs are immediately usable. Shots with unequal trace counts
are zero-padded and a warning is printed. Requires the optional `segyio`
package; the file is read into memory, and one estimated above `max_gb`
(8 GB by default) is refused.

## Remote usage

1. **VS Code Remote-SSH (recommended).** Run any example in the integrated
   terminal; VS Code auto-forwards the port and the printed
   `http://localhost:8080` becomes clickable. Zero setup.
2. **Jupyter.** Omit `port=` and the returned app renders inline in the
   notebook — no extra port at all.
3. **Bare terminal.** `ssh -L 8080:localhost:8080 user@gpu-server`, then open
   the URL locally. If the port is busy gathervis auto-picks a free one and
   prints it.

## Array semantics

The last axis is always the vertical one: **time** for data, **depth** for
property volumes. A bare 3-D array defaults to `(shot, rec, time)` and is
browsed shot by shot; a 4-D array defaults to `(shot, recy, recx, time)`, one
3-D shot record per shot. Pass `view='slices'` to slice a 3-D array as a
volume instead, or declare other semantics explicitly with `axes=`. `axes`
says what the array *is*, `view` says how to *look* at it, and `sort` (in the
app) says in what order — the three are independent.

## Design notes

* **Lazy loading.** Data opens as a memmap; browsing a shot reads only that
  shot's bytes, slicing reads only that slice, and filtering and spectra apply
  to the displayed gather only. Sorting by acquisition order or receiver line
  is lazy too (a reshape, a slice); only offset/azimuth order materialises the
  shot — one shot, not the file.
* **Wire format.** Panels are stride-decimated to a pixel budget and quantized
  to uint8 server-side before being sent to the browser.
* **Ordering separate from processing.** A sort only changes trace order and
  returns the shot's own traces; the filter/gain chain then runs on the result,
  so 2-D and 3-D shots share one processing path.
* **Two rendering paths.** 2-D panels are bokeh images or wiggles, volumes are
  plotly WebGL slice planes, both fed by the same decimate → uint8 pipeline.
* **Export path independent of display.** Image export bypasses decimation:
  the same processed array is rasterized at native resolution and encoded as a
  PNG with numpy and stdlib zlib, adding no plotting or imaging dependency.
* **Interaction state preserved.** Sliders update while dragging; the 3-D
  camera, stretch factors and drawn windows survive updates. Every 2-D panel
  has x-only / y-only wheel zoom and a client-side cursor readout (trace, time,
  amplitude under the mouse); the spectrum panels have the same readout and a
  toggleable crosshair.

## Roadmap

**M2** (remaining): synced comparison panels · client-side volume cache for
cigvis-grade slice scrubbing (AGC / trace balance / CuPy: done). **3-D
acquisition QC** (remaining): shot-record time slices over the receiver grid ·
offset-azimuth rose diagrams per bin (3-D shot sorting, CMP fold map: done). **M3** (remaining): header indexing, gather extraction by any key
(SEG-Y import: done). **M4** (remaining):
common-offset/time slicing, NMO preview (event picking: done). Full plan:
[`docs/plan.md`](https://github.com/zzzzswh/gathervis/blob/main/docs/plan.md). *And whatever you ask for — see the note at
the top.*

## Acknowledgements

The `petrel` colormap anchors come from
[cigvis](https://github.com/JintaoLee-Roger/cigvis) (MIT). Modelling examples use
[deepwave](https://github.com/ar4/deepwave). Built on
[Panel](https://panel.holoviz.org/), [Bokeh](https://bokeh.org/) and
[Plotly](https://plotly.com/javascript/).

## License

MIT
