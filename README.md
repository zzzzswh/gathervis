# gathervis

**Interactive pre-stack seismic gather viewer for the Python / GPU-server era.**

叠前地震数据交互式查看器：像 cigvis 之于叠后数据一样，让你在 GPU 服务器上用几行代码 + 一个浏览器查看正演/实测的炮集数据。SSH 端口转发即可交互，无需 X11、无需把数据拖回本地。

## Install

```bash
pip install -e .            # numpy + panel + bokeh
pip install deepwave        # optional, for the modelling example (pulls torch)
```

## Quickstart

```python
import gathervis as gv

# -- mode A: quick look, no geometry --
gv.show(gather_2d, dt=0.002, port=8080)          # a (trace, time) panel
gv.show(line_3d, dt=0.002, port=8080)            # default (shot, rec, time): browse shots
gv.show(line_3d, view='slices', port=8080)       # same array as a sliceable volume
gv.show(vol, axes=('recy','recx','time'))        # a 3-D shot gather -> slice view
gv.show('shots.bin', shape=(60,192,1200),        # raw binary: lazy memmap, opens instantly
        dtype='float32', dt=0.002, port=8080)

# -- mode B: with acquisition geometry --
ds = gv.from_array(data, src=src_xyz, rec=rec_xyz, dt=0.002)
gv.show(ds, port=8080)   # layout map; tap a source point -> its shot gather
```

## Remote usage (pick one)

1. **VS Code Remote-SSH (recommended).** Run any example in the integrated
   terminal; VS Code auto-forwards the port and the printed
   `http://localhost:8080` becomes clickable. Zero setup.
2. **Jupyter.** Omit `port=` and the returned app renders inline in the
   notebook -- no extra port at all.
3. **Bare terminal (fallback).** Forward the port yourself, then open the URL
   locally:

   ```bash
   ssh -L 8080:localhost:8080 user@gpu-server
   ```

   One-time setup: add `LocalForward 8080 127.0.0.1:8080` to the host entry in
   your local `~/.ssh/config`. If the port is busy gathervis auto-picks a free
   one and prints it.

**Fast viewing of saved data** (no torch import) via the CLI:

```bash
gathervis line2d_data.npy --geom line2d_geom.npz
gathervis shots.bin --shape 60 192 1200 --dt 0.002
```

Try it without any data: `python examples/quickstart.py` (analytic synthetic), or
`python examples/deepwave_line2d.py` for real wave-equation modelling with deepwave.

## Semantics (the one rule to remember)

The last axis is always **time**. A bare 3-D array defaults to `(shot, rec, time)`
and is browsed shot by shot; pass `view='slices'` to slice it as a volume instead,
or declare other semantics explicitly with `axes=`. `axes` says what the array
*is*, `view` says how to *look* at it — the two are independent.

## Design notes

* **Lazy everywhere.** memmap-backed data; browsing a shot reads only that
  shot's bytes; slicing reads only that slice.
* **Cheap wire format.** Panels are stride-decimated to a pixel budget and
  quantized to uint8 server-side before shipping to the browser.
* **One rendering primitive.** Every view is the same `ImagePane`; ~700 lines
  of package code total. Keep it simple, stupid.

## Roadmap

M1 (this release): dual-mode shot-gather viewing, acquisition map with linked
browsing, web-first serving. M2: AGC / trace balance / gain / wiggle display /
synced panels (CuPy-accelerated when data lives on the GPU server). M3: header
indexing, CMP binning, gather extraction by any key, SEG-Y import. M4: common-
offset/time slicing, event picking, NMO preview. Full plan: `docs/plan.md`.

## License

MIT
