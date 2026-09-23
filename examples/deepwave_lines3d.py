"""Five 2-D lines over a rich 3-D velocity model, modelled with deepwave's
3-D scalar (acoustic) engine, browsed with gathervis.

The model (2.4 km x 1.6 km x 1.2 km): a gently dipping overburden, an
anticline dome, a deeper undulating interface, mild vertical gradients, and a
low-velocity channel lens that only some lines cross -- so the five lines look
genuinely different.

Acquisition: five parallel 2-D lines (shots and receivers on the same line,
fixed spread per line), wave propagation fully 3-D.

Run:       python examples/deepwave_lines3d.py            # server scale
           python examples/deepwave_lines3d.py --quick    # small/CPU scale
Data is cached; reruns skip torch and open the viewer in seconds:
    gathervis lines3d_data.npy --geom lines3d_geom.npz
QC the velocity model itself as a 3-D cuboid:
    gathervis lines3d_vel.npy --axes x y depth --dt 10 --cmap rainbow
"""
import argparse
import time
from pathlib import Path

import numpy as np

import gathervis as gv

DATA = Path("lines3d_data.npy")
GEOM = Path("lines3d_geom.npz")
VEL = Path("lines3d_vel.npy")


# ---------------------------------------------------------------------------
# the 3-D velocity model, v[z, y, x] in m/s (deepwave's index order)
# ---------------------------------------------------------------------------

_MISSING_TORCH = """
This example forward-models the data, which needs torch and deepwave. The
viewer itself does not -- they are declared as the `examples` extra, so they
are not installed by default (torch is ~2.5 GB against a 37 KB viewer).

    uv sync --extra examples          # add them to the project environment
    uv run --extra examples {script}  # or just for this one run

pyproject pins torch to the cu121 index. Without a GPU, or to keep it small:

    uv run --extra examples --index https://download.pytorch.org/whl/cpu {script}

With pip:  pip install "gathervis[examples]"
"""


def _import_deepwave(script):
    """torch + deepwave, or a message that says what to do about it."""
    try:
        import torch
        import deepwave
        from deepwave import scalar
    except ImportError as exc:
        raise SystemExit(f"{exc}\n{_MISSING_TORCH.format(script=script)}") from None
    return torch, deepwave, scalar


def build_model(nz, ny, nx, d):
    lx, ly = (nx - 1) * d, (ny - 1) * d
    x = np.arange(nx) * d
    y = np.arange(ny) * d
    z = np.arange(nz) * d
    X, Y = np.meshgrid(x, y, indexing="xy")            # (ny, nx) surfaces
    Z = z[:, None, None]                               # broadcast to (nz,ny,nx)

    # interfaces as depth surfaces z(x, y)
    z1 = 210.0 + 0.045 * X - 0.030 * Y                             # dipping
    z2 = 560.0 - 190.0 * np.exp(-(((X - 0.52 * lx) ** 2            # anticline
                                   + (Y - 0.55 * ly) ** 2)
                                  / (2 * 420.0 ** 2)))
    z3 = 800.0 + 0.085 * X + 35.0 * np.sin(2 * np.pi * Y / ly)     # undulating

    v = np.full((nz, ny, nx), 1600.0, dtype=np.float32)
    v[Z >= z1] = 2050.0
    v[Z >= z2] = 2500.0
    v[Z >= z3] = 3100.0
    v += (0.20 * Z).astype(np.float32)                 # mild vertical gradient

    # low-velocity channel lens inside layer 2 (only some lines cross it)
    cx, cy, cz, rx, ry, rz = 0.72 * lx, 0.32 * ly, 470.0, 320.0, 170.0, 55.0
    lens = (((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
            + ((Z - cz) / rz) ** 2) <= 1.0
    v[lens] = 1850.0
    return v


# ---------------------------------------------------------------------------
# modelling
# ---------------------------------------------------------------------------
def model(p):
    _t0 = time.perf_counter()
    torch, deepwave, scalar = _import_deepwave("examples/deepwave_lines3d.py")
    print(f"[timer] import torch/deepwave: {time.perf_counter() - _t0:.1f}s")

    def _pick_device():
        if torch.cuda.is_available():
            try:
                (torch.zeros(1, device="cuda") + 1).item()
                return torch.device("cuda")
            except Exception as e:
                print(f"CUDA present but unusable ({e}); falling back to CPU.")
        return torch.device("cpu")

    device = _pick_device()
    print(f"modelling on {device}")

    v_np = build_model(p.nz, p.ny, p.nx, p.d)
    np.save(VEL, v_np.transpose(2, 1, 0))              # (x, y, depth) for QC
    v = torch.from_numpy(v_np).to(device)

    # -- five parallel lines along x --------------------------------------
    line_iy = np.linspace(0.22 * p.ny, 0.78 * p.ny, 5).round().astype(int)
    # Stations on whole grid points at a constant step, not linspace().round().
    # Rounding a fractional step gives alternating 1- and 2-cell gaps, and the
    # midpoints inherit the jitter -- which shows up on the fold map as holes
    # inside the live area, because a CMP bin is half a station interval and
    # has nowhere to put a midpoint that landed off the grid.
    rec_step = max(1, (p.nx - 17) // p.nrec)
    rec_ix = 8 + rec_step * np.arange(p.nrec)
    src_step = max(1, (rec_ix[-1] - rec_ix[0]) // (p.nsrc + 1) // rec_step * rec_step)
    src_ix = rec_ix[0] + src_step + src_step * np.arange(p.nsrc)
    iz = 1                                             # just below the surface

    src_list, rec_list = [], []                        # (iz, iy, ix) per shot
    for iy in line_iy:
        for sx in src_ix:
            src_list.append((iz, iy, sx))
            rec_list.append([(iz, iy, rx) for rx in rec_ix])
    ns = len(src_list)

    src_loc = torch.tensor(src_list, dtype=torch.long,
                           device=device).reshape(ns, 1, 3)
    rec_loc = torch.tensor(rec_list, dtype=torch.long, device=device)

    wavelet = deepwave.wavelets.ricker(p.freq, p.nt, p.dt, 1.5 / p.freq)
    amp = wavelet.reshape(1, 1, -1).to(device)

    data = np.empty((ns, p.nrec, p.nt), dtype=np.float32)
    _t0 = time.perf_counter()
    for s0 in range(0, ns, p.batch):                   # chunk to bound memory
        s1 = min(s0 + p.batch, ns)
        out = scalar(v, p.d, p.dt,
                     source_amplitudes=amp.repeat(s1 - s0, 1, 1),
                     source_locations=src_loc[s0:s1],
                     receiver_locations=rec_loc[s0:s1],
                     accuracy=p.accuracy, pml_freq=p.freq)
        data[s0:s1] = out[-1].detach().cpu().numpy()
        done = time.perf_counter() - _t0
        print(f"  shots {s1}/{ns}  ({done:.0f}s, {done / s1:.1f}s/shot)",
              flush=True)
    print(f"[timer] modelling {data.shape}: {time.perf_counter() - _t0:.1f}s")

    # world coordinates (x, y) for the layout map
    src_xy = np.array([(sx * p.d, iy * p.d) for (_, iy, sx) in src_list])
    rec_xy = np.array([[(rx * p.d, r[0][1] * p.d) for (_, _, rx) in r]
                       for r in rec_list])             # (ns, nrec, 2)
    np.save(DATA, data)
    np.savez(GEOM, src=src_xy, rec=rec_xy, dt=p.dt)
    print(f"saved {DATA}, {GEOM}, {VEL}")
    print(f"QC the velocity model:\n    gathervis {VEL} "
          f"--axes x y depth --dt {p.d:g} --cmap rainbow")


def _parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true",
                   help="small CPU-friendly scale (coarser grid, fewer shots)")
    p.add_argument("--port", type=int, default=8080)
    return p


class P:                                               # parameter bundle
    pass


def params(quick):
    p = P()
    if quick:      # ~ minutes on a laptop CPU
        p.nx, p.ny, p.nz, p.d = 120, 80, 60, 20.0
        p.nsrc, p.nrec = 6, 96                         # per line x 5 lines
        p.nt, p.dt, p.freq = 600, 0.002, 12.0
        p.batch, p.accuracy = 2, 4
    else:          # GPU-server scale
        p.nx, p.ny, p.nz, p.d = 240, 160, 120, 10.0
        p.nsrc, p.nrec = 12, 192
        p.nt, p.dt, p.freq = 1250, 0.002, 15.0
        p.batch, p.accuracy = 6, 8
    return p


if __name__ == "__main__":
    args = _parser().parse_args()
    if not DATA.exists():
        model(params(args.quick))
    else:
        print(f"found {DATA} - reusing cached data (delete it to re-model)")

    geom = np.load(GEOM)
    ds = gv.from_array(np.load(DATA, mmap_mode="r"),
                       src=geom["src"], rec=geom["rec"], dt=float(geom["dt"]),
                       name="3-D acoustic modelling, five 2-D lines, "
                            "dome + channel-lens 3-D velocity model")
    print(ds)
    gv.show(ds, port=args.port)
