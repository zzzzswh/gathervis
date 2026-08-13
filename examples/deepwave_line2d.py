"""Model a 2-D line with deepwave, then browse it with gathervis.

Requires:  pip install deepwave   (pulls torch; runs on GPU if available)
Run:       python examples/deepwave_line2d.py
The modelled data is cached (line2d_data.npy); reruns skip torch entirely and
open the viewer in seconds. Delete the file to re-model.
Viewing saved data directly (no torch, fastest):
    gathervis line2d_data.npy --geom line2d_geom.npz
"""
import time
from pathlib import Path

import numpy as np

import gathervis as gv

DATA, GEOM = Path("line2d_data.npy"), Path("line2d_geom.npz")


def model():
    """Forward-model the line with deepwave (imports torch: slow first time)."""
    _t0 = time.perf_counter()
    import torch
    import deepwave
    from deepwave import scalar
    print(f"[timer] import torch/deepwave: {time.perf_counter() - _t0:.1f}s")

    def _pick_device():
        if torch.cuda.is_available():
            try:
                (torch.zeros(1, device="cuda") + 1).item()
                return torch.device("cuda")
            except Exception as e:  # e.g. cudaErrorNoKernelImageForDevice
                print(f"CUDA present but unusable ({e}); falling back to CPU.")
        return torch.device("cpu")

    device = _pick_device()
    print(f"modelling on {device}")

    # ---- model: three flat layers --------------------------------------
    nz, nx = 160, 600
    dx = 5.0                      # grid spacing (m)
    v = torch.full((nz, nx), 1500.0)
    v[60:] = 2100.0
    v[110:] = 2800.0
    v = v.to(device)

    # ---- acquisition: fixed spread, shots rolling along the line -------
    nt, dt, freq = 1200, 0.002, 15.0
    n_shots, n_rec = 60, 192
    src_iz, rec_iz = 1, 1
    src_ix = torch.linspace(20, nx - 20, n_shots).long()
    rec_ix = torch.linspace(10, nx - 10, n_rec).long()

    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[:, 0, 0] = src_iz
    source_locations[:, 0, 1] = src_ix
    receiver_locations = torch.zeros(n_shots, n_rec, 2, dtype=torch.long,
                                     device=device)
    receiver_locations[..., 0] = rec_iz
    receiver_locations[..., 1] = rec_ix

    wavelet = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(device)
    source_amplitudes = wavelet.repeat(n_shots, 1, 1)

    _t0 = time.perf_counter()
    out = scalar(v, dx, dt,
                 source_amplitudes=source_amplitudes,
                 source_locations=source_locations,
                 receiver_locations=receiver_locations,
                 accuracy=8, pml_freq=freq)
    data = out[-1].detach().cpu().numpy().astype(np.float32)
    print(f"[timer] modelling {data.shape}: {time.perf_counter() - _t0:.1f}s")

    src_xyz = np.stack([src_ix.numpy() * dx, np.zeros(n_shots)], axis=1)
    rec_xyz = np.stack([rec_ix.numpy() * dx, np.zeros(n_rec)], axis=1)
    np.save(DATA, data)
    np.savez(GEOM, src=src_xyz, rec=rec_xyz, dt=dt)


if not DATA.exists():
    model()
else:
    print(f"found {DATA} - reusing cached data (delete it to re-model)")

geom = np.load(GEOM)
ds = gv.from_array(np.load(DATA, mmap_mode="r"),
                   src=geom["src"], rec=geom["rec"], dt=float(geom["dt"]),
                   name="2-D acoustic modelling, 3-layer 2-D velocity model")
print(ds)
gv.show(ds, port=8080)
