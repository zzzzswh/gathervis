"""Model a 2-D line with deepwave, then browse it with gathervis.

Requires:  pip install deepwave   (pulls torch; runs on GPU if available)
Run:       python examples/deepwave_line2d.py
Then open http://localhost:8080 (use `ssh -L 8080:localhost:8080 user@server`
when the script runs on a remote GPU server).
"""
import numpy as np
import torch
import deepwave
from deepwave import scalar

import gathervis as gv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"modelling on {device}")

# ---- model: three flat layers ------------------------------------------------
nz, nx = 160, 600
dx = 5.0                      # grid spacing (m)
v = torch.full((nz, nx), 1500.0)
v[60:] = 2100.0
v[110:] = 2800.0
v = v.to(device)

# ---- acquisition: fixed spread, shots rolling along the line -----------------
nt, dt, freq = 1200, 0.002, 15.0
n_shots = 60
n_rec = 192
src_iz, rec_iz = 1, 1                                # near-surface indices
src_ix = torch.linspace(20, nx - 20, n_shots).long()
rec_ix = torch.linspace(10, nx - 10, n_rec).long()

source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
source_locations[:, 0, 0] = src_iz
source_locations[:, 0, 1] = src_ix

receiver_locations = torch.zeros(n_shots, n_rec, 2, dtype=torch.long, device=device)
receiver_locations[..., 0] = rec_iz
receiver_locations[..., 1] = rec_ix

wavelet = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(device)
source_amplitudes = wavelet.repeat(n_shots, 1, 1)    # (n_shots, 1, nt)

# ---- forward modelling -------------------------------------------------------
out = scalar(v, dx, dt,
             source_amplitudes=source_amplitudes,
             source_locations=source_locations,
             receiver_locations=receiver_locations,
             accuracy=8, pml_freq=freq)
data = out[-1].detach().cpu().numpy().astype(np.float32)   # (n_shots, n_rec, nt)
print("modelled:", data.shape)

# ---- geometry in physical coordinates ----------------------------------------
src_xyz = np.stack([src_ix.numpy() * dx, np.zeros(n_shots)], axis=1)
rec_xyz = np.stack([rec_ix.numpy() * dx, np.zeros(n_rec)], axis=1)

np.save("line2d_data.npy", data)
np.savez("line2d_geom.npz", src=src_xyz, rec=rec_xyz, dt=dt)

# ---- view ---------------------------------------------------------------------
ds = gv.from_array(data, src=src_xyz, rec=rec_xyz, dt=dt)
print(ds)
gv.show(ds, port=8080)
