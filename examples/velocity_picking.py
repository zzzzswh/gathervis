"""Velocity analysis by hand, on a CMP gather whose answer is known.

The model below is built from (depth, velocity) layers, so each reflector
sits at a zero-offset time of ``2 * depth / v`` with moveout velocity ``v``.
The script prints where the spectrum's peaks are, so you can check your own
picks against them once the panel is open.

Run:  python examples/velocity_picking.py [port]
      then open http://localhost:8080 (SSH port-forward on a remote server)

What to do once it is open
--------------------------
Take the point tool in the spectrum's toolbar and tap the bright spots. Each
tap adds a control point; drag to refine, tap-select and BACKSPACE to remove.

Watch the **moveout corrected** panel while you drag. Too slow and the event
arches up at the far offsets, too fast and it smiles down, right and it is
flat -- that is the only test of a pick, and the reason the two panels sit
side by side. The **stack** on the right is what the whole gather collapses
to under your pick.

When it looks right, *export velocity file* writes ``time_s,velocity`` with
the bin's location in the header. It reads back in through the file input
next to it.
"""
import sys

import numpy as np

import gathervis as gv
from gathervis import cmp, survey
from gathervis.demo import synthetic_line

LAYERS = ((300.0, 1800.0), (700.0, 2300.0), (1150.0, 2900.0))
TRUTH = [(2.0 * depth / v, v) for depth, v in LAYERS]
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

line = synthetic_line(ns=60, nr=120, nt=1001, d_src=50.0, d_rec=25.0,
                      layers=LAYERS, noise=0.05)
print(line)

# Fold is geometry only, so finding the best-illuminated bin costs nothing;
# reading its traces is the first thing that goes to disk.
grid = survey.fold(line.geometry)
cg = cmp.gather_in_bin(line, grid, *cmp.best_bin(grid))
print(f"\n{cg}")

# The panel takes the gather it is given. A CmpGather carries its own
# offsets and bin location, so there is nothing else to pass.
print("\n  the answer, for checking your picks against:")
print("  t0 (s)   v true")
for t0, v in TRUTH:
    print(f"  {t0:6.3f}   {v:6.0f}")

print(f"\nserving on http://localhost:{port} -- tap the spectrum to pick")
gv.velocity_analysis(cg, dt=line.dt, port=port)
