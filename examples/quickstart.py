"""Torch-free quickstart: every M1 view on a tiny analytic synthetic.

Run:  python examples/quickstart.py [mode]
      mode = geo (default) | browse | slices | single | 4d ; optional 2nd arg = port
Then open http://localhost:8080 (SSH port-forward on a remote server).
"""
import sys

import numpy as np

import gathervis as gv
from gathervis.demo import synthetic_line

mode = sys.argv[1] if len(sys.argv) > 1 else "geo"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
line = synthetic_line(ns=48, nr=120, nt=1001)
print(line)

if mode == "single":                      # one 2-D gather, plain and fast
    gv.show(line.shot(10), dt=line.dt, port=port)
elif mode == "browse":                    # (shot, rec, time): default browsing
    gv.show(line.data, dt=line.dt, port=port)
elif mode == "slices":                    # same array viewed as a volume
    gv.show(line.data, dt=line.dt, view="slices", port=port)
elif mode == "4d":                        # 3-D shot gathers: per-shot slice view
    d = np.asarray(line.data).reshape(48, 6, 20, 1001)  # fake (shot,recy,recx,t)
    gv.show(d, dt=line.dt, port=port)
else:                                     # geometry: map + linked shot gathers
    gv.show(line, port=port)
