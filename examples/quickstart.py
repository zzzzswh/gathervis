"""Torch-free quickstart: every view on a tiny analytic synthetic.

Run:  python examples/quickstart.py [mode]
      mode = geo (default) | browse | slices | single | patch
      optional 2nd arg = port
Then open http://localhost:8080 (SSH port-forward on a remote server).
"""
import sys

import gathervis as gv
from gathervis.demo import synthetic_line, synthetic_patch

mode = sys.argv[1] if len(sys.argv) > 1 else "geo"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

if mode == "patch":
    # A 3-D shot record: 10 receiver lines of 96 stations. The Shot gathers
    # tab opens in acquisition order -- ten nested hyperbolas, one per
    # receiver line -- and the 'sort' selector re-lays the same shot out one
    # line at a time, by offset, or by azimuth.
    patch = synthetic_patch(ns=12, nly=10, nlx=96, nt=751)
    print(patch)
    gv.show(patch, port=port)
    sys.exit()

line = synthetic_line(ns=48, nr=120, nt=1001)
print(line)

if mode == "single":                      # one 2-D gather, plain and fast
    gv.show(line.shot(10), dt=line.dt, port=port)
elif mode == "browse":                    # (shot, rec, time): default browsing
    gv.show(line.data, dt=line.dt, port=port)
elif mode == "slices":                    # same array as a 3-D cuboid w/ slices
    gv.show(line.data, dt=line.dt, view="slices", port=port)
else:                                     # geometry: map + linked shot gathers
    gv.show(line, port=port)