"""gathervis: interactive pre-stack seismic gather viewer for the Python/GPU-server era.

Quickstart::

    import gathervis as gv
    gv.show(gather_2d)                       # (trace, time) panel
    gv.show(line_3d)                         # default (shot, rec, time), browse shots
    gv.show(line_3d, view='slices')          # same data as a sliceable volume
    ds = gv.from_array(data, src=src_xyz, rec=rec_xyz, dt=0.002)
    gv.show(ds, port=8080)                   # acquisition map + linked shot gathers
"""
from .core import Gathers, Geometry, from_array, from_file, from_segy

__version__ = "0.15.0"
__all__ = ["Gathers", "Geometry", "from_array", "from_file", "from_segy", "show"]


def __getattr__(name):
    # Defer the heavy panel/bokeh import to the first gv.show() call, so that
    # `import gathervis` stays instant for core/IO-only usage.
    if name == "show":
        from .viewer import show
        return show
    raise AttributeError(f"module 'gathervis' has no attribute {name!r}")
