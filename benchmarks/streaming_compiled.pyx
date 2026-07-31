# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""Compiled event loops, calling each library exactly as a Python user would.

This is the "so compile it" scenario, done honestly. A user who Cython-compiles
their event loop still calls screamer, TA-Lib, numpy or pandas through their
Python objects: compiling removes the interpreter's loop overhead, it does not
remove the boundary crossing into each library. So these loops are compiled and
the calls inside them are not, which is exactly what such a user gets.

Deliberately *not* done here: calling screamer's C++ recurrence directly via
`cdef extern`. That would measure something a Python user cannot obtain without
writing C++ themselves, and would flatter screamer against libraries being
measured through their wrappers. The number for screamer's engine with no
per-event boundary crossing is already available a different way, through the
array API and `Pipeline`, which are compiled loops the user does not have to
write. That distinction is the point rather than a detail: screamer offers a
tier that a batch library cannot offer at all for streaming.

`window_recompute_loop` is the shape any batch API forces on a streaming
problem: keep the last `window` samples and hand them to the library every
event, which is O(window) per event however fast the library is.
"""
import numpy as np
cimport numpy as cnp

cnp.import_array()


def screamer_loop(double[::1] values, object operator):
    """Compiled loop, one call into the operator per event.

    The call is a Python call, so it pays pybind11 dispatch. That cost is
    charged to every library here equally.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t n = values.shape[0]
    cdef object out = 0.0
    for i in range(n):
        out = operator(values[i])
    return out


def window_recompute_loop(double[::1] values, object function, int window):
    """Compiled loop, handing the trailing window to a batch function per event.

    The ring buffer is preallocated and reused, so nothing is allocated per
    event. `function` receives the window and returns its result; only the last
    element is kept, which is what a streaming caller needs.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t n = values.shape[0]
    cdef Py_ssize_t position = 0
    buffer = np.zeros(window, dtype=np.float64)
    cdef double[::1] view = buffer
    cdef object out = 0.0
    for i in range(n):
        view[position] = values[i]
        position += 1
        if position == window:
            position = 0
        out = function(buffer)
    return out


def hand_written_mean_loop(double[::1] values, int window):
    """The recurrence written by hand in Cython: no library call at all.

    This is the honest floor for "I would just write it myself and compile it".
    It is the same algorithm screamer runs, with no boundary crossing, and it
    implements the same NaN and warmup policy so the comparison is like for
    like.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t n = values.shape[0]
    cdef Py_ssize_t position = 0
    cdef Py_ssize_t seen = 0
    cdef double total = 0.0
    cdef double v, old
    cdef double nan = float("nan")
    cdef double out = 0.0
    buffer = np.zeros(window, dtype=np.float64)
    cdef double[::1] buf = buffer
    for i in range(n):
        v = values[i]
        if v != v:                       # NaN: skip, leave state untouched
            out = nan
            continue
        old = buf[position]
        buf[position] = v
        total += v - old
        position += 1
        if position == window:
            position = 0
        if seen < window:
            seen += 1
        out = total / window if seen == window else nan
    return out
