"""Streaming references: cost per event when samples arrive one at a time.

The batch suite (`reference_impls.py`) hands every library the whole array at
once. That is the regime the reference libraries are built for. This file
measures the other one: an event arrives, and the current value of the
statistic is needed before the next event.

**The alternatives are given the strongest implementation available, not a
naive one.** Recomputing over the entire history per event would be O(n) and
would make the comparison worthless. What a practitioner actually writes is a
ring buffer of the last `window` samples with a reduction over it per event,
which is O(window), and that is what the `*_window` variants below do: the
buffer is preallocated, the new sample overwrites the oldest, and the reduction
is the fastest one numpy offers. No allocation happens per event.

A second alternative is included because it is the honest reply to "I would
just write the update myself": `*_incremental` is a hand-written O(1) Python
update, the same algorithm screamer runs in C++. For a rolling mean that is a
running sum; for a rolling extremum it is a monotonic deque. It is O(1) like
screamer and loses only on interpreter overhead, which is the fair thing to
show.

What the numbers actually show, measured on 8k events (macOS/arm64, so treat
the absolute values as indicative and the ratios as the point):

  * `screamer_engine`, the same recurrence run without crossing the Python
    boundary per event, costs 2-3 ns/event and does not grow with `window`.
    This is the number that answers "how fast can this consume a feed".
  * The per-event Python API costs ~115 ns/event, flat in `window`. Almost all
    of that is the interpreter and the pybind11 call, not the operator.
  * A per-event numpy reduction over the window costs 1.2-1.9 us for a mean,
    4.0-7.1 us for a standard deviation; pandas 11.8-16.0 us. So against the
    way this is usually written, the per-event API is 11x to 140x quicker, and
    the engine three orders of magnitude.
  * `*_incremental`, the hand-written O(1) Python update, costs about the same
    as screamer's per-event API: 91-135 ns. Both are paying for the
    interpreter. This is worth stating plainly rather than hiding: if you write
    the recurrence yourself in Python and only ever call it per event from
    Python, you will not be slower. What you give up is every operator being
    correct, `NaN`-compliant and identical in batch, and the engine speed above
    once events stop crossing the boundary.
  * The `*_window` variants grow with `window` more slowly than O(window)
    would suggest at these sizes, roughly 1.5x from window 10 to 5000, because
    numpy's fixed per-call overhead dominates its own reduction until the
    window is large.

Every variant is named `<Func>__<lib>` and takes `(values, window_size)`, where
`values` is the event sequence. Each returns the last output, so nothing is
optimised away.
"""
import inspect
import sys
from collections import deque

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:                                  # pragma: no cover
    njit = None

try:
    import streaming_compiled as _cy
except ImportError:                                  # pragma: no cover
    _cy = None                                       # build it: see build_compiled.py

try:
    import talib
except ImportError:                                  # pragma: no cover
    talib = None

import screamer as sc


# ---------------------------------------------------------------------------
# RollingMean
# ---------------------------------------------------------------------------

def RollingMean__screamer(values, window_size):
    op = sc.RollingMean(window_size)
    out = 0.0
    for v in values:
        out = op(v)
    return out


def RollingMean__numpy_window(values, window_size):
    """Ring buffer plus a numpy reduction per event: O(window)."""
    buffer = np.zeros(window_size)
    position = 0
    out = 0.0
    for v in values:
        buffer[position] = v
        position += 1
        if position == window_size:
            position = 0
        out = buffer.mean()
    return out


def RollingMean__pandas_window(values, window_size):
    buffer = np.zeros(window_size)
    position = 0
    out = 0.0
    for v in values:
        buffer[position] = v
        position += 1
        if position == window_size:
            position = 0
        out = pd.Series(buffer).mean()
    return out


def RollingMean__incremental(values, window_size):
    """Hand-written O(1) update in Python: the same algorithm, interpreted."""
    buffer = [0.0] * window_size
    position = 0
    total = 0.0
    out = 0.0
    for v in values:
        total += v - buffer[position]
        buffer[position] = v
        position += 1
        if position == window_size:
            position = 0
        out = total / window_size
    return out


# ---------------------------------------------------------------------------
# RollingMax
# ---------------------------------------------------------------------------

def RollingMax__screamer(values, window_size):
    op = sc.RollingMax(window_size)
    out = 0.0
    for v in values:
        out = op(v)
    return out


def RollingMax__numpy_window(values, window_size):
    buffer = np.full(window_size, -np.inf)
    position = 0
    out = 0.0
    for v in values:
        buffer[position] = v
        position += 1
        if position == window_size:
            position = 0
        out = buffer.max()
    return out


def RollingMax__incremental(values, window_size):
    """Monotonic deque in Python: O(1) amortised, the same algorithm screamer
    runs in C++."""
    window = deque()
    index = 0
    out = 0.0
    for v in values:
        while window and window[-1][0] <= v:
            window.pop()
        window.append((v, index))
        if window[0][1] <= index - window_size:
            window.popleft()
        index += 1
        out = window[0][0]
    return out


# ---------------------------------------------------------------------------
# RollingStd
# ---------------------------------------------------------------------------

def RollingStd__screamer(values, window_size):
    op = sc.RollingStd(window_size)
    out = 0.0
    for v in values:
        out = op(v)
    return out


def RollingStd__numpy_window(values, window_size):
    buffer = np.zeros(window_size)
    position = 0
    out = 0.0
    for v in values:
        buffer[position] = v
        position += 1
        if position == window_size:
            position = 0
        out = buffer.std(ddof=1)
    return out


# ---------------------------------------------------------------------------
# EwMean: no window, so the batch alternative has to see the whole history
# ---------------------------------------------------------------------------

def EwMean__screamer(values, window_size):
    op = sc.EwMean(span=window_size)
    out = 0.0
    for v in values:
        out = op(v)
    return out


def EwMean__incremental(values, window_size):
    alpha = 2.0 / (window_size + 1.0)
    weighted = 0.0
    weight = 0.0
    out = 0.0
    for v in values:
        weighted = weighted * (1.0 - alpha) + v
        weight = weight * (1.0 - alpha) + 1.0
        out = weighted / weight
    return out


# ---------------------------------------------------------------------------
# The engine, without the per-event boundary crossing
# ---------------------------------------------------------------------------
# The variants above all loop in Python, so they all pay ~100 ns per event for
# the interpreter and, for screamer, the pybind11 call. That cost is real for a
# Python-driven event loop and is charged to everyone equally, but it is not
# screamer's engine speed and it hides the thing the engine is good at.
#
# These feed the same recurrence through the array API, which runs the identical
# per-sample code path in C++ with no boundary crossing per event. It is the
# number to quote for "how fast can this consume a feed", and the honest way to
# reach it from Python is a Pipeline, where events stay in C++.

def RollingMean__screamer_engine(values, window_size):
    return sc.RollingMean(window_size)(values)[-1]


def RollingMax__screamer_engine(values, window_size):
    return sc.RollingMax(window_size)(values)[-1]


def RollingStd__screamer_engine(values, window_size):
    return sc.RollingStd(window_size)(values)[-1]


def EwMean__screamer_engine(values, window_size):
    return sc.EwMean(span=window_size)(values)[-1]


# ---------------------------------------------------------------------------
# Compiled Python
# ---------------------------------------------------------------------------
# The obvious reply to any of this is "so compile it". These are the answer:
# the same recurrences under numba, consuming the whole stream inside compiled
# code, which is the fair comparison against `screamer_engine` above.
#
# Calling a compiled function *per event* from Python is not that comparison
# and is worth knowing about separately: a numba jitclass method costs about
# 360 ns/event against screamer's 111, because every call boxes and unboxes its
# arguments. Compiling the operator does not help if Python still drives the
# loop; it helps when the loop itself is compiled.
#
# What these show: for a simple recurrence, compiled Python is in the same
# league as the C++ engine, and the argument for screamer there is not speed
# but that 231 operators are already written, NaN-compliant and identical in
# batch. For anything with a real algorithm behind it the gap reopens, because
# what matters is the algorithm rather than the language: the rolling maximum
# below is a hand-written monotonic deque, which is the obvious thing to write
# and several times slower than the block decomposition screamer uses.

if njit is not None:

    @njit(cache=True)
    def _numba_rolling_mean(values, window):
        n = len(values)
        out = np.empty(n)
        buffer = np.zeros(window)
        position = 0
        total = 0.0
        seen = 0
        for i in range(n):
            v = values[i]
            if np.isnan(v):                  # the `ignore` policy, as screamer has it
                out[i] = np.nan
                continue
            total += v - buffer[position]
            buffer[position] = v
            position += 1
            if position == window:
                position = 0
            if seen < window:
                seen += 1
            out[i] = total / window if seen == window else np.nan
        return out

    @njit(cache=True)
    def _numba_rolling_max(values, window):
        n = len(values)
        out = np.empty(n)
        capacity = window + 1
        vals = np.empty(capacity)
        idxs = np.empty(capacity, dtype=np.int64)
        head = 0
        count = 0
        for i in range(n):
            v = values[i]
            while count > 0:
                back = head + count - 1
                if back >= capacity:
                    back -= capacity
                if vals[back] > v:
                    break
                count -= 1
            slot = head + count
            if slot >= capacity:
                slot -= capacity
            vals[slot] = v
            idxs[slot] = i
            count += 1
            if idxs[head] <= i - window:
                head += 1
                if head == capacity:
                    head = 0
                count -= 1
            out[i] = vals[head]
        return out

    def RollingMean__numba_engine(values, window_size):
        return _numba_rolling_mean(values, window_size)[-1]

    def RollingMax__numba_engine(values, window_size):
        return _numba_rolling_max(values, window_size)[-1]


# ---------------------------------------------------------------------------
# Compiled event loop, libraries still called through Python
# ---------------------------------------------------------------------------
# The realistic "so compile it" case: the user Cython-compiles the loop that
# drives their event handler, but still calls screamer, TA-Lib, numpy or pandas
# through their Python objects. Compiling removes the interpreter's loop
# overhead; it does not remove the boundary crossing into each library, and it
# removes it for nobody, so the comparison stays fair.
#
# Measured on 200k events, ns/event (macOS/arm64):
#
#     window          10     100    1000
#     screamer      89.5    92.0    92.6      flat
#     talib        547.4   618.8  1324.4      grows with the window
#     numpy       1482.4  1521.4  1686.5
#     pandas     11491.2 11708.7 12954.4
#     hand-written   0.87    0.87    0.88     no boundary crossing at all
#     screamer array 1.60    0.91    1.58     no boundary crossing either
#
# Three things fall out of that table.
#
# Compiling the loop helps: screamer goes from ~115 ns/event under a Python
# loop to ~90. It helps everyone equally, and it does not change the ordering.
#
# In this regime screamer is 6-14x quicker than TA-Lib, 16-18x than numpy and
# ~130x than pandas, and only screamer is flat in the window: the others have
# to hand the whole window back every event because they have no incremental
# state to update.
#
# But ~90 ns of that is the boundary crossing, not the operator, and the last
# two columns are the interesting ones: a hand-written compiled loop and
# screamer's own array path both run at ~1 ns/event, a hundred times quicker,
# because neither crosses the boundary per event. That is the real lesson. The
# per-event call is a tax every library pays, and the way out is to let the
# loop live in compiled code. screamer hands you that loop already written, for
# 231 operators, with the NaN policy and batch/stream equality that a
# hand-written one does not have. TA-Lib has no equivalent to offer at all,
# because it has no streaming API to compile a loop around.

if _cy is not None:

    def RollingMean__compiled_screamer(values, window_size):
        return _cy.screamer_loop(values, sc.RollingMean(window_size))

    def RollingMean__compiled_numpy(values, window_size):
        return _cy.window_recompute_loop(values, np.mean, window_size)

    def RollingMean__compiled_handwritten(values, window_size):
        return _cy.hand_written_mean_loop(values, window_size)

    if talib is not None:
        def RollingMean__compiled_talib(values, window_size):
            return _cy.window_recompute_loop(
                values, lambda b: talib.SMA(b, window_size), window_size)

    def RollingMean__compiled_pandas(values, window_size):
        return _cy.window_recompute_loop(
            values, lambda b: pd.Series(b).mean(), window_size)

    def RollingMax__compiled_screamer(values, window_size):
        return _cy.screamer_loop(values, sc.RollingMax(window_size))

    def RollingMax__compiled_numpy(values, window_size):
        return _cy.window_recompute_loop(values, np.max, window_size)


def all():
    """Metadata table (func, lib, callable) built by introspection."""
    rows = []
    for name, _ in inspect.getmembers(sys.modules[__name__], inspect.isfunction):
        if "__" not in name or name.startswith("_"):
            continue
        func, lib = name.split("__", 1)
        rows.append({"func": func, "lib": lib, "callable": name})
    return pd.DataFrame(rows)
