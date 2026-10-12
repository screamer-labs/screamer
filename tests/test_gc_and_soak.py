"""Garbage-collector support for streams, and a soak test (tracker row C5).

A stream stored on the object that feeds it forms a reference cycle. The
iterator types implement tp_traverse and tp_clear, so the collector frees the
cycle. The soak test creates and drops streams, pipelines and live sessions
many times and checks that memory stays flat.
"""
import asyncio
import ctypes
import gc
import resource
import subprocess
import sys
import textwrap
import weakref

import numpy as np
import pytest

import screamer as s
from screamer import screamer_bindings as _b


class Src:
    def __init__(self, xs): self.xs = iter(xs); self.it = None
    def __iter__(self): return self
    def __next__(self): return next(self.xs)


def test_cycle_through_sync_stream_is_collected():
    src = Src([1.0, 2.0, 3.0]); src.it = s.RollingMean(2)(src); next(src.it)
    r = weakref.ref(src); del src; gc.collect()
    assert r() is None


def test_cycle_through_async_stream_is_collected():
    class ASrc:
        def __init__(self): self.it = None
        def __aiter__(self): return self
        async def __anext__(self): return 1.0
    src = ASrc(); src.it = s.RollingMean(2)(src)
    r = weakref.ref(src); del src; gc.collect()
    assert r() is None


def test_no_nanobind_leak_report_at_exit():
    code = textwrap.dedent("""
        import screamer as s
        class Src:
            def __init__(self): self.xs = iter([1.0, 2.0]); self.it = None
            def __iter__(self): return self
            def __next__(self): return next(self.xs)
        src = Src(); src.it = s.RollingMean(2)(src); next(src.it)
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "leaked" not in p.stderr, p.stderr


# ---- the other classes that hold Python objects -----------------------------

def test_cycle_through_row_stream_is_collected():
    src = Src([np.arange(4.0), np.arange(4.0)]); src.it = s.CrossRank()(src); next(src.it)
    r = weakref.ref(src); del src; gc.collect()
    assert r() is None


def test_cycle_through_asset_stream_is_collected():
    # Array events: the stepper holds per-asset operator copies and the event arrays.
    src = Src([np.arange(3.0), np.arange(3.0)]); src.it = s.RollingMean(2)(src); next(src.it)
    r = weakref.ref(src); del src; gc.collect()
    assert r() is None


def test_cycle_through_operator_subclass_is_collected():
    # The operator, not the source, holds the stream.
    class Mean(s.RollingMean):
        pass
    op = Mean(2)
    op.it = op(iter([1.0, 2.0]))
    next(op.it)
    r = weakref.ref(op); del op; gc.collect()
    assert r() is None


def test_cycle_through_lazy_pipeline_is_collected():
    a = s.Input("a"); p = s.Pipeline([a], [s.RollingMean(2)(a)])
    src = Src([(1.0, 0), (2.0, 1), (3.0, 2)]); src.it = p(src); next(src.it)
    r = weakref.ref(src); del src; gc.collect()
    assert r() is None


def test_cycle_through_lazy_merge_is_collected():
    src = Src([1.0, 2.0, 3.0]); src.it = s.Merge()(src, iter([1.0, 2.0])); next(src.it)
    r = weakref.ref(src); del src; gc.collect()
    assert r() is None


# ---- after tp_clear, the iterators end cleanly -------------------------------

def _tp_clear(obj):
    """Call the type's tp_clear slot, as the collector does on a cycle."""
    get = ctypes.pythonapi.PyType_GetSlot
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_int]
    fn = get(type(obj), 51)                  # Py_tp_clear
    assert fn, f"{type(obj).__name__} has no tp_clear"
    assert ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object)(fn)(obj) == 0   # holds the GIL


def test_sync_stream_after_clear_stops():
    it = s.RollingMean(2)(iter([1.0, 2.0, 3.0])); next(it)
    _tp_clear(it)
    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_async_stream_after_clear_stops():
    async def agen():
        for x in (1.0, 2.0, 3.0):
            yield x
    it = s.RollingMean(2)(agen())

    async def go():
        await it.__anext__()
        _tp_clear(it)
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()
    asyncio.run(go())


def test_async_step_after_clear_raises():
    async def agen():
        yield 1.0
    it = s.RollingMean(2)(agen())
    _tp_clear(it)
    with pytest.raises(StopAsyncIteration):
        it._step([1.0])


def test_lazy_pipeline_after_clear_stops():
    a = s.Input("a"); p = s.Pipeline([a], [s.RollingMean(2)(a)])
    it = p(iter([(1.0, 0), (2.0, 1), (3.0, 2)])); next(it)
    _tp_clear(it)
    with pytest.raises(StopIteration):
        next(it)


def test_merge_puller_after_clear_ends():
    puller = _b._MergeLazyPuller_i64([iter([1.0, 2.0]), iter([3.0])], True)
    assert puller.next() is not None
    _tp_clear(puller)
    assert puller.next() is None


# ---- soak --------------------------------------------------------------------

def _max_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024    # Linux reports KB


def test_soak_memory_is_flat():
    import tracemalloc
    def cycle():
        for _ in range(200):
            list(s.RollingMean(5)(iter(np.arange(50.))))
            list(s.Add()(iter(np.arange(50.)), iter(np.arange(50.))))
            s.CrossRank()(np.random.rand(20, 5))
            a = s.Input("a"); p = s.Pipeline([a], [s.RollingMean(3)(a)])
            live = p.live()
            for t in range(20): live.push("a", t, float(t))
            live.result()
        gc.collect()
    cycle(); tracemalloc.start(); base = tracemalloc.get_traced_memory()[0]
    for _ in range(5): cycle()
    grown = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.stop()
    assert grown < 256 * 1024, grown

    # tracemalloc sees only Python allocations; the peak RSS also catches
    # memory the C++ side never frees.
    rss0 = _max_rss_bytes()
    for _ in range(20): cycle()
    rss_grown = _max_rss_bytes() - rss0
    assert rss_grown < 5 * 1024 * 1024, rss_grown
