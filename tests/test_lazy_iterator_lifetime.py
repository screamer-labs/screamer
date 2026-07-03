"""Lifetime safety for the lazy (iterator / async-generator) call paths.

A functor applied to an iterable returns a lazy iterator that pulls one value
at a time. If the functor is a *transient* — e.g. ``RollingMean(5)(gen())`` where
nothing else holds a reference to the ``RollingMean(5)`` instance — the lazy
iterator must keep that functor alive for as long as it is consumed. Otherwise
the functor's Python wrapper is garbage-collected and the iterator dereferences
freed memory (segfault).

These tests force a GC between constructing the lazy iterator and consuming it,
so a dangling-reference regression crashes the test process deterministically.
"""

import asyncio
import gc

import numpy as np

from screamer import RollingMean


def test_transient_functor_over_iterator_survives_gc():
    # Only the returned lazy iterator references the RollingMean instance.
    lazy = RollingMean(5)(iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    # Force-collect the now-unreferenced transient functor before consuming.
    gc.collect()
    result = list(lazy)
    assert len(result) == 8
    # Equal to the batch result over the same data (streaming == batch).
    batch = RollingMean(5)(np.arange(1.0, 9.0))
    np.testing.assert_array_equal(np.array(result), batch)


def test_transient_functor_over_generator_survives_gc():
    def gen():
        for v in range(10):
            yield float(v)

    lazy = RollingMean(3)(gen())
    gc.collect()
    result = list(lazy)
    assert len(result) == 10
    # Values must match the batch result — a live-but-wrong-state regression
    # would keep the length while corrupting the numbers.
    batch = RollingMean(3)(np.arange(10.0))
    np.testing.assert_array_equal(np.array(result), batch)


def test_transient_functor_over_async_generator_survives_gc():
    async def agen():
        for v in range(10):
            yield float(v)

    async def run():
        lazy = RollingMean(3)(agen())
        gc.collect()
        return [x async for x in lazy]

    result = asyncio.run(run())
    assert len(result) == 10
    batch = RollingMean(3)(np.arange(10.0))
    np.testing.assert_array_equal(np.array(result), batch)
