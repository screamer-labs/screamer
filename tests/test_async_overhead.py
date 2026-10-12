"""The async path adds a bounded cost per event over the raw async generator.

The cost is the per-event coroutine plus one operator step. The bound is
relative to the raw generator rate (measured in the same run) and generous, so
a loaded machine does not fail it; it catches an order-of-magnitude regression
such as a rebuilt coroutine module or a per-event import.
"""
import asyncio
import time

import pytest

import screamer as sm

N = 20000


async def _gen(n):
    for i in range(n):
        yield float(i)


async def _drain(aiter):
    t = time.perf_counter()
    async for _ in aiter:
        pass
    return (time.perf_counter() - t) / N


def _best(make, reps=5):
    async def run():
        return min([await _drain(make()) for _ in range(reps)])
    return asyncio.run(run())


async def _gen2(n):
    # The raw cost of awaiting two sources per event, as Add() does.
    a, b = _gen(n), _gen(n)
    while True:
        try:
            x, y = await a.__anext__(), await b.__anext__()
        except StopAsyncIteration:
            return
        yield x + y


@pytest.mark.perf
@pytest.mark.parametrize("make, baseline", [
    (lambda: sm.RollingMean(5)(_gen(N)), lambda: _gen(N)),
    (lambda: sm.Add()(_gen(N), _gen(N)), lambda: _gen2(N)),
], ids=["RollingMean", "Add"])
def test_async_overhead_per_event(make, baseline):
    raw = _best(baseline)
    cost = _best(make)
    assert cost - raw < 2e-6
