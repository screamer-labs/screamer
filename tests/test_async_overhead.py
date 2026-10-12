"""The async path adds a bounded cost per event over the raw async generator.

The cost is the per-event coroutine plus one operator step. The bound is
relative to the raw generator rate (measured in the same run) and generous, so
a loaded machine does not fail it; it catches an order-of-magnitude regression
such as a per-event copy of the items or a rebuilt coroutine module.
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


@pytest.mark.perf
@pytest.mark.parametrize("make", [
    lambda: sm.RollingMean(5)(_gen(N)),
    lambda: sm.Add()(_gen(N), _gen(N)),
], ids=["RollingMean", "Add"])
def test_async_overhead_per_event(make):
    raw = _best(lambda: _gen(N))
    cost = _best(make)
    assert cost < 20 * raw + 5e-6
