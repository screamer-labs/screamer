import asyncio

import numpy as np
import pytest

from screamer import streams


def _make_series(n_series, size, dtype, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_series):
        if dtype == np.int64:
            k = np.sort(rng.integers(0, size * 4, size=size)).astype(np.int64)
        else:
            k = np.sort(rng.uniform(0, size * 4.0, size=size)).astype(np.float64)
        v = rng.standard_normal(size)
        out.append((k, v))
    return out


CONFIGS = [
    (n_series, dtype)
    for n_series in (2, 3, 5)
    for dtype in (np.int64, np.float64)
]


@pytest.mark.parametrize("n_series,dtype", CONFIGS)
def test_merge_batch_equals_stream(n_series, dtype):
    series = _make_series(n_series, 80, dtype, seed=100 + n_series)
    bk, bv, bs = streams.merge(*series)
    events = list(streams.merge_iter(*series))
    np.testing.assert_array_equal([e[0] for e in events], bk)
    np.testing.assert_array_equal([e[1] for e in events], bv)
    np.testing.assert_array_equal(np.array([e[2] for e in events], dtype=np.uint32), bs)


@pytest.mark.parametrize("n_series,dtype", CONFIGS)
@pytest.mark.parametrize("emit", ["when_all", "on_any"])
def test_combine_latest_batch_equals_stream(n_series, dtype, emit):
    series = _make_series(n_series, 80, dtype, seed=200 + n_series)
    # New API: pass Stream objects (values + index), receive Stream back.
    stream_list = [streams.Stream(v, k) for k, v in series]
    result = streams.combine_latest(*stream_list, emit=emit)
    # result is a Stream (regime="stream"): one row per distinct index (coalesced)
    bk = result.index
    ba = result.values
    events = list(streams.combine_latest_iter(*stream_list, emit=emit))
    # combine_latest_iter yields (row, index) - coalesced, one per distinct index
    got_k = np.array([idx for _, idx in events], dtype=bk.dtype)
    got_a = np.array([np.asarray(row) for row, _ in events],
                     dtype=np.float64).reshape(len(events), n_series)
    np.testing.assert_array_equal(got_k, bk)
    np.testing.assert_array_equal(got_a, ba)


@pytest.mark.parametrize("n_series,dtype", CONFIGS)
def test_pace_infinite_equals_merge(n_series, dtype):
    series = _make_series(n_series, 60, dtype, seed=300 + n_series)
    bk, bv, _ = streams.merge(*series)

    async def drain():
        out = []
        async for e in streams.pace(*series, speed=float("inf")):
            out.append(e)
        return out

    events = asyncio.run(drain())
    np.testing.assert_array_equal([e[0] for e in events], bk)
    np.testing.assert_array_equal([e[1] for e in events], bv)


def test_dropna_batch_equals_stream_on_combine_output():
    # combine_latest on_any produces NaN warmup rows; dropna must remove the
    # same rows whether applied to the batch array or the event stream.
    series = _make_series(3, 50, np.int64, seed=42)
    stream_list = [streams.Stream(v, k) for k, v in series]
    result = streams.combine_latest(*stream_list, emit="on_any")
    bk = result.index
    ba = result.values
    dk, dv = streams.dropna(bk, ba, how="any")

    events = list(streams.combine_latest_iter(*stream_list, emit="on_any"))
    # combine_latest_iter yields (row, index); filter NaN rows inline
    kept = [(row, idx) for row, idx in events
            if not np.any(np.isnan(np.asarray(row, dtype=np.float64)))]
    dk_stream = np.array([idx for _, idx in kept], dtype=bk.dtype)
    dv_stream = np.array([np.asarray(row) for row, _ in kept],
                         dtype=np.float64).reshape(len(kept), -1)
    assert dv_stream.shape[1] == 3
    np.testing.assert_array_equal(dk_stream, dk)
    np.testing.assert_array_equal(dv_stream, dv)
