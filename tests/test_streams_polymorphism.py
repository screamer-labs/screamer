"""The multi-stream operators (merge, split) accept tuple inputs in
addition to raw arrays, matching combine_latest and the single-input operators.
Raw and tuple inputs must produce identical results. Lazy (generator) inputs
dispatch to the k-way merge path and yield identical events to the batch oracle.
"""
import asyncio

import numpy as np
import pytest

from screamer import Merge, split, Input


def _drain(agen):
    async def run():
        return [e async for e in agen]
    return asyncio.run(run())


AV, AK = np.array([1.0, 2.0, 3.0]), np.array([10, 20, 30])
BV, BK = np.array([4.0, 5.0]), np.array([15, 25])


def test_merge_stream_equals_raw():
    raw = Merge()(AV, BV, index=[AK, BK])
    strm = Merge()((AV, AK), (BV, BK))
    for x, y in zip(raw, strm):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_merge_mixes_stream_and_raw():
    raw = Merge()(AV, BV, index=[AK, BK])
    mixed = Merge()((AV, AK), BV, index=[None, BK])
    for x, y in zip(raw, mixed):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_merge_lazy_generators_equal_batch():
    """Lazy Merge()(*generators) gives identical events to the batch merge oracle."""
    bv, bs, bi = Merge()(AV, BV, index=[AK, BK])
    ga = ((float(v), int(k)) for v, k in zip(AV, AK))
    gb = ((float(v), int(k)) for v, k in zip(BV, BK))
    events = list(Merge()(ga, gb))
    np.testing.assert_allclose([e[0] for e in events], bv)
    np.testing.assert_array_equal([e[1] for e in events], bi)
    np.testing.assert_array_equal([e[2] for e in events], bs)



def test_positional_streams_independent_lengths():
    # merge's positional mode allows unequal lengths (unlike combine_latest).
    out = Merge()(AV, BV)          # no index -> row-number ordering
    assert out[2] is None                        # positional -> index None
    np.testing.assert_array_equal(np.sort(out[0]), np.sort(np.concatenate([AV, BV])))


def test_split_tuple_returns_tuples():
    mv, ms, mk = Merge()(AV, BV, index=[AK, BK])
    raw_parts = split(mv, ms, index=mk)
    tuple_parts = split((mv, mk), ms)
    assert all(isinstance(p, tuple) for p in raw_parts)
    assert all(isinstance(p, tuple) and isinstance(p[0], np.ndarray) for p in tuple_parts)
    # reconstructs the original inputs
    np.testing.assert_array_equal(tuple_parts[0][0], AV)
    np.testing.assert_array_equal(tuple_parts[0][1], AK)
    np.testing.assert_array_equal(tuple_parts[1][0], BV)


def test_split_tuple_positional_index_none():
    mv, ms, _ = Merge()(AV, BV)                     # positional merge -> index None
    parts = split((mv, None), ms)
    assert all(isinstance(p, tuple) for p in parts)
    assert all(p[1] is None for p in parts)


def test_merge_roundtrips_through_split_streams():
    mv, sources, mk = Merge()(AV, BV, index=[AK, BK])
    merged = (mv, mk)
    parts = split(merged, sources)
    np.testing.assert_array_equal(parts[0][0], AV)
    np.testing.assert_array_equal(parts[1][0], BV)


def test_node_input_raises_clear_error():
    a, b = Input("a"), Input("b")
    with pytest.raises(ValueError, match="not supported as a DAG graph node"):
        Merge()(a, b)
