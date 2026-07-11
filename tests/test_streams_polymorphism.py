"""The multi-stream operators (merge, replay, split) accept Stream inputs in
addition to raw arrays, matching combine_latest and the single-input operators.
Raw and Stream inputs must produce identical results. Lazy (generator) inputs
dispatch to the k-way merge path and yield identical events to the batch oracle.
"""
import asyncio

import numpy as np
import pytest

from screamer import Merge, split, Input
from screamer.streams import Stream, replay


def _drain(agen):
    async def run():
        return [e async for e in agen]
    return asyncio.run(run())


AV, AK = np.array([1.0, 2.0, 3.0]), np.array([10, 20, 30])
BV, BK = np.array([4.0, 5.0]), np.array([15, 25])


def test_merge_stream_equals_raw():
    raw = Merge()(AV, BV, index=[AK, BK])
    strm = Merge()(Stream(AV, AK), Stream(BV, BK))
    for x, y in zip(raw, strm):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_merge_mixes_stream_and_raw():
    raw = Merge()(AV, BV, index=[AK, BK])
    mixed = Merge()(Stream(AV, AK), BV, index=[None, BK])
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


def test_replay_stream_equals_raw():
    raw = _drain(replay(AV, BV, index=[AK, BK], speed=float("inf")))
    strm = _drain(replay(Stream(AV, AK), Stream(BV, BK), speed=float("inf")))
    assert raw == strm


def test_positional_streams_independent_lengths():
    # merge's positional mode allows unequal lengths (unlike combine_latest).
    out = Merge()(Stream(AV), Stream(BV))          # no index -> row-number ordering
    assert out[2] is None                        # positional -> index None
    np.testing.assert_array_equal(np.sort(out[0]), np.sort(np.concatenate([AV, BV])))


def test_split_stream_returns_streams():
    mv, ms, mk = Merge()(AV, BV, index=[AK, BK])
    raw_parts = split(mv, ms, index=mk)
    stream_parts = split(Stream(mv, mk), ms)
    assert all(isinstance(p, tuple) for p in raw_parts)
    assert all(isinstance(p, Stream) for p in stream_parts)
    # reconstructs the original inputs
    np.testing.assert_array_equal(stream_parts[0].values, AV)
    np.testing.assert_array_equal(stream_parts[0].index, AK)
    np.testing.assert_array_equal(stream_parts[1].values, BV)


def test_split_stream_positional_index_none():
    mv, ms, _ = Merge()(AV, BV)                     # positional merge -> index None
    parts = split(Stream(mv, None), ms)
    assert all(isinstance(p, Stream) for p in parts)
    assert all(p.index is None for p in parts)


def test_merge_roundtrips_through_split_streams():
    merged = Stream(*[Merge()(AV, BV, index=[AK, BK])[i] for i in (0, 2)])  # (values, index)
    _, sources, _ = Merge()(AV, BV, index=[AK, BK])
    parts = split(merged, sources)
    np.testing.assert_array_equal(parts[0].values, AV)
    np.testing.assert_array_equal(parts[1].values, BV)


@pytest.mark.parametrize("op", ["merge", "replay"])
def test_node_input_raises_clear_error(op):
    a, b = Input("a"), Input("b")
    fn = {"merge": Merge(), "replay": lambda *v: _drain(replay(*v))}[op]
    with pytest.raises(ValueError, match="not supported as a DAG graph node"):
        fn(a, b)
