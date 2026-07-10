import numpy as np
import pytest
from screamer import RollingMean, Diff, Sub, Add, Input, Dag, combine_latest
from screamer.streams import Stream
from tests._dag_oracle import run_oracle, lazy_batch as _lazy_batch


def _row(v):
    v = np.ascontiguousarray(v, dtype=np.float64)
    return v, np.arange(v.size, dtype=np.int64)   # (values, index) - values-first


def _series(size, seed):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.integers(0, size * 3, size=size)).astype(np.int64)
    vals = rng.standard_normal(size)
    return vals, idx   # (values, index) - values-first


def _chain():
    x = Input("x")
    return Dag(inputs=[x], outputs=[Diff(1)(RollingMean(5)(x))]), [_row(np.random.default_rng(0).standard_normal(150))]


def _fanout():
    x = Input("x")
    s = RollingMean(5)(x)
    return Dag(inputs=[x], outputs=[Diff(1)(s), RollingMean(3)(s)]), [_row(np.random.default_rng(1).standard_normal(150))]


def _combine():
    a, b = Input("a"), Input("b")
    z = RollingMean(4)(Sub()(combine_latest(a, b)))
    return Dag(inputs=[a, b], outputs=[z]), [_series(120, 2), _series(120, 3)]


def _divergent():
    a, b, c = Input("a"), Input("b"), Input("c")
    ab = Sub()(combine_latest(a, b))
    ac = Add()(combine_latest(a, c))
    dag = Dag(inputs=[a, b, c], outputs=[ab, ac], align_outputs=True)
    return dag, [_series(100, 5), _series(80, 6), _series(80, 7)]


def _chain_stream_feed():
    """Same graph as _chain but fed via Stream objects."""
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(80)
    idx = np.sort(rng.integers(0, 300, size=80)).astype(np.int64)
    stream_feed = Stream(vals, idx)
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[Diff(1)(RollingMean(5)(x))])
    return dag, [stream_feed], (vals, idx)


def _chain_values_index_feed():
    """Same graph as _chain but fed via (values, index) pairs (values-first)."""
    rng = np.random.default_rng(7)
    vals = rng.standard_normal(60)
    idx = np.sort(rng.integers(0, 200, size=60)).astype(np.int64)
    vi_feed = (vals, idx)   # values-first pair
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[Diff(1)(RollingMean(5)(x))])
    return dag, [vi_feed], (vals, idx)


def _to_pairs(result):
    """Normalize dag output to a list of (values, index) pairs.

    Single-output dags return (values, index); multi-output dags return a tuple
    of (values, index) pairs. Distinguish by checking the first element type.
    """
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], np.ndarray):
        return [result]
    return list(result)


@pytest.mark.parametrize("factory", [_chain, _fanout, _combine, _divergent])
def test_batch_equals_oracle(factory):
    dag, feeds = factory()
    got = _to_pairs(dag(*feeds))
    exp = _to_pairs(run_oracle(dag, {nm: f for nm, f in zip(dag._names, feeds)}))
    for (gv, gi), (ev, ei) in zip(got, exp):
        np.testing.assert_array_equal(gv, ev)
        np.testing.assert_array_equal(gi, ei)


@pytest.mark.parametrize("factory", [_chain, _fanout, _combine, _divergent])
def test_lazy_equals_batch(factory):
    # Includes _divergent (two outputs from three series with divergent indices):
    # the lazy drain forward-fills each output's latest value across drains, so its
    # multi-output when_all alignment matches batch combine_latest exactly.
    dag, feeds = factory()
    batch = _to_pairs(dag(*feeds))
    lazy = _to_pairs(_lazy_batch(dag, *feeds))
    for (bv, bi), (sv, si) in zip(batch, lazy):
        np.testing.assert_array_equal(bv, sv)
        np.testing.assert_array_equal(bi, si)


def test_stream_feed_matches_array_feed():
    """Dag fed with Stream objects gives the same result as bare (values, index) pairs."""
    dag, stream_feeds, (vals, idx) = _chain_stream_feed()
    array_feed = (vals, idx)   # (values, index) pair
    result_stream = _to_pairs(dag(*stream_feeds))
    result_array = _to_pairs(dag(array_feed))
    for (sv, si), (av, ai) in zip(result_stream, result_array):
        np.testing.assert_array_equal(sv, av)
        np.testing.assert_array_equal(si, ai)


def test_values_index_feed_matches_array_feed():
    """Dag fed with (values, index) pairs gives the same result as a bare Stream."""
    dag, vi_feeds, (vals, idx) = _chain_values_index_feed()
    result_vi = _to_pairs(dag(*vi_feeds))
    # cross-check: lazy path also matches
    result_vi_lazy = _to_pairs(_lazy_batch(dag, *vi_feeds))
    for (bv, bi), (sv, si) in zip(result_vi, result_vi_lazy):
        np.testing.assert_array_equal(bv, sv)
        np.testing.assert_array_equal(bi, si)
