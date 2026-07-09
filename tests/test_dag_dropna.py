import numpy as np
import pytest

from screamer import Input, Dag
from screamer.streams import dropna


def _lazy_batch(dag_obj, *feeds):
    """Run dag via the lazy iterator path (generators), return in batch format.

    Each feed is a (values_arr, keys_arr) pair (values-first). Returns result in
    the same format as dag(*feeds): a (values, index) pair for single-output dags,
    or a tuple of such pairs for multi-output dags (align_outputs=True only).
    """
    def _gen(v_arr, k_arr):
        return ((float(v), int(k)) for v, k in zip(v_arr, k_arr))

    n_out = len(dag_obj.outputs)
    gen_feeds = [_gen(v_arr, k_arr) for v_arr, k_arr in feeds]
    events = list(dag_obj(*gen_feeds))
    if n_out == 1:
        if not events:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
        sv = np.array([e[0] for e in events], dtype=np.float64)
        sk = np.array([e[1] for e in events], dtype=np.int64)
        return sv, sk
    if not events:
        empty = np.array([], dtype=np.float64)
        return tuple((empty, np.array([], dtype=np.int64)) for _ in range(n_out))
    sk = np.array([e[-1] for e in events], dtype=np.int64)
    return tuple((np.array([e[i] for e in events], dtype=np.float64), sk)
                 for i in range(n_out))


def _run_modes(dag, feed):
    """Return (batch, lazy) results as (values, index) for a single-output dag."""
    bv, bk = dag(feed)
    sv, sk = _lazy_batch(dag, feed)
    return (bv, bk), (sv, sk)


def test_dropna_graph_matches_eager_any():
    keys = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    vals = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[dropna(x)])
    (bv, bk), (sv, sk) = _run_modes(dag, (vals, keys))    # (values, index) feed
    ev, ek = dropna(vals, index=keys)          # eager oracle (values-first)
    np.testing.assert_array_equal(bk, ek)
    np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek)
    np.testing.assert_array_equal(sv.reshape(-1), ev)


def test_dropna_graph_all_dropped():
    keys = np.array([1, 2], dtype=np.int64)
    vals = np.array([np.nan, np.nan])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[dropna(x)])
    bv, bk = dag((vals, keys))
    assert len(bk) == 0


def test_dropna_graph_none_dropped():
    keys = np.array([1, 2, 3], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[dropna(x)])
    bv, bk = dag((vals, keys))
    np.testing.assert_array_equal(bk, keys)
    np.testing.assert_array_equal(bv.reshape(-1), vals)


def test_dropna_before_functor():
    from screamer import RollingMean
    keys = np.array([1, 2, 3, 4], dtype=np.int64)
    vals = np.array([2.0, np.nan, 4.0, 6.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[RollingMean(2)(dropna(x))])
    bv, bk = dag((vals, keys))
    sv, sk = _lazy_batch(dag, (vals, keys))
    np.testing.assert_array_equal(bk, sk)
    np.testing.assert_array_equal(bv, sv)
    # dropna removes the NaN row, leaving keys [1,3,4]; RollingMean(2) over [2,4,6]
    np.testing.assert_array_equal(bk, [1, 3, 4])


def test_filter_rejected_in_graph():
    x = Input("x")
    from screamer.streams import filter as sfilter
    with pytest.raises(ValueError, match="not supported"):
        sfilter(x, lambda r: r > 0)


def test_dropna_graph_bad_how_rejected():
    # graph form validates `how` up front, same as the eager form
    x = Input("x")
    with pytest.raises(ValueError, match="how must be"):
        dropna(x, how="typo")


def test_dropna_all_over_wide_combine_latest():
    # dropna(how="all") on a WIDE (width-2) stream produced by combine_latest:
    # exercises the width-passthrough node_width path and the multi-value NaN
    # scan in-graph. Assert graph == eager oracle AND batch == stream, without
    # hand-predicting combine_latest's aligned output.
    from screamer.streams import combine_latest
    ak = np.array([1, 2, 3], dtype=np.int64); av = np.array([np.nan, 2.0, np.nan])
    bk = np.array([1, 2, 3], dtype=np.int64); bv = np.array([np.nan, np.nan, 3.0])
    a, b = Input("a"), Input("b")
    dag = Dag(inputs=[a, b], outputs=[dropna(combine_latest(a, b), how="all")])
    bv_, bk_ = dag((av, ak), (bv, bk))
    sv_, sk_ = _lazy_batch(dag, (av, ak), (bv, bk))
    # eager oracle: align (values-first), then drop all-NaN rows (values-first)
    cv, ck = combine_latest(av, bv, index=[ak, bk])
    ev, ek = dropna(cv, index=ck, how="all")
    np.testing.assert_array_equal(bk_, ek)
    np.testing.assert_array_equal(bv_, ev)
    np.testing.assert_array_equal(sk_, ek)
    np.testing.assert_array_equal(sv_, ev)


def test_dropna_fanout_to_two_consumers():
    # dropna output feeds TWO downstream functors (Broadcast fan-out from a
    # cardinality-reducing upstream). Verify batch result correctness.
    # Note: the lazy path (dag(generators)) does not yet support align_outputs=False
    # with multiple outputs; batch-only assertions are used here.
    from screamer import RollingMean, Lag
    keys = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    vals = np.array([2.0, np.nan, 4.0, 6.0, np.nan])
    x = Input("x")
    d = dropna(x)
    dag = Dag(inputs=[x], outputs=[RollingMean(2)(d), Lag(1)(d)], align_outputs=False)
    (bm, bl) = dag((vals, keys))
    # each pair is (values, index)
    # dropna removes the two NaN rows -> surviving keys [1,3,4]
    np.testing.assert_array_equal(bm[1], [1, 3, 4])
    np.testing.assert_array_equal(bl[1], [1, 3, 4])
