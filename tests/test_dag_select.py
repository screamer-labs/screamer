import numpy as np
import pytest

from screamer import Input, Dag, RollingMean
from screamer.streams import select, combine_latest


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


def test_select_column_from_combine_latest():
    ak = np.array([1, 2, 3], dtype=np.int64); av = np.array([10.0, 20.0, 30.0])
    bk = np.array([1, 2, 3], dtype=np.int64); bv = np.array([1.0, 2.0, 3.0])
    a, b = Input("a"), Input("b")
    # combine_latest(a, b) is width-2; select column 0 -> a's latest.
    dag = Dag(inputs=[a, b], outputs=[select(combine_latest(a, b), 0)])
    bv_, bk_ = dag((av, ak), (bv, bk))     # (values, index) feeds; values-first result
    sv_, sk_ = _lazy_batch(dag, (av, ak), (bv, bk))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)
    # eager oracle (values-first): align, then project – regression-proof against
    # combine_latest emit changes (no hand-counted row expectations).
    cv, ck = combine_latest(av, bv, index=[ak, bk])
    ev, ek = select(cv, 0, index=ck)
    np.testing.assert_array_equal(bk_, ek)
    np.testing.assert_array_equal(bv_.reshape(-1), ev)


def test_select_two_columns_reorder():
    ak = np.array([1, 2], dtype=np.int64); av = np.array([10.0, 20.0])
    bk = np.array([1, 2], dtype=np.int64); bv = np.array([1.0, 2.0])
    a, b = Input("a"), Input("b")
    dag = Dag(inputs=[a, b], outputs=[select(combine_latest(a, b), [1, 0])])
    bv_, bk_ = dag((av, ak), (bv, bk))
    sv_, sk_ = _lazy_batch(dag, (av, ak), (bv, bk))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)
    # [1,0] swaps columns; compare against the eager oracle.
    cv, ck = combine_latest(av, bv, index=[ak, bk])
    ev, ek = select(cv, [1, 0], index=ck)
    np.testing.assert_array_equal(bk_, ek)
    np.testing.assert_array_equal(bv_, ev)


def test_select_feeds_functor():
    ak = np.array([1, 2, 3], dtype=np.int64); av = np.array([2.0, 4.0, 6.0])
    bk = np.array([1, 2, 3], dtype=np.int64); bv = np.array([0.0, 0.0, 0.0])
    a, b = Input("a"), Input("b")
    # select a's column then smooth it
    dag = Dag(inputs=[a, b], outputs=[RollingMean(2)(select(combine_latest(a, b), 0))])
    bv_, bk_ = dag((av, ak), (bv, bk))
    sv_, sk_ = _lazy_batch(dag, (av, ak), (bv, bk))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)
    # value oracle: same graph computed eagerly (align -> select col 0 -> RollingMean(2))
    cv, ck = combine_latest(av, bv, index=[ak, bk])
    sel, _ = select(cv, 0, index=ck)
    ev = RollingMean(2)(sel)
    np.testing.assert_array_equal(bv_.reshape(-1), ev)


def test_select_out_of_range_errors_batch_and_stream():
    ak = np.array([1], dtype=np.int64); av = np.array([10.0])
    bk = np.array([1], dtype=np.int64); bv = np.array([1.0])
    a, b = Input("a"), Input("b")
    # combine_latest(a, b) is width-2; column 5 is out of range.
    dag = Dag(inputs=[a, b], outputs=[select(combine_latest(a, b), 5)])
    with pytest.raises(RuntimeError, match="out of range"):
        dag((av, ak), (bv, bk))
    with pytest.raises(RuntimeError, match="out of range"):
        _lazy_batch(dag, (av, ak), (bv, bk))


def test_select_missing_columns_raises_dag():
    """columns is a required positional argument; omitting it raises TypeError."""
    with pytest.raises(TypeError):
        select(np.array([[1.0, 2.0], [3.0, 4.0]]))
