import numpy as np
import pytest

from screamer import Input, Dag, RollingMean
from screamer.streams import Select, CombineLatest
from tests._dag_oracle import lazy_batch as _lazy_batch


def test_select_column_from_combine_latest():
    ak = np.array([1, 2, 3], dtype=np.int64); av = np.array([10.0, 20.0, 30.0])
    bk = np.array([1, 2, 3], dtype=np.int64); bv = np.array([1.0, 2.0, 3.0])
    a, b = Input("a"), Input("b")
    # CombineLatest()(a, b) is width-2; select column 0 -> a's latest.
    dag = Dag(inputs=[a, b], outputs=[Select(0)(CombineLatest()(a, b))])
    bv_, bk_ = dag((av, ak), (bv, bk))     # (values, index) feeds; values-first result
    sv_, sk_ = _lazy_batch(dag, (av, ak), (bv, bk))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)
    # eager oracle (values-first): align, then project – regression-proof against
    # CombineLatest emit changes (no hand-counted row expectations).
    cv, ck = CombineLatest()(av, bv, index=[ak, bk])
    ev, ek = Select(0)(cv, index=ck)
    np.testing.assert_array_equal(bk_, ek)
    np.testing.assert_array_equal(bv_.reshape(-1), ev)


def test_select_two_columns_reorder():
    ak = np.array([1, 2], dtype=np.int64); av = np.array([10.0, 20.0])
    bk = np.array([1, 2], dtype=np.int64); bv = np.array([1.0, 2.0])
    a, b = Input("a"), Input("b")
    dag = Dag(inputs=[a, b], outputs=[Select([1, 0])(CombineLatest()(a, b))])
    bv_, bk_ = dag((av, ak), (bv, bk))
    sv_, sk_ = _lazy_batch(dag, (av, ak), (bv, bk))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)
    # [1,0] swaps columns; compare against the eager oracle.
    cv, ck = CombineLatest()(av, bv, index=[ak, bk])
    ev, ek = Select([1, 0])(cv, index=ck)
    np.testing.assert_array_equal(bk_, ek)
    np.testing.assert_array_equal(bv_, ev)


def test_select_feeds_functor():
    ak = np.array([1, 2, 3], dtype=np.int64); av = np.array([2.0, 4.0, 6.0])
    bk = np.array([1, 2, 3], dtype=np.int64); bv = np.array([0.0, 0.0, 0.0])
    a, b = Input("a"), Input("b")
    # select a's column then smooth it
    dag = Dag(inputs=[a, b], outputs=[RollingMean(2)(Select(0)(CombineLatest()(a, b)))])
    bv_, bk_ = dag((av, ak), (bv, bk))
    sv_, sk_ = _lazy_batch(dag, (av, ak), (bv, bk))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)
    # value oracle: same graph computed eagerly (align -> select col 0 -> RollingMean(2))
    cv, ck = CombineLatest()(av, bv, index=[ak, bk])
    sel, _ = Select(0)(cv, index=ck)
    ev = RollingMean(2)(sel)
    np.testing.assert_array_equal(bv_.reshape(-1), ev)


def test_select_out_of_range_errors_batch_and_stream():
    ak = np.array([1], dtype=np.int64); av = np.array([10.0])
    bk = np.array([1], dtype=np.int64); bv = np.array([1.0])
    a, b = Input("a"), Input("b")
    # CombineLatest()(a, b) is width-2; column 5 is out of range.
    dag = Dag(inputs=[a, b], outputs=[Select(5)(CombineLatest()(a, b))])
    with pytest.raises(RuntimeError, match="out of range"):
        dag((av, ak), (bv, bk))
    with pytest.raises(RuntimeError, match="out of range"):
        _lazy_batch(dag, (av, ak), (bv, bk))


def test_select_missing_columns_raises_dag():
    """columns is a required positional argument; omitting it raises TypeError."""
    with pytest.raises(TypeError):
        Select()(np.array([[1.0, 2.0], [3.0, 4.0]]))
