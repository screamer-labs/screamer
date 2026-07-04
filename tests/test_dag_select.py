import numpy as np
import pytest

from screamer import Input, Dag, RollingMean
from screamer.streams import select, combine_latest


def test_select_column_from_combine_latest():
    ak = np.array([1, 2, 3], dtype=np.int64); av = np.array([10.0, 20.0, 30.0])
    bk = np.array([1, 2, 3], dtype=np.int64); bv = np.array([1.0, 2.0, 3.0])
    a, b = Input("a"), Input("b")
    # combine_latest(a, b) is width-2; select column 0 -> a's latest.
    dag = Dag(inputs=[a, b], outputs=[select(combine_latest(a, b), 0)])
    bk_, bv_ = dag((ak, av), (bk, bv))
    sk_, sv_ = dag.stream((ak, av), (bk, bv))
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
    bk_, bv_ = dag((ak, av), (bk, bv))
    sk_, sv_ = dag.stream((ak, av), (bk, bv))
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
    bk_, bv_ = dag((ak, av), (bk, bv))
    sk_, sv_ = dag.stream((ak, av), (bk, bv))
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
        dag((ak, av), (bk, bv))
    with pytest.raises(RuntimeError, match="out of range"):
        dag.stream((ak, av), (bk, bv))


def test_select_missing_columns_raises():
    """columns is a required positional argument; omitting it raises TypeError."""
    with pytest.raises(TypeError):
        select(np.array([[1.0, 2.0], [3.0, 4.0]]))
