import numpy as np
import pytest

from screamer import Input, Dag, RollingMean
from screamer.streams import select, combine_latest


def test_select_column_from_combine_latest():
    ak = np.array([1, 2, 3], dtype=np.int64); av = np.array([10.0, 20.0, 30.0])
    bk = np.array([1, 2, 3], dtype=np.int64); bv = np.array([1.0, 2.0, 3.0])
    a, b = Input("a"), Input("b")
    # combine_latest(a, b) is width-2; select column 0 -> a's latest.
    # With equal-keyed inputs combine_latest emits at every event once warmed up:
    # key=1 (b fires → 1 row), key=2 (a then b → 2 rows), key=3 (a then b → 2 rows) = 5 rows.
    dag = Dag(inputs=[a, b], outputs=[select(combine_latest(a, b), 0)])
    bk_, bv_ = dag((ak, av), (bk, bv))
    sk_, sv_ = dag.stream((ak, av), (bk, bv))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)
    # column 0 always carries a's latest value at each emitted row
    np.testing.assert_array_equal(bv_.reshape(-1), [10.0, 20.0, 20.0, 30.0, 30.0])


def test_select_two_columns_reorder():
    ak = np.array([1, 2], dtype=np.int64); av = np.array([10.0, 20.0])
    bk = np.array([1, 2], dtype=np.int64); bv = np.array([1.0, 2.0])
    a, b = Input("a"), Input("b")
    dag = Dag(inputs=[a, b], outputs=[select(combine_latest(a, b), [1, 0])])
    bk_, bv_ = dag((ak, av), (bk, bv))
    sk_, sv_ = dag.stream((ak, av), (bk, bv))
    np.testing.assert_array_equal(bv_, sv_)
    # [1,0] swaps -> column0 = b, column1 = a.
    # With equal-keyed inputs: key=1 yields 1 row (b fires first), key=2 yields 2 rows = 3 total.
    np.testing.assert_array_equal(bv_, [[1.0, 10.0], [1.0, 20.0], [2.0, 20.0]])


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


def test_select_out_of_range_errors():
    ak = np.array([1], dtype=np.int64); av = np.array([10.0])
    bk = np.array([1], dtype=np.int64); bv = np.array([1.0])
    a, b = Input("a"), Input("b")
    dag = Dag(inputs=[a, b], outputs=[select(combine_latest(a, b), 5)])
    with pytest.raises(Exception):
        dag((ak, av), (bk, bv))
