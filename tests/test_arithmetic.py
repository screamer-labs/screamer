import numpy as np
from screamer import Add, Sub, Mul, Div, combine_latest


def test_arithmetic_scalar():
    assert Add()(2.0, 3.0) == 5.0
    assert Sub()(5.0, 2.0) == 3.0
    assert Mul()(4.0, 2.0) == 8.0
    assert Div()(6.0, 3.0) == 2.0


def test_arithmetic_arrays():
    a = np.arange(1.0, 6.0)
    b = np.arange(10.0, 15.0)
    np.testing.assert_array_equal(Add()(a, b), a + b)
    np.testing.assert_array_equal(Sub()(a, b), a - b)
    np.testing.assert_array_equal(Mul()(a, b), a * b)
    np.testing.assert_array_equal(Div()(a, b), a / b)


def test_sub_over_aligned_columns_TxN():
    # the graph spread idiom: align two series, then a C++ Sub over the columns.
    a_k = np.array([1, 2, 3], dtype=np.int64)
    a_v = np.array([10.0, 30.0, 50.0])
    b_k = np.array([1, 2, 3], dtype=np.int64)
    b_v = np.array([1.0, 2.0, 3.0])
    aligned, _ = combine_latest(a_v, b_v, index=[a_k, b_k])  # (T, 2)
    spread = Sub()(aligned)                        # (T,N) convention -> columns as inputs
    np.testing.assert_array_equal(spread, aligned[:, 0] - aligned[:, 1])
