"""Container/rank dispatch (Rule A): a length-1 array is a time series of one and
returns an array; only an actual scalar returns a scalar. Length is irrelevant to
rank. Regression guard for the length-1 collapse bug across all functor arities."""

import numpy as np
import pytest
from screamer import CumSum, RollingMean, BOP, Add, Cart2Polar


def test_1i_length1_array_returns_array():
    out = CumSum()(np.array([4.0]))
    assert isinstance(out, np.ndarray) and out.shape == (1,)
    np.testing.assert_array_equal(out, [4.0])


def test_1i_scalar_returns_scalar():
    out = CumSum()(4.0)
    assert isinstance(out, float) and out == 4.0


def test_1i_empty_array_returns_empty_array():
    # size 0 used to over-read data[0]; must return an empty array now.
    out = RollingMean(2)(np.array([]))
    assert isinstance(out, np.ndarray) and out.shape == (0,)


def test_1i_length1_matches_length2_prefix():
    # The single-element result must equal the first element of the batch result
    # (same warmup), proving it is a genuine length-1 series, not a scalar.
    one = RollingMean(2)(np.array([5.0]))
    two = RollingMean(2)(np.array([5.0, 6.0]))
    assert one.shape == (1,)
    np.testing.assert_array_equal(one, two[:1])


def test_Ni_length1_arrays_return_array():
    out = BOP()(np.array([1.0]), np.array([2.0]), np.array([0.5]), np.array([1.5]))
    assert isinstance(out, np.ndarray) and out.shape == (1,)
    assert out[0] == pytest.approx((1.5 - 1.0) / (2.0 - 0.5))   # (close-open)/(high-low)


def test_Ni_scalars_return_scalar():
    out = BOP()(1.0, 2.0, 0.5, 1.5)
    assert isinstance(out, float) and out == pytest.approx((1.5 - 1.0) / (2.0 - 0.5))


def test_Ni_1o_add_length1():
    out = Add()(np.array([1.0]), np.array([2.0]))
    assert isinstance(out, np.ndarray) and out.shape == (1,)
    np.testing.assert_array_equal(out, [3.0])


def test_Ni_Mo_length1_returns_array_row():
    # 2-in / 2-out: length-1 inputs -> a (1, 2) array, not a scalar/collapsed row.
    out = Cart2Polar()(np.array([1.0]), np.array([1.0]))
    out = np.asarray(out)
    assert out.shape == (1, 2)


def test_2d_matrix_form_unchanged():
    # The vector-core matrix form is untouched: (T, N) -> (T,), time on axis 0.
    out = BOP()(np.array([[1.0, 2, 0.5, 1.5], [2, 3, 1, 2.5], [3, 4, 2, 3.5]]))
    assert np.asarray(out).shape == (3,)


def test_1i_zero_d_array_returns_scalar():
    # Rank 0 (a 0-d array) has no time axis: one sample -> scalar, like a Python
    # scalar. Distinct from a length-1 array (rank 1 -> array).
    out = CumSum()(np.array(4.0))
    assert isinstance(out, float) and out == 4.0


def test_Ni_zero_d_arrays_return_scalar():
    out = BOP()(np.array(1.0), np.array(2.0), np.array(0.5), np.array(1.5))
    assert isinstance(out, float) and out == pytest.approx((1.5 - 1.0) / (2.0 - 0.5))
