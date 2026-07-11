"""Tests for logic/comparison operator family: GreaterThan, LessThan,
GreaterEqual, LessEqual, Equal, NotEqual, And, Or, Where, Not, IsNan,
IsFinite.

Truth tables on numpy arrays including NaN inputs.  Binary comparison ops
propagate NaN.  IsNan and IsFinite classify NaN (they do NOT propagate it).
Not propagates NaN.
"""

import math
import numpy as np
import pytest
import screamer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply(op, *arrays):
    """Call op on matching numpy arrays and return the result as a plain list."""
    result = op(*[np.array(a, dtype=float) for a in arrays])
    return list(result)


def apply_scalar(op, *scalars):
    """Call op on individual scalar values (streaming mode)."""
    return op(*[float(s) for s in scalars])


nan = float("nan")
inf = float("inf")


# ---------------------------------------------------------------------------
# GreaterThan
# ---------------------------------------------------------------------------

class TestGreaterThan:
    def test_basic(self):
        result = apply(screamer.GreaterThan(), [1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        assert result == [0.0, 0.0, 1.0]

    def test_all_less(self):
        result = apply(screamer.GreaterThan(), [0.0, 1.0], [5.0, 5.0])
        assert result == [0.0, 0.0]

    def test_all_greater(self):
        result = apply(screamer.GreaterThan(), [5.0, 6.0], [1.0, 2.0])
        assert result == [1.0, 1.0]

    def test_equal_values(self):
        result = apply(screamer.GreaterThan(), [2.0, 2.0], [2.0, 2.0])
        assert result == [0.0, 0.0]

    def test_nan_a(self):
        result = apply(screamer.GreaterThan(), [nan, 1.0], [1.0, 1.0])
        assert math.isnan(result[0])
        assert result[1] == 0.0

    def test_nan_b(self):
        result = apply(screamer.GreaterThan(), [1.0, 2.0], [nan, 1.0])
        assert math.isnan(result[0])
        assert result[1] == 1.0

    def test_scalar(self):
        assert apply_scalar(screamer.GreaterThan(), 3.0, 2.0) == 1.0
        assert apply_scalar(screamer.GreaterThan(), 1.0, 2.0) == 0.0


# ---------------------------------------------------------------------------
# LessThan
# ---------------------------------------------------------------------------

class TestLessThan:
    def test_basic(self):
        result = apply(screamer.LessThan(), [1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        assert result == [1.0, 0.0, 0.0]

    def test_nan_propagates(self):
        result = apply(screamer.LessThan(), [nan, 1.0], [1.0, 1.0])
        assert math.isnan(result[0])
        assert result[1] == 0.0

    def test_scalar(self):
        assert apply_scalar(screamer.LessThan(), 1.0, 2.0) == 1.0
        assert apply_scalar(screamer.LessThan(), 2.0, 2.0) == 0.0


# ---------------------------------------------------------------------------
# GreaterEqual
# ---------------------------------------------------------------------------

class TestGreaterEqual:
    def test_basic(self):
        result = apply(screamer.GreaterEqual(), [1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        assert result == [0.0, 1.0, 1.0]

    def test_nan_propagates(self):
        result = apply(screamer.GreaterEqual(), [nan], [1.0])
        assert math.isnan(result[0])

    def test_scalar(self):
        assert apply_scalar(screamer.GreaterEqual(), 2.0, 2.0) == 1.0
        assert apply_scalar(screamer.GreaterEqual(), 1.0, 2.0) == 0.0


# ---------------------------------------------------------------------------
# LessEqual
# ---------------------------------------------------------------------------

class TestLessEqual:
    def test_basic(self):
        result = apply(screamer.LessEqual(), [1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        assert result == [1.0, 1.0, 0.0]

    def test_nan_propagates(self):
        result = apply(screamer.LessEqual(), [nan], [1.0])
        assert math.isnan(result[0])

    def test_scalar(self):
        assert apply_scalar(screamer.LessEqual(), 2.0, 2.0) == 1.0
        assert apply_scalar(screamer.LessEqual(), 3.0, 2.0) == 0.0


# ---------------------------------------------------------------------------
# Equal
# ---------------------------------------------------------------------------

class TestEqual:
    def test_basic(self):
        result = apply(screamer.Equal(), [1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        assert result == [0.0, 1.0, 0.0]

    def test_nan_propagates(self):
        result = apply(screamer.Equal(), [nan], [nan])
        assert math.isnan(result[0])

    def test_scalar(self):
        assert apply_scalar(screamer.Equal(), 2.0, 2.0) == 1.0
        assert apply_scalar(screamer.Equal(), 1.0, 2.0) == 0.0


# ---------------------------------------------------------------------------
# NotEqual
# ---------------------------------------------------------------------------

class TestNotEqual:
    def test_basic(self):
        result = apply(screamer.NotEqual(), [1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        assert result == [1.0, 0.0, 1.0]

    def test_nan_propagates(self):
        result = apply(screamer.NotEqual(), [nan], [1.0])
        assert math.isnan(result[0])

    def test_scalar(self):
        assert apply_scalar(screamer.NotEqual(), 1.0, 2.0) == 1.0
        assert apply_scalar(screamer.NotEqual(), 2.0, 2.0) == 0.0


# ---------------------------------------------------------------------------
# And
# ---------------------------------------------------------------------------

class TestAnd:
    def test_basic(self):
        # (nonzero, nonzero) -> 1; anything zero -> 0
        result = apply(screamer.And(), [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
        assert result == [1.0, 0.0, 0.0, 0.0]

    def test_nonzero_nonzero(self):
        # Any nonzero value (not just 1.0) counts as true
        result = apply(screamer.And(), [5.0, -3.0], [2.0, 7.0])
        assert result == [1.0, 1.0]

    def test_nan_propagates(self):
        result = apply(screamer.And(), [nan], [1.0])
        assert math.isnan(result[0])

    def test_nan_both(self):
        result = apply(screamer.And(), [nan], [nan])
        assert math.isnan(result[0])

    def test_scalar(self):
        assert apply_scalar(screamer.And(), 1.0, 1.0) == 1.0
        assert apply_scalar(screamer.And(), 1.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Or
# ---------------------------------------------------------------------------

class TestOr:
    def test_basic(self):
        result = apply(screamer.Or(), [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
        assert result == [1.0, 1.0, 1.0, 0.0]

    def test_nonzero_values(self):
        result = apply(screamer.Or(), [0.0, 0.0], [0.0, 5.0])
        assert result == [0.0, 1.0]

    def test_nan_propagates(self):
        result = apply(screamer.Or(), [nan], [0.0])
        assert math.isnan(result[0])

    def test_scalar(self):
        assert apply_scalar(screamer.Or(), 0.0, 1.0) == 1.0
        assert apply_scalar(screamer.Or(), 0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Where  (3 inputs: mask, a, b)
# ---------------------------------------------------------------------------

class TestWhere:
    def test_basic(self):
        result = apply(
            screamer.Where(),
            [1.0, 0.0, 1.0, 0.0],  # mask
            [10.0, 20.0, 30.0, 40.0],  # a (returned when mask nonzero)
            [100.0, 200.0, 300.0, 400.0],  # b (returned when mask zero)
        )
        assert result == [10.0, 200.0, 30.0, 400.0]

    def test_nonzero_mask(self):
        # Any nonzero mask value counts as true
        result = apply(
            screamer.Where(),
            [5.0, -3.0],
            [1.0, 2.0],
            [9.0, 9.0],
        )
        assert result == [1.0, 2.0]

    def test_nan_mask_propagates(self):
        result = apply(
            screamer.Where(),
            [nan],
            [1.0],
            [2.0],
        )
        assert math.isnan(result[0])

    def test_nan_a_passthrough(self):
        # If mask is nonzero and a is NaN, output is NaN (the NaN in a is passed through)
        result = apply(
            screamer.Where(),
            [1.0],
            [nan],
            [5.0],
        )
        assert math.isnan(result[0])

    def test_nan_b_passthrough(self):
        # If mask is zero and b is NaN, output is NaN
        result = apply(
            screamer.Where(),
            [0.0],
            [5.0],
            [nan],
        )
        assert math.isnan(result[0])

    def test_scalar(self):
        assert apply_scalar(screamer.Where(), 1.0, 10.0, 99.0) == 10.0
        assert apply_scalar(screamer.Where(), 0.0, 10.0, 99.0) == 99.0


# ---------------------------------------------------------------------------
# Not  (1 input)
# ---------------------------------------------------------------------------

class TestNot:
    def test_basic(self):
        result = apply(screamer.Not(), [1.0, 0.0, 5.0, -3.0])
        assert result == [0.0, 1.0, 0.0, 0.0]

    def test_nan_propagates(self):
        result = apply(screamer.Not(), [nan, 1.0])
        assert math.isnan(result[0])
        assert result[1] == 0.0

    def test_scalar(self):
        assert apply_scalar(screamer.Not(), 0.0) == 1.0
        assert apply_scalar(screamer.Not(), 1.0) == 0.0
        assert apply_scalar(screamer.Not(), -7.0) == 0.0


# ---------------------------------------------------------------------------
# IsNan  (1 input)
# ---------------------------------------------------------------------------

class TestIsNan:
    def test_basic(self):
        result = apply(screamer.IsNan(), [1.0, nan, 0.0, nan])
        assert result == [0.0, 1.0, 0.0, 1.0]

    def test_nan_returns_one(self):
        # IsNan classifies NaN - it does NOT propagate it
        assert apply_scalar(screamer.IsNan(), nan) == 1.0

    def test_finite_returns_zero(self):
        assert apply_scalar(screamer.IsNan(), 0.0) == 0.0
        assert apply_scalar(screamer.IsNan(), 3.14) == 0.0
        assert apply_scalar(screamer.IsNan(), -1.0) == 0.0

    def test_inf_returns_zero(self):
        # inf is not NaN
        assert apply_scalar(screamer.IsNan(), inf) == 0.0
        assert apply_scalar(screamer.IsNan(), -inf) == 0.0


# ---------------------------------------------------------------------------
# IsFinite  (1 input)
# ---------------------------------------------------------------------------

class TestIsFinite:
    def test_basic(self):
        result = apply(screamer.IsFinite(), [1.0, nan, inf, -inf, 0.0])
        assert result == [1.0, 0.0, 0.0, 0.0, 1.0]

    def test_nan_returns_zero(self):
        # IsFinite classifies NaN - it does NOT propagate it
        assert apply_scalar(screamer.IsFinite(), nan) == 0.0

    def test_inf_returns_zero(self):
        assert apply_scalar(screamer.IsFinite(), inf) == 0.0
        assert apply_scalar(screamer.IsFinite(), -inf) == 0.0

    def test_finite_returns_one(self):
        assert apply_scalar(screamer.IsFinite(), 0.0) == 1.0
        assert apply_scalar(screamer.IsFinite(), 3.14) == 1.0
        assert apply_scalar(screamer.IsFinite(), -1e10) == 1.0
