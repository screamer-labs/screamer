"""
Tests for RollingRange = max - min.

The class deliberately *composes* the same algorithm RollingMinMax
uses (two monotonic deques). The composition reference is exactly
that subtraction; we validate equality to floating-point precision.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import RollingRange, RollingMinMax, RollingMin, RollingMax


@pytest.mark.parametrize("w", [3, 5, 10, 20])
def test_matches_rolling_min_max_composition(w):
    """The dedicated RollingRange must equal RollingMinMax's max - min."""
    rng = np.random.default_rng(w)
    x = rng.standard_normal(80)
    ours = RollingRange(w)(x)
    mm = RollingMinMax(w)(x)
    np.testing.assert_array_equal(ours, mm[:, 1] - mm[:, 0])


@pytest.mark.parametrize("w", [3, 5, 10, 20])
def test_matches_rolling_max_minus_rolling_min(w):
    """Same algorithm, two parallel classes -- result must agree."""
    rng = np.random.default_rng(w + 50)
    x = rng.standard_normal(80)
    np.testing.assert_array_equal(RollingRange(w)(x),
                                  RollingMax(w)(x) - RollingMin(w)(x))


def test_matches_pandas():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    w = 7
    ours = RollingRange(w)(x)
    s = pd.Series(x)
    ref = s.rolling(w).max().to_numpy() - s.rolling(w).min().to_numpy()
    # RollingRange returns range from the start (no NaN warmup); pandas
    # has NaN for the first w-1. Compare only after warmup.
    np.testing.assert_allclose(ours[w - 1:], ref[w - 1:], atol=1e-12)


def test_constant_input_is_zero():
    out = RollingRange(5)(np.full(20, 3.7))
    np.testing.assert_array_equal(out, 0.0)


def test_monotonic_input():
    """Monotonically increasing arange: range over a window of w is w-1."""
    x = np.arange(20, dtype=float)
    w = 6
    out = RollingRange(w)(x)
    np.testing.assert_array_equal(out[w - 1:], w - 1)


def test_range_non_negative():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(100)
    out = RollingRange(7)(x)
    assert np.all(out >= 0)


def test_scalar_loop_matches_array():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(50)
    w = 7
    obj = RollingRange(w)
    streamed = np.array([obj(v) for v in x])
    np.testing.assert_array_equal(streamed, RollingRange(w)(x))
