"""
Tests for the cumulative-from-t=0 family and Detrend.

  CumSum, CumProd, CumMax, CumMin: O(1)-memory running reductions over
  all history. Match numpy.cumsum / cumprod / maximum.accumulate /
  minimum.accumulate exactly. NaN propagates (numpy semantics, not
  pandas skipna).

  Detrend(window): x[t] - RollingMean(window)(x)[t].
"""
import numpy as np
import pandas as pd
import pytest

from screamer import (
    CumSum, CumProd, CumMax, CumMin, Detrend,
    RollingMean,
)


# ---------------------------------------------------------------------------
# Cumulative reductions
# ---------------------------------------------------------------------------

CUM_PAIRS = [
    (CumSum,  np.cumsum),
    (CumProd, np.cumprod),
    (CumMax,  np.maximum.accumulate),
    (CumMin,  np.minimum.accumulate),
]


@pytest.mark.parametrize("cls,reference", CUM_PAIRS, ids=[c.__name__ for c, _ in CUM_PAIRS])
def test_array_matches_numpy(cls, reference):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    np.testing.assert_array_equal(cls()(x), reference(x))


@pytest.mark.parametrize("cls,reference", CUM_PAIRS, ids=[c.__name__ for c, _ in CUM_PAIRS])
def test_scalar_loop_matches_array(cls, reference):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(50)

    obj = cls()
    streamed = np.array([obj(v) for v in x])
    np.testing.assert_array_equal(streamed, reference(x))


@pytest.mark.parametrize("cls,reference", CUM_PAIRS, ids=[c.__name__ for c, _ in CUM_PAIRS])
def test_nan_propagates(cls, reference):
    """Once an input is NaN, all subsequent outputs must be NaN. Numpy
    behaviour, not pandas skipna behaviour."""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    out = cls()(x)
    expected = reference(x)
    # NaN positions must match.
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))


def test_cum_max_monotonic():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(100)
    out = CumMax()(x)
    # Each value is >= the previous.
    assert np.all(np.diff(out) >= 0)


def test_cum_min_monotonic():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(100)
    out = CumMin()(x)
    assert np.all(np.diff(out) <= 0)


def test_cum_max_drawdown_use_case():
    """The canonical use case: drawdown = current / peak - 1."""
    prices = np.array([100.0, 105.0, 102.0, 110.0, 95.0, 99.0, 115.0])
    peak = CumMax()(prices)
    drawdown = prices / peak - 1.0
    expected = np.array([0.0, 0.0, -3 / 105, 0.0, -15 / 110, -11 / 110, 0.0])
    np.testing.assert_allclose(drawdown, expected, atol=1e-12)


def test_cum_prod_with_zero():
    """Multiplication by zero pins the product to zero forever."""
    out = CumProd()(np.array([2.0, 3.0, 0.0, 5.0, 7.0]))
    np.testing.assert_array_equal(out, [2.0, 6.0, 0.0, 0.0, 0.0])


def test_reset_returns_to_initial():
    cm = CumMax()
    cm(5.0)
    cm(10.0)
    assert cm(7.0) == 10.0
    cm.reset()
    assert cm(3.0) == 3.0      # back to initial state


@pytest.mark.parametrize("cls", [CumSum, CumProd, CumMax, CumMin])
def test_2d_per_column_independence(cls):
    """Cumulative reductions: each column independently."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((30, 4))
    out_2d = cls()(X)
    for k in range(X.shape[1]):
        np.testing.assert_array_equal(out_2d[:, k], cls()(X[:, k].copy()))


# ---------------------------------------------------------------------------
# Detrend
# ---------------------------------------------------------------------------

class TestDetrend:

    @pytest.mark.parametrize("window", [3, 5, 10, 20])
    def test_matches_x_minus_rolling_mean(self, window):
        rng = np.random.default_rng(window)
        x = rng.standard_normal(80)
        out = Detrend(window)(x)
        ref = x - RollingMean(window)(x)
        # Two different evaluation orders can drift by an ULP or two.
        np.testing.assert_allclose(out, ref, equal_nan=True, atol=1e-12)

    def test_matches_pandas_subtraction(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(60)
        ref = x - pd.Series(x).rolling(10).mean().to_numpy()
        np.testing.assert_allclose(Detrend(10)(x), ref, equal_nan=True, atol=1e-10)

    def test_constant_input_yields_zero(self):
        """When x is constant the rolling mean equals x, so detrended is 0."""
        x = np.full(30, 7.5)
        out = Detrend(5)(x)
        # Warmup is NaN under strict policy; after that the result is 0.
        np.testing.assert_array_equal(np.isnan(out[:4]), [True] * 4)
        np.testing.assert_allclose(out[4:], 0.0, atol=1e-12)

    def test_window_size_argument(self):
        # Trivial sanity that argument flows through.
        d_small = Detrend(3)
        d_big = Detrend(30)
        x = np.random.default_rng(6).standard_normal(50)
        # Different windows must produce different results.
        assert not np.array_equal(d_small(x), d_big(x))

    def test_2d_per_column_independence(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((40, 3))
        out_2d = Detrend(5)(X)
        for k in range(X.shape[1]):
            np.testing.assert_array_equal(out_2d[:, k], Detrend(5)(X[:, k].copy()))
