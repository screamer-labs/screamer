"""
Tests for WilliamsR: 3->1 (high, low, close) -> %R in [-100, 0].

  %R[t] = -100 * (high_n - close) / (high_n - low_n)

where high_n / low_n are the rolling max / min of high / low over a
window of size n.

Cross-validation against TA-Lib's WILLR lives in
tests/test_third_party_alignment.py. This file pins the internal
behaviour without requiring TA-Lib to be installed.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import WilliamsR


def _ohlc(n, seed=0):
    """Make a self-consistent (high, low, close) sample."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    return high, low, close


def ref_williams_r(high, low, close, n):
    """Plain numpy reference."""
    out = np.full(len(close), np.nan)
    for t in range(n - 1, len(close)):
        h = high[t - n + 1:t + 1].max()
        l = low[t - n + 1:t + 1].min()
        rng = h - l
        out[t] = 0.0 if rng <= 0 else -100.0 * (h - close[t]) / rng
    return out


# ---------------------------------------------------------------------------
# Manual reference parity
# ---------------------------------------------------------------------------

class TestManualReference:

    @pytest.mark.parametrize("n", [5, 10, 14, 20])
    def test_matches_numpy_reference(self, n):
        high, low, close = _ohlc(150, seed=n)
        ours = WilliamsR(n)(high, low, close)
        ref = ref_williams_r(high, low, close, n)
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)


# ---------------------------------------------------------------------------
# Bounds and edge cases
# ---------------------------------------------------------------------------

class TestBounds:

    def test_bounded_minus100_to_zero(self):
        high, low, close = _ohlc(200, seed=1)
        out = WilliamsR(14)(high, low, close)
        finite = out[~np.isnan(out)]
        assert np.all(finite <= 0.0 + 1e-12)
        assert np.all(finite >= -100.0 - 1e-12)

    def test_close_at_high_gives_zero(self):
        """When close == period high, %R = 0."""
        n = 5
        # high == close == ascending, low always slightly below.
        close = np.arange(20, dtype=float)
        high = close.copy()
        low = close - 1.0
        out = WilliamsR(n)(high, low, close)
        np.testing.assert_allclose(out[n - 1:], 0.0, atol=1e-12)

    def test_close_at_low_gives_minus100(self):
        """When close == period low, %R = -100."""
        n = 5
        close = -np.arange(20, dtype=float)   # decreasing
        high = close + 1.0
        low = close.copy()
        out = WilliamsR(n)(high, low, close)
        np.testing.assert_allclose(out[n - 1:], -100.0, atol=1e-12)

    def test_flat_input_returns_zero(self):
        """high == low everywhere -> range = 0 -> %R = 0 by convention."""
        x = np.full(20, 5.0)
        out = WilliamsR(7)(x, x, x)
        np.testing.assert_array_equal(out[6:], 0.0)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

class TestWarmup:

    def test_strict_warmup(self):
        high, low, close = _ohlc(30, seed=2)
        n = 10
        out = WilliamsR(n)(high, low, close)
        assert np.all(np.isnan(out[:n - 1]))
        assert np.all(np.isfinite(out[n - 1:]))


# ---------------------------------------------------------------------------
# Dispatcher / parity (3-input HLC follows N->1 dispatch)
# ---------------------------------------------------------------------------

class TestDispatcher:

    def test_scalar_triple(self):
        out = WilliamsR(1)(10.0, 5.0, 7.5)
        # window=1: single sample, high == low (= 10 and 5? no, single sample
        # means high_n=10, low_n=5, range=5, %R = -100*(10-7.5)/5 = -50.
        assert out == pytest.approx(-50.0)

    def test_three_parallel_1d_arrays(self):
        high, low, close = _ohlc(50, seed=3)
        n = 10
        out = WilliamsR(n)(high, low, close)
        assert out.shape == (50,)

    def test_three_parallel_2d_arrays(self):
        rng = np.random.default_rng(4)
        H = 100 + rng.standard_normal((40, 3)).cumsum(axis=0) + 1.0
        L = H - rng.uniform(0.5, 1.5, (40, 3))
        C = L + (H - L) * rng.uniform(0.0, 1.0, (40, 3))
        n = 10
        out_2d = WilliamsR(n)(H, L, C)
        for k in range(3):
            np.testing.assert_array_equal(
                out_2d[:, k],
                WilliamsR(n)(H[:, k].copy(), L[:, k].copy(), C[:, k].copy()),
            )

    def test_iterables(self):
        h = [10.0, 11.0, 12.0]
        l = [5.0, 6.0, 7.0]
        c = [7.5, 8.5, 11.0]
        out = WilliamsR(1)(iter(h), iter(l), iter(c))
        assert hasattr(out, "__next__") and not isinstance(out, list)
        vals = list(out)
        # window=1: at each step, high_n=h[i], low_n=l[i], close=c[i].
        # %R[2] = -100 * (12 - 11) / (12 - 5 in window=1 = 12-7) ... wait window=1,
        # so high_n = h[2] = 12, low_n = l[2] = 7. %R = -100 * (12-11)/(12-7) = -20.
        assert vals[2] == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# Reset / scalar-loop parity
# ---------------------------------------------------------------------------

class TestParity:

    def test_scalar_loop_matches_array(self):
        high, low, close = _ohlc(50, seed=5)
        n = 10
        obj = WilliamsR(n)
        streamed = np.array([obj(h, l, c) for h, l, c in zip(high, low, close)])
        np.testing.assert_allclose(streamed, WilliamsR(n)(high, low, close),
                                   equal_nan=True, atol=1e-12)

    def test_reset_clears_history(self):
        high, low, close = _ohlc(40, seed=6)
        obj = WilliamsR(10)
        first = obj(high, low, close)
        obj.reset()
        second = obj(high, low, close)
        np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_default_window_is_14(self):
        high, low, close = _ohlc(30, seed=7)
        np.testing.assert_array_equal(WilliamsR()(high, low, close),
                                      WilliamsR(14)(high, low, close))

    def test_zero_window_raises(self):
        with pytest.raises(ValueError):
            WilliamsR(0)
