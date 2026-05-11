"""
Tests for Stoch: 3->2 (high, low, close) -> (%K, %D).

  raw_K[t] = 100 * (close - L_n) / (H_n - L_n)
  %K[t]    = SMA(raw_K, smooth_k)
  %D[t]    = SMA(%K,    d)

smooth_k=1 collapses to the "fast" Stochastic (Lane's original);
smooth_k>=2 gives the "slow" Stochastic that talib.STOCH returns.

Cross-validation against talib.STOCH / talib.STOCHF /
pandas-ta-classic.stoch lives in tests/test_third_party_alignment.py.
This file pins the internal behaviour without requiring those
libraries.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import Stoch


def _ohlc(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    return high, low, close


def ref_stoch(high, low, close, fastk, smooth_k, d):
    """Plain numpy reference. Both outputs are NaN until %D is valid
    (TA-Lib's convention; we follow it)."""
    n = len(close)
    raw_k = np.full(n, np.nan)
    for t in range(fastk - 1, n):
        h = high[t - fastk + 1:t + 1].max()
        l = low[t - fastk + 1:t + 1].min()
        rng = h - l
        raw_k[t] = 0.0 if rng <= 0 else 100.0 * (close[t] - l) / rng

    slow_k = pd.Series(raw_k).rolling(smooth_k).mean().to_numpy()
    slow_d = pd.Series(slow_k).rolling(d).mean().to_numpy()

    # Gate both until the later one (%D) is valid.
    warmup = fastk + smooth_k + d - 3
    slow_k = slow_k.copy()
    slow_k[:warmup] = np.nan
    return slow_k, slow_d


# ---------------------------------------------------------------------------
# Manual reference parity
# ---------------------------------------------------------------------------

class TestManualReference:

    @pytest.mark.parametrize("fastk,smooth_k,d", [
        (14, 3, 3),    # slow stoch, charting default
        (14, 1, 3),    # fast stoch
        (5,  3, 3),    # talib.STOCH default
        (21, 5, 5),    # longer
    ])
    def test_matches_numpy_reference(self, fastk, smooth_k, d):
        high, low, close = _ohlc(150, seed=fastk + smooth_k + d)
        out = Stoch(fastk, smooth_k, d)(high, low, close)
        ref_k, ref_d = ref_stoch(high, low, close, fastk, smooth_k, d)
        np.testing.assert_allclose(out[:, 0], ref_k, equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(out[:, 1], ref_d, equal_nan=True, atol=1e-12)


# ---------------------------------------------------------------------------
# Bounds, edge cases, warmup
# ---------------------------------------------------------------------------

class TestBounds:

    def test_outputs_bounded_zero_to_hundred(self):
        high, low, close = _ohlc(200, seed=1)
        out = Stoch()(high, low, close)
        finite = out[~np.isnan(out)]
        assert np.all(finite >= 0.0 - 1e-12)
        assert np.all(finite <= 100.0 + 1e-12)

    def test_close_at_high_gives_100(self):
        """close == period high => raw_K = 100. SMAs of 100s are still 100."""
        n = 30
        close = np.arange(n, dtype=float)
        high = close.copy()
        low = close - 1.0
        out = Stoch(5, 3, 3)(high, low, close)
        warmup = 5 + 3 + 3 - 3   # 8
        np.testing.assert_allclose(out[warmup:, 0], 100.0, atol=1e-12)
        np.testing.assert_allclose(out[warmup:, 1], 100.0, atol=1e-12)

    def test_close_at_low_gives_zero(self):
        n = 30
        close = -np.arange(n, dtype=float)
        high = close + 1.0
        low = close.copy()
        out = Stoch(5, 3, 3)(high, low, close)
        warmup = 5 + 3 + 3 - 3
        np.testing.assert_allclose(out[warmup:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(out[warmup:, 1], 0.0, atol=1e-12)

    def test_flat_input_returns_zero(self):
        """high == low => range = 0 => raw_K = 0 by convention."""
        x = np.full(30, 5.0)
        out = Stoch(5, 3, 3)(x, x, x)
        warmup = 5 + 3 + 3 - 3
        np.testing.assert_array_equal(out[warmup:, 0], 0.0)
        np.testing.assert_array_equal(out[warmup:, 1], 0.0)


class TestWarmup:

    @pytest.mark.parametrize("fastk,smooth_k,d", [
        (14, 3, 3),
        (5,  1, 3),
        (10, 5, 3),
    ])
    def test_warmup_length(self, fastk, smooth_k, d):
        high, low, close = _ohlc(50, seed=fastk * 100 + smooth_k * 10 + d)
        out = Stoch(fastk, smooth_k, d)(high, low, close)
        warmup = fastk + smooth_k + d - 3
        assert np.all(np.isnan(out[:warmup]))
        assert np.all(np.isfinite(out[warmup:]))

    def test_fast_stoch_when_smooth_k_is_one(self):
        """smooth_k=1 means SMA(1) = identity, so slow_K == raw_K.
        Warmup length matches fastk + d - 2."""
        high, low, close = _ohlc(40, seed=11)
        fastk, d = 14, 3
        out = Stoch(fastk, 1, d)(high, low, close)
        warmup = fastk + 1 + d - 3
        assert np.all(np.isnan(out[:warmup]))
        assert np.all(np.isfinite(out[warmup:]))


# ---------------------------------------------------------------------------
# Dispatcher and parity
# ---------------------------------------------------------------------------

class TestDispatcher:

    def test_scalar_triple_returns_tuple(self):
        out = Stoch(1, 1, 1)(10.0, 5.0, 7.5)
        # window=1: high_n=10, low_n=5, range=5; raw_k = 100*(7.5-5)/5 = 50.
        # smooth_k=1 -> slow_k = 50. d=1 -> slow_d = 50.
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0] == pytest.approx(50.0)
        assert out[1] == pytest.approx(50.0)

    def test_1d_arrays_shape(self):
        high, low, close = _ohlc(50, seed=2)
        out = Stoch()(high, low, close)
        assert out.shape == (50, 2)

    def test_2d_arrays_shape(self):
        rng = np.random.default_rng(3)
        H = 100 + rng.standard_normal((40, 3)).cumsum(axis=0) + 1.0
        L = H - rng.uniform(0.5, 1.5, (40, 3))
        C = L + (H - L) * rng.uniform(0.0, 1.0, (40, 3))
        out = Stoch()(H, L, C)
        assert out.shape == (40, 3, 2)

    def test_2d_per_column_independence(self):
        rng = np.random.default_rng(4)
        H = 100 + rng.standard_normal((50, 3)).cumsum(axis=0) + 1.0
        L = H - rng.uniform(0.5, 1.5, (50, 3))
        C = L + (H - L) * rng.uniform(0.0, 1.0, (50, 3))
        out_2d = Stoch()(H, L, C)
        for k in range(3):
            np.testing.assert_allclose(
                out_2d[:, k, :],
                Stoch()(H[:, k].copy(), L[:, k].copy(), C[:, k].copy()),
                equal_nan=True, atol=1e-12,
            )


class TestParity:

    def test_scalar_loop_matches_array(self):
        high, low, close = _ohlc(50, seed=5)
        obj = Stoch()
        streamed = np.array([obj(h, l, c) for h, l, c in zip(high, low, close)])
        np.testing.assert_allclose(streamed, Stoch()(high, low, close),
                                   equal_nan=True, atol=1e-12)

    def test_reset_clears_history(self):
        high, low, close = _ohlc(40, seed=6)
        obj = Stoch()
        first = obj(high, low, close)
        obj.reset()
        second = obj(high, low, close)
        np.testing.assert_array_equal(first, second)


class TestConstructor:

    def test_default_parameters(self):
        """Defaults are (14, 3, 3) - matches charting and pandas-ta convention."""
        high, low, close = _ohlc(50, seed=7)
        np.testing.assert_array_equal(Stoch()(high, low, close),
                                      Stoch(14, 3, 3)(high, low, close))

    def test_zero_parameters_raise(self):
        with pytest.raises(ValueError):
            Stoch(0, 3, 3)
        with pytest.raises(ValueError):
            Stoch(14, 0, 3)
        with pytest.raises(ValueError):
            Stoch(14, 3, 0)
