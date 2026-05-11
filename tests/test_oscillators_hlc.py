"""
Tests for BOP, CCI, UltimateOscillator, StochRSI.

The first three take HLC inputs (BOP also needs Open); StochRSI takes
a single price stream. Cross-validation against TA-Lib lives in
tests/test_third_party_alignment.py; this file pins behaviour with
plain numpy references.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import BOP, CCI, UltimateOscillator, StochRSI, RollingRSI


def _ohlc(n, seed=0):
    rng = np.random.default_rng(seed)
    open_ = 100 + np.cumsum(rng.normal(0, 1, n))
    close = open_ + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    return open_, high, low, close


# ---------------------------------------------------------------------------
# BOP -- (close - open) / (high - low)
# ---------------------------------------------------------------------------

class TestBOP:

    def test_matches_formula(self):
        open_, high, low, close = _ohlc(100, seed=0)
        ours = BOP()(open_, high, low, close)
        ref = (close - open_) / (high - low)
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_bounded_in_minus1_to_1(self):
        """When close is between low and high (always true for a sane bar),
        BOP is bounded by [-1, 1]."""
        open_, high, low, close = _ohlc(200, seed=1)
        # Ensure open is also within [low, high] for the bound to be tight.
        open_ = np.clip(open_, low, high)
        out = BOP()(open_, high, low, close)
        assert np.all(out >= -1.0 - 1e-12)
        assert np.all(out <= 1.0 + 1e-12)

    def test_flat_bar_returns_zero(self):
        x = np.full(10, 5.0)
        out = BOP()(x, x, x, x)
        np.testing.assert_array_equal(out, 0.0)

    def test_close_at_high_with_open_at_low(self):
        """(close - open) / (high - low) = (h - l) / (h - l) = 1."""
        out = BOP()(np.array([1.0]), np.array([3.0]),
                    np.array([1.0]), np.array([3.0]))
        assert out[0] == pytest.approx(1.0)

    def test_no_warmup(self):
        out = BOP()(np.array([1.0]), np.array([2.0]),
                    np.array([0.5]), np.array([1.5]))
        assert np.isfinite(out[0])


# ---------------------------------------------------------------------------
# CCI -- (TP - SMA(TP)) / (0.015 * MAD(TP))
# ---------------------------------------------------------------------------

def ref_cci(high, low, close, n):
    tp = (high + low + close) / 3.0
    out = np.full(len(close), np.nan)
    for t in range(n - 1, len(close)):
        window = tp[t - n + 1:t + 1]
        mean = window.mean()
        mad = np.mean(np.abs(window - mean))
        out[t] = 0.0 if mad == 0 else (tp[t] - mean) / (0.015 * mad)
    return out


class TestCCI:

    @pytest.mark.parametrize("n", [5, 10, 14, 20])
    def test_matches_numpy_reference(self, n):
        _, high, low, close = _ohlc(120, seed=n)
        np.testing.assert_allclose(
            CCI(n)(high, low, close),
            ref_cci(high, low, close, n),
            equal_nan=True, atol=1e-12,
        )

    def test_strict_warmup_is_nan(self):
        _, high, low, close = _ohlc(40, seed=2)
        out = CCI(14)(high, low, close)
        assert np.all(np.isnan(out[:13]))
        assert np.all(np.isfinite(out[13:]))

    def test_constant_input_returns_zero(self):
        """TP constant => MAD = 0 => return 0 by convention."""
        x = np.full(30, 5.0)
        out = CCI(14)(x, x, x)
        np.testing.assert_array_equal(out[13:], 0.0)


# ---------------------------------------------------------------------------
# UltimateOscillator -- weighted-average BP/TR over three periods
# ---------------------------------------------------------------------------

class TestUltimateOscillator:

    def test_bounded_zero_to_hundred(self):
        _, high, low, close = _ohlc(200, seed=3)
        out = UltimateOscillator()(high, low, close)
        finite = out[~np.isnan(out)]
        assert np.all(finite >= 0.0 - 1e-9)
        assert np.all(finite <= 100.0 + 1e-9)

    def test_first_valid_at_max_period(self):
        _, high, low, close = _ohlc(50, seed=4)
        out = UltimateOscillator(period1=7, period2=14, period3=28)(high, low, close)
        assert np.all(np.isnan(out[:28]))
        assert np.all(np.isfinite(out[28:]))

    def test_custom_periods(self):
        """With shorter periods, warmup is correspondingly shorter."""
        _, high, low, close = _ohlc(50, seed=5)
        out = UltimateOscillator(2, 4, 8)(high, low, close)
        assert np.all(np.isnan(out[:8]))
        assert np.all(np.isfinite(out[8:]))


# ---------------------------------------------------------------------------
# StochRSI -- Stoch applied to a Wilder RSI
# ---------------------------------------------------------------------------

def ref_stoch_rsi(close, rsi_period, stoch_period, smooth_k, d):
    """Manual reference: use our RollingRSI to build the RSI series,
    then apply rolling min/max + two SMAs."""
    from screamer import RollingRSI
    rsi = RollingRSI(rsi_period)(close)
    raw_k = np.full(len(close), np.nan)
    rsi_min = pd.Series(rsi).rolling(stoch_period).min().to_numpy()
    rsi_max = pd.Series(rsi).rolling(stoch_period).max().to_numpy()
    for t in range(len(close)):
        if np.isnan(rsi_max[t]) or rsi_max[t] - rsi_min[t] <= 0:
            continue
        raw_k[t] = 100.0 * (rsi[t] - rsi_min[t]) / (rsi_max[t] - rsi_min[t])
    slow_k = pd.Series(raw_k).rolling(smooth_k).mean().to_numpy()
    slow_d = pd.Series(slow_k).rolling(d).mean().to_numpy()
    # Gate both until %D is valid -- our class does this.
    warmup_after_rsi = stoch_period + smooth_k + d - 2
    rsi_first_valid = rsi_period
    # First fully-valid sample
    final_warmup = rsi_first_valid + warmup_after_rsi - 1
    slow_k = slow_k.copy()
    slow_k[:final_warmup] = np.nan
    slow_d[:final_warmup] = np.nan
    return slow_k, slow_d


class TestStochRSI:

    def test_matches_reference(self):
        rng = np.random.default_rng(7)
        close = 100 + np.cumsum(rng.normal(0, 1, 200))
        out = StochRSI(14, 14, 1, 3)(close)
        ref_k, ref_d = ref_stoch_rsi(close, 14, 14, 1, 3)
        np.testing.assert_allclose(out[:, 0], ref_k, equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(out[:, 1], ref_d, equal_nan=True, atol=1e-12)

    def test_bounded_zero_to_hundred(self):
        rng = np.random.default_rng(8)
        close = 100 + np.cumsum(rng.normal(0, 1, 200))
        out = StochRSI(14, 14, 1, 3)(close)
        finite = out[~np.isnan(out)]
        assert np.all(finite >= -1e-9)
        assert np.all(finite <= 100.0 + 1e-9)

    def test_smooth_k_3_slow_form(self):
        """smooth_k=3 gives the slow StochRSI; longer warmup, smoother output."""
        rng = np.random.default_rng(9)
        close = 100 + np.cumsum(rng.normal(0, 1, 200))
        fast = StochRSI(14, 14, 1, 3)(close)
        slow = StochRSI(14, 14, 3, 3)(close)
        # Slow has more NaN warmup than fast.
        assert np.sum(np.isnan(slow[:, 0])) > np.sum(np.isnan(fast[:, 0]))


# ---------------------------------------------------------------------------
# Dispatcher parity (all four)
# ---------------------------------------------------------------------------

class TestDispatcher:

    def test_bop_scalar_quadruple(self):
        out = BOP()(1.0, 3.0, 1.0, 3.0)
        assert out == pytest.approx(1.0)

    def test_cci_scalar_triple(self):
        # Single sample -- below warmup, returns NaN.
        out = CCI(5)(10.0, 5.0, 7.5)
        assert np.isnan(out)

    def test_ultosc_scalar_triple(self):
        out = UltimateOscillator(2, 4, 8)(10.0, 5.0, 7.5)
        assert np.isnan(out)

    def test_stochrsi_scalar(self):
        out = StochRSI()(50.0)
        assert isinstance(out, tuple)
        assert all(np.isnan(v) for v in out)


# ---------------------------------------------------------------------------
# Reset / scalar-loop parity
# ---------------------------------------------------------------------------

class TestParity:

    def test_bop_scalar_loop(self):
        open_, high, low, close = _ohlc(40, seed=10)
        obj = BOP()
        streamed = np.array([obj(o, h, l, c)
                             for o, h, l, c in zip(open_, high, low, close)])
        np.testing.assert_allclose(streamed, BOP()(open_, high, low, close), atol=1e-12)

    def test_cci_reset(self):
        _, h, l, c = _ohlc(40, seed=11)
        obj = CCI(10)
        first = obj(h, l, c)
        obj.reset()
        second = obj(h, l, c)
        np.testing.assert_array_equal(first, second)

    def test_ultosc_reset(self):
        _, h, l, c = _ohlc(60, seed=12)
        obj = UltimateOscillator()
        first = obj(h, l, c)
        obj.reset()
        second = obj(h, l, c)
        np.testing.assert_array_equal(first, second)

    def test_stochrsi_reset(self):
        rng = np.random.default_rng(13)
        close = 100 + np.cumsum(rng.normal(0, 1, 80))
        obj = StochRSI()
        first = obj(close)
        obj.reset()
        second = obj(close)
        np.testing.assert_array_equal(first, second)
