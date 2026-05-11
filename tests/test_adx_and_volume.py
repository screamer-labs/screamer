"""
Tests for ADX (last momentum/oscillator) and the volume-aware
indicators (VWAP, OBV, AD, ADOSC, MFI).

Cross-validation against TA-Lib lives in
tests/test_third_party_alignment.py; this file pins behaviour with
plain numpy references.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import (
    ADX, RollingVWAP, OBV, AD, ADOSC, MFI,
    EwMean,
)


def _hlcv(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    volume = 1000 + np.abs(rng.normal(0, 200, n))
    return high, low, close, volume


def _ohlcv(n, seed=0):
    rng = np.random.default_rng(seed)
    open_ = 100 + np.cumsum(rng.normal(0, 1, n))
    close = open_ + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    volume = 1000 + np.abs(rng.normal(0, 200, n))
    return open_, high, low, close, volume


# ---------------------------------------------------------------------------
# ADX -- formula and warmup invariants
# ---------------------------------------------------------------------------

class TestADX:

    def test_warmup_boundaries(self):
        h, l, c, _ = _hlcv(60, seed=0)
        w = 14
        out = ADX(w)(h, l, c)
        # +DI/-DI valid from sample w; ADX valid from sample 2w-1.
        assert np.all(np.isnan(out[:w, 0]))
        assert np.all(np.isfinite(out[w:, 0]))
        assert np.all(np.isnan(out[:w, 1]))
        assert np.all(np.isfinite(out[w:, 1]))
        assert np.all(np.isnan(out[:2 * w - 1, 2]))
        assert np.all(np.isfinite(out[2 * w - 1:, 2]))

    def test_outputs_bounded(self):
        h, l, c, _ = _hlcv(200, seed=1)
        out = ADX(14)(h, l, c)
        for k in range(3):
            finite = out[~np.isnan(out[:, k]), k]
            assert np.all(finite >= 0.0 - 1e-12)
            assert np.all(finite <= 100.0 + 1e-12)

    def test_monotonic_input_full_trend(self):
        """Monotonically rising input -> -DM = 0, DX = 100, so ADX -> 100.
        (+DI < 100 because TR includes the |low - prev_close| component
        which is non-zero on a smooth uptrend.)"""
        h = np.arange(60.0) + 100
        l = h - 0.5
        c = h - 0.1
        out = ADX(10)(h, l, c)
        post = 2 * 10 - 1
        np.testing.assert_allclose(out[post:, 1], 0.0, atol=1e-12)   # -DI = 0
        np.testing.assert_allclose(out[post:, 2], 100.0, atol=1e-12) # ADX = 100
        assert np.all(out[post:, 0] > 80.0)                          # +DI is high

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            ADX(1)


# ---------------------------------------------------------------------------
# OBV -- cumulative signed volume
# ---------------------------------------------------------------------------

class TestOBV:

    def test_matches_manual_reference(self):
        rng = np.random.default_rng(2)
        c = 100 + np.cumsum(rng.normal(0, 1, 50))
        v = 1000 + np.abs(rng.normal(0, 200, 50))
        ours = OBV()(c, v)
        ref = np.empty(50)
        ref[0] = v[0]
        for t in range(1, 50):
            if c[t] > c[t-1]:   ref[t] = ref[t-1] + v[t]
            elif c[t] < c[t-1]: ref[t] = ref[t-1] - v[t]
            else:               ref[t] = ref[t-1]
        np.testing.assert_array_equal(ours, ref)

    def test_no_change_in_close_leaves_obv(self):
        c = np.array([100.0, 100.0, 100.0])
        v = np.array([1000.0, 500.0, 200.0])
        ours = OBV()(c, v)
        # OBV[0] = v[0] = 1000; then close doesn't change -> OBV stays at 1000.
        np.testing.assert_array_equal(ours, [1000.0, 1000.0, 1000.0])


# ---------------------------------------------------------------------------
# AD -- cumulative weighted volume
# ---------------------------------------------------------------------------

class TestAD:

    def test_matches_manual_reference(self):
        h, l, c, v = _hlcv(50, seed=3)
        ours = AD()(h, l, c, v)
        ref = np.empty(50)
        ad = 0.0
        for t in range(50):
            rng = h[t] - l[t]
            if rng > 0:
                clv = ((c[t] - l[t]) - (h[t] - c[t])) / rng
                ad += clv * v[t]
            ref[t] = ad
        # Tiny float-order differences between C++ and Python AD
        # accumulations (ULP-level).
        np.testing.assert_allclose(ours, ref, atol=1e-9)

    def test_flat_bar_no_change(self):
        """high == low => CLV undefined => AD unchanged. We use 0 (matches TA-Lib)."""
        out = AD()(np.array([5.0, 5.0]), np.array([5.0, 5.0]),
                   np.array([5.0, 5.0]), np.array([1000.0, 1000.0]))
        np.testing.assert_array_equal(out, [0.0, 0.0])


# ---------------------------------------------------------------------------
# ADOSC -- EMA(AD, fast) - EMA(AD, slow)
# ---------------------------------------------------------------------------

class TestADOSC:

    def test_matches_manual_pandas_composition(self):
        h, l, c, v = _hlcv(80, seed=4)
        ours = ADOSC(3, 10)(h, l, c, v)
        ad = AD()(h, l, c, v)
        ema_fast = pd.Series(ad).ewm(span=3, adjust=True).mean().to_numpy()
        ema_slow = pd.Series(ad).ewm(span=10, adjust=True).mean().to_numpy()
        ref = ema_fast - ema_slow
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_first_sample_is_zero(self):
        """At t=0, AD=0 and both EMAs = 0, so ADOSC = 0."""
        out = ADOSC()(np.array([10.0]), np.array([5.0]),
                       np.array([7.5]), np.array([1000.0]))
        assert out == 0.0

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            ADOSC(10, 5)  # fast >= slow


# ---------------------------------------------------------------------------
# MFI -- volume-weighted RSI on typical price
# ---------------------------------------------------------------------------

def ref_mfi(h, l, c, v, n):
    tp = (h + l + c) / 3.0
    mf = tp * v
    pos = np.zeros(len(c))
    neg = np.zeros(len(c))
    for t in range(1, len(c)):
        if tp[t] > tp[t-1]: pos[t] = mf[t]
        elif tp[t] < tp[t-1]: neg[t] = mf[t]
    out = np.full(len(c), np.nan)
    for t in range(n, len(c)):
        pos_sum = pos[t-n+1:t+1].sum()
        neg_sum = neg[t-n+1:t+1].sum()
        total = pos_sum + neg_sum
        out[t] = 100.0 * pos_sum / total if total > 0 else 100.0
    return out


class TestMFI:

    @pytest.mark.parametrize("n", [5, 10, 14, 20])
    def test_matches_manual_reference(self, n):
        h, l, c, v = _hlcv(100, seed=n)
        ours = MFI(n)(h, l, c, v)
        ref = ref_mfi(h, l, c, v, n)
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_outputs_in_zero_to_hundred(self):
        h, l, c, v = _hlcv(100, seed=5)
        out = MFI(14)(h, l, c, v)
        finite = out[~np.isnan(out)]
        assert np.all(finite >= 0.0 - 1e-12)
        assert np.all(finite <= 100.0 + 1e-12)


# ---------------------------------------------------------------------------
# RollingVWAP -- rolling typical-price-weighted average
# ---------------------------------------------------------------------------

class TestRollingVWAP:

    @pytest.mark.parametrize("w", [5, 10, 20])
    def test_matches_pandas_composition(self, w):
        h, l, c, v = _hlcv(100, seed=w)
        ours = RollingVWAP(w)(h, l, c, v)
        tp = (h + l + c) / 3.0
        ref_pv = pd.Series(tp * v).rolling(w).sum().to_numpy()
        ref_v = pd.Series(v).rolling(w).sum().to_numpy()
        ref = ref_pv / ref_v
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_warmup_is_window_minus_one(self):
        h, l, c, v = _hlcv(30, seed=6)
        w = 10
        out = RollingVWAP(w)(h, l, c, v)
        assert np.all(np.isnan(out[:w - 1]))
        assert np.all(np.isfinite(out[w - 1:]))


# ---------------------------------------------------------------------------
# Reset / scalar-loop parity
# ---------------------------------------------------------------------------

class TestParity:

    @pytest.mark.parametrize("ctor,n_inputs", [
        (lambda: ADX(10),         3),
        (lambda: RollingVWAP(10), 4),
        (lambda: OBV(),           2),
        (lambda: AD(),            4),
        (lambda: ADOSC(),         4),
        (lambda: MFI(10),         4),
    ])
    def test_reset_clears_history(self, ctor, n_inputs):
        h, l, c, v = _hlcv(40, seed=20)
        obj = ctor()
        if n_inputs == 2:
            args = (c, v)
        elif n_inputs == 3:
            args = (h, l, c)
        else:
            args = (h, l, c, v)
        first = obj(*args)
        obj.reset()
        second = obj(*args)
        np.testing.assert_array_equal(first, second)
