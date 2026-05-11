"""
Tests for DonchianChannels, KeltnerChannels, RollingYangZhangVar/Vol.

Channels are envelope-style indicators. Yang-Zhang completes the
range-based volatility quartet (Parkinson / Garman-Klass /
Rogers-Satchell / Yang-Zhang).
"""
import numpy as np
import pandas as pd
import pytest

from screamer import (
    DonchianChannels, KeltnerChannels,
    RollingYangZhangVar, RollingYangZhangVol,
    EwMean, ATR,
)


def _ohlc(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, n))
    return open_, high, low, close


# ---------------------------------------------------------------------------
# DonchianChannels
# ---------------------------------------------------------------------------

class TestDonchianChannels:

    @pytest.mark.parametrize("w", [5, 10, 20, 50])
    def test_matches_rolling_max_min_composition(self, w):
        _, h, l, _ = _ohlc(150, seed=w)
        dc = DonchianChannels(w)(h, l)
        upper = pd.Series(h).rolling(w).max().to_numpy()
        lower = pd.Series(l).rolling(w).min().to_numpy()
        mid = (upper + lower) / 2
        np.testing.assert_allclose(dc[:, 0], lower, equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(dc[:, 1], mid,   equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(dc[:, 2], upper, equal_nan=True, atol=1e-12)

    def test_warmup_is_window_minus_one(self):
        _, h, l, _ = _ohlc(30, seed=1)
        dc = DonchianChannels(10)(h, l)
        for k in range(3):
            assert np.all(np.isnan(dc[:9, k]))
            assert np.all(np.isfinite(dc[9:, k]))

    def test_invariants(self):
        """Lower <= mid <= upper everywhere; mid = (lower + upper) / 2."""
        _, h, l, _ = _ohlc(100, seed=2)
        dc = DonchianChannels(14)(h, l)
        finite = ~np.isnan(dc[:, 0])
        assert np.all(dc[finite, 0] <= dc[finite, 1])
        assert np.all(dc[finite, 1] <= dc[finite, 2])
        np.testing.assert_allclose(
            dc[finite, 1], 0.5 * (dc[finite, 0] + dc[finite, 2]), atol=1e-12,
        )

    def test_constant_input_collapses(self):
        x = np.full(20, 5.0)
        dc = DonchianChannels(5)(x, x)
        np.testing.assert_array_equal(dc[4:, 0], 5.0)
        np.testing.assert_array_equal(dc[4:, 1], 5.0)
        np.testing.assert_array_equal(dc[4:, 2], 5.0)


# ---------------------------------------------------------------------------
# KeltnerChannels
# ---------------------------------------------------------------------------

class TestKeltnerChannels:

    @pytest.mark.parametrize("w", [10, 14, 20])
    def test_matches_ema_plus_atr_composition(self, w):
        _, h, l, c = _ohlc(150, seed=w)
        kc = KeltnerChannels(w, num_atr=2.0)(h, l, c)
        mid_ref = EwMean(span=w)(c)
        atr_ref = ATR(w)(h, l, c)
        upper_ref = mid_ref + 2.0 * atr_ref
        lower_ref = mid_ref - 2.0 * atr_ref
        # Keltner gates ALL three lines together until ATR is valid
        # (sample index w), so the mid is NaN during that window even
        # though the raw EMA is defined from t=0.
        post = slice(w, None)
        np.testing.assert_allclose(kc[post, 0], lower_ref[post], atol=1e-12)
        np.testing.assert_allclose(kc[post, 1], mid_ref[post],   atol=1e-12)
        np.testing.assert_allclose(kc[post, 2], upper_ref[post], atol=1e-12)
        assert np.all(np.isnan(kc[:w, 1]))

    def test_warmup_matches_atr_warmup(self):
        _, h, l, c = _ohlc(50, seed=3)
        kc = KeltnerChannels(14, 2.0)(h, l, c)
        for k in range(3):
            assert np.all(np.isnan(kc[:14, k]))
            assert np.all(np.isfinite(kc[14:, k]))

    def test_num_atr_scales_width(self):
        _, h, l, c = _ohlc(100, seed=4)
        kc1 = KeltnerChannels(14, 1.0)(h, l, c)
        kc2 = KeltnerChannels(14, 2.0)(h, l, c)
        # Width at num_atr=2 is exactly 2x width at num_atr=1.
        w1 = kc1[:, 2] - kc1[:, 0]
        w2 = kc2[:, 2] - kc2[:, 0]
        np.testing.assert_allclose(w2, 2.0 * w1, equal_nan=True, atol=1e-12)

    def test_invariants(self):
        _, h, l, c = _ohlc(100, seed=5)
        kc = KeltnerChannels(14)(h, l, c)
        finite = ~np.isnan(kc[:, 0])
        assert np.all(kc[finite, 0] <= kc[finite, 1])
        assert np.all(kc[finite, 1] <= kc[finite, 2])


# ---------------------------------------------------------------------------
# Yang-Zhang
# ---------------------------------------------------------------------------

def ref_yz(o, h, l, c, n):
    """Plain numpy reference."""
    nbar = len(c)
    out = np.full(nbar, np.nan)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    overnight = np.log(o[1:] / c[:-1])
    oc = np.log(c / o)
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    for t in range(n, nbar):
        on_win = overnight[t - n:t]
        oc_win = oc[t - n + 1:t + 1]
        rs_win = rs[t - n + 1:t + 1]
        var_on = on_win.var(ddof=1)
        var_oc = oc_win.var(ddof=1)
        rs_mean = rs_win.mean()
        out[t] = var_on + k * var_oc + (1 - k) * rs_mean
    return out


class TestYangZhang:

    @pytest.mark.parametrize("n", [5, 10, 14, 20])
    def test_matches_manual_reference(self, n):
        o, h, l, c = _ohlc(150, seed=n)
        ours = RollingYangZhangVar(n)(o, h, l, c)
        ref = ref_yz(o, h, l, c, n)
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_vol_equals_sqrt_var(self):
        o, h, l, c = _ohlc(100, seed=11)
        var = RollingYangZhangVar(10)(o, h, l, c)
        vol = RollingYangZhangVol(10)(o, h, l, c)
        np.testing.assert_allclose(vol, np.sqrt(var), equal_nan=True, atol=1e-15)

    def test_warmup_first_valid_at_window_size(self):
        o, h, l, c = _ohlc(30, seed=12)
        out = RollingYangZhangVar(10)(o, h, l, c)
        assert np.all(np.isnan(out[:10]))
        assert np.all(np.isfinite(out[10:]))

    def test_yz_is_non_negative(self):
        o, h, l, c = _ohlc(200, seed=13)
        out = RollingYangZhangVar(14)(o, h, l, c)
        finite = out[~np.isnan(out)]
        # YZ is a weighted sum of variances + a mean of non-negative
        # RS values, so it must be >= 0 modulo rounding.
        assert np.all(finite >= -1e-15)

    def test_k_factor_correctness(self):
        """Verify our k = 0.34 / (1.34 + (n+1)/(n-1)) by checking against
        the manual reference at one specific window."""
        n = 20
        expected_k = 0.34 / (1.34 + (n + 1) / (n - 1))
        # The class uses k internally; we can verify indirectly by
        # confirming output matches the manual ref that uses the same k.
        o, h, l, c = _ohlc(100, seed=14)
        ours = RollingYangZhangVar(n)(o, h, l, c)
        ref = ref_yz(o, h, l, c, n)
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)
        # Spot-check the formula component at one position (post-warmup).
        t = 50
        overnight = np.log(o[1:] / c[:-1])
        oc = np.log(c / o)
        rs = np.log(h/c)*np.log(h/o) + np.log(l/c)*np.log(l/o)
        var_on = overnight[t-n:t].var(ddof=1)
        var_oc = oc[t-n+1:t+1].var(ddof=1)
        rs_mean = rs[t-n+1:t+1].mean()
        expected = var_on + expected_k * var_oc + (1 - expected_k) * rs_mean
        assert ours[t] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Parity (reset / scalar loop / 2D)
# ---------------------------------------------------------------------------

class TestParity:

    @pytest.mark.parametrize("ctor,n_inputs", [
        (lambda: DonchianChannels(10), 2),
        (lambda: KeltnerChannels(10), 3),
        (lambda: RollingYangZhangVar(10), 4),
        (lambda: RollingYangZhangVol(10), 4),
    ])
    def test_reset_clears_history(self, ctor, n_inputs):
        o, h, l, c = _ohlc(40, seed=20)
        obj = ctor()
        args = (h, l) if n_inputs == 2 else (h, l, c) if n_inputs == 3 else (o, h, l, c)
        first = obj(*args)
        obj.reset()
        second = obj(*args)
        np.testing.assert_array_equal(first, second)
