"""
Tests for the True Range / ATR / NATR family (Wilder).

  TR[t] = max(H - L, |H - prev_C|, |L - prev_C|)
  ATR    -- Wilder-smoothed rolling average of TR
  NATR   -- 100 * ATR / close
"""
import numpy as np
import pandas as pd
import pytest

from screamer import TrueRange, ATR, NATR


def _hlc(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    return high, low, close


def ref_tr(high, low, close):
    """Plain numpy TR reference. TR[0] = NaN."""
    out = np.full(len(close), np.nan)
    for t in range(1, len(close)):
        out[t] = max(high[t] - low[t],
                     abs(high[t] - close[t - 1]),
                     abs(low[t]  - close[t - 1]))
    return out


def ref_atr(high, low, close, n):
    """Plain numpy ATR reference using Wilder smoothing with SMA seed."""
    tr = ref_tr(high, low, close)
    out = np.full(len(close), np.nan)
    # Need n TR values (indices 1..n inclusive) for the SMA seed at index n.
    if len(close) <= n:
        return out
    out[n] = np.mean(tr[1:n + 1])
    for t in range(n + 1, len(close)):
        out[t] = ((n - 1) * out[t - 1] + tr[t]) / n
    return out


# ---------------------------------------------------------------------------
# TrueRange
# ---------------------------------------------------------------------------

class TestTrueRange:

    def test_first_sample_is_nan(self):
        h, l, c = _hlc(20, seed=0)
        out = TrueRange()(h, l, c)
        assert np.isnan(out[0])
        assert np.all(np.isfinite(out[1:]))

    def test_matches_manual_reference(self):
        h, l, c = _hlc(100, seed=1)
        ours = TrueRange()(h, l, c)
        ref = ref_tr(h, l, c)
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_no_overnight_gap_simplifies_to_h_minus_l(self):
        """If close == prev_close (no overnight movement), TR collapses to H - L."""
        c = np.full(10, 5.0)
        h = np.full(10, 6.0)
        l = np.full(10, 4.0)
        out = TrueRange()(h, l, c)
        # TR[0] is NaN. From t=1 onward TR = max(2, 1, 1) = 2.
        np.testing.assert_array_equal(out[1:], 2.0)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:

    @pytest.mark.parametrize("n", [5, 10, 14, 20])
    def test_matches_manual_reference(self, n):
        h, l, c = _hlc(150, seed=n)
        ours = ATR(n)(h, l, c)
        ref = ref_atr(h, l, c, n)
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_first_valid_at_window_size(self):
        n = 14
        h, l, c = _hlc(50, seed=2)
        out = ATR(n)(h, l, c)
        assert np.all(np.isnan(out[:n]))
        assert np.all(np.isfinite(out[n:]))

    def test_constant_input_atr_is_zero(self):
        """If H == L == C (no movement), every TR is 0, so ATR is 0."""
        x = np.full(30, 5.0)
        out = ATR(14)(x, x, x)
        # First 14 samples are NaN warmup; rest is 0.
        np.testing.assert_array_equal(out[14:], 0.0)

    def test_atr_is_non_negative(self):
        h, l, c = _hlc(100, seed=3)
        out = ATR(14)(h, l, c)
        finite = out[~np.isnan(out)]
        assert np.all(finite >= 0.0)

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            ATR(1)


# ---------------------------------------------------------------------------
# NATR
# ---------------------------------------------------------------------------

class TestNATR:

    def test_equals_100_times_atr_over_close(self):
        h, l, c = _hlc(80, seed=4)
        atr = ATR(14)(h, l, c)
        natr = NATR(14)(h, l, c)
        np.testing.assert_allclose(natr, 100.0 * atr / c, equal_nan=True, atol=1e-12)

    def test_first_valid_at_window_size(self):
        n = 14
        h, l, c = _hlc(50, seed=5)
        out = NATR(n)(h, l, c)
        assert np.all(np.isnan(out[:n]))
        assert np.all(np.isfinite(out[n:]))


# ---------------------------------------------------------------------------
# Dispatcher / parity
# ---------------------------------------------------------------------------

class TestDispatcher:

    def test_truerange_scalar_triple(self):
        out = TrueRange()(10.0, 8.0, 9.0)
        assert np.isnan(out)  # first call -- no prev_close

    def test_2d_per_column(self):
        rng = np.random.default_rng(6)
        H = 100 + rng.standard_normal((40, 3)).cumsum(axis=0) + 1.0
        L = H - rng.uniform(0.5, 1.5, (40, 3))
        C = L + (H - L) * rng.uniform(0.0, 1.0, (40, 3))
        for cls in (TrueRange, lambda: ATR(10), lambda: NATR(10)):
            obj = cls() if callable(cls) else cls()
            out_2d = obj(H, L, C)
            for k in range(3):
                ref_obj = cls() if callable(cls) else cls()
                np.testing.assert_allclose(
                    out_2d[:, k],
                    ref_obj(H[:, k].copy(), L[:, k].copy(), C[:, k].copy()),
                    equal_nan=True, atol=1e-12,
                )


class TestParity:

    @pytest.mark.parametrize("ctor", [
        lambda: TrueRange(),
        lambda: ATR(10),
        lambda: NATR(10),
    ])
    def test_reset_clears_history(self, ctor):
        h, l, c = _hlc(40, seed=7)
        obj = ctor()
        first = obj(h, l, c)
        obj.reset()
        second = obj(h, l, c)
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("ctor", [
        lambda: TrueRange(),
        lambda: ATR(10),
        lambda: NATR(10),
    ])
    def test_scalar_loop_matches_array(self, ctor):
        h, l, c = _hlc(30, seed=8)
        obj = ctor()
        streamed = np.array([obj(hh, ll, cc) for hh, ll, cc in zip(h, l, c)])
        arr_result = ctor()(h, l, c)
        np.testing.assert_allclose(streamed, arr_result,
                                   equal_nan=True, atol=1e-12)
