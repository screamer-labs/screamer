"""
Tests for KAMA: Kaufman's Adaptive Moving Average.

KAMA's smoothing constant adapts to the "efficiency" of the recent
price action. We validate against:

  1. A manual numpy reference implementing the formula step-by-step
     (no dependency on third-party libraries).
  2. TA-Lib's KAMA in tests/test_third_party_alignment.py.
  3. pandas-ta-classic's kama in tests/test_third_party_alignment.py.
"""
import numpy as np
import pytest

from screamer import KAMA


# ---------------------------------------------------------------------------
# Manual numpy reference
# ---------------------------------------------------------------------------

def ref_kama(x, n, fast=2, slow=30):
    """Direct translation of the KAMA recurrence. Seeded at sample n
    with previous KAMA = x[n-1] (matching TA-Lib)."""
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)
    alpha_diff = fast_alpha - slow_alpha
    abs_deltas = np.abs(np.diff(x))
    prev_kama = x[n - 1]
    for t in range(n, len(x)):
        direction = abs(x[t] - x[t - n])
        volatility = abs_deltas[t - n:t].sum()  # n absolute one-step deltas
        er = direction / volatility if volatility > 0 else 0.0
        sc_root = er * alpha_diff + slow_alpha
        sc = sc_root * sc_root
        prev_kama = prev_kama + sc * (x[t] - prev_kama)
        out[t] = prev_kama
    return out


# ---------------------------------------------------------------------------
# Manual reference parity (the strict validation)
# ---------------------------------------------------------------------------

class TestManualReference:

    @pytest.mark.parametrize("n", [5, 10, 14, 30])
    def test_default_constants_match_reference(self, n):
        rng = np.random.default_rng(n)
        x = rng.standard_normal(200).cumsum()
        np.testing.assert_allclose(KAMA(n)(x), ref_kama(x, n),
                                   equal_nan=True, atol=1e-12)

    @pytest.mark.parametrize("fast,slow", [(2, 30), (3, 50), (5, 20)])
    def test_custom_fast_slow_match_reference(self, fast, slow):
        rng = np.random.default_rng(fast * 100 + slow)
        x = rng.standard_normal(150).cumsum()
        n = 10
        np.testing.assert_allclose(
            KAMA(n, fast=fast, slow=slow)(x),
            ref_kama(x, n, fast=fast, slow=slow),
            equal_nan=True, atol=1e-12,
        )

    def test_default_kwargs(self):
        """Verify the documented defaults (window_size=anyway, fast=2, slow=30)."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(80)
        n = 14
        np.testing.assert_array_equal(KAMA(n)(x),
                                      KAMA(n, fast=2, slow=30)(x))


# ---------------------------------------------------------------------------
# Warmup and edge cases
# ---------------------------------------------------------------------------

class TestWarmup:

    def test_first_valid_at_index_n(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(30).cumsum()
        n = 10
        out = KAMA(n)(x)
        assert np.all(np.isnan(out[:n]))
        assert np.all(np.isfinite(out[n:]))

    def test_constant_input_is_constant(self):
        """If x is constant c, every delta is 0, volatility is 0, and
        ER = 0 by our convention. SC = slow_alpha^2. KAMA recurrence
        starts with kama_=x[n-1]=c, and updates by sc*(c-c)=0 each
        step, so KAMA stays at c."""
        out = KAMA(10)(np.full(30, 3.7))
        np.testing.assert_allclose(out[10:], 3.7, atol=1e-12)

    def test_monotonic_input_efficiency_ratio_is_one(self):
        """For x[t] = t, every step has delta=+1 and net direction =
        n. ER = n/n = 1, SC = fast_alpha^2 (= 4/9 with default
        fast=2). KAMA converges toward the price level."""
        x = np.arange(50.0)
        out = KAMA(10)(x)
        # After enough steps post-warmup, the gap (x[t] - KAMA[t])
        # should be small (KAMA tracks the line closely with the fast SC).
        gap = x[-1] - out[-1]
        assert 0 < gap < 5.0   # tracks closely but with some lag


# ---------------------------------------------------------------------------
# Cross-mode parity
# ---------------------------------------------------------------------------

class TestParity:

    def test_scalar_loop_matches_array(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(60).cumsum()
        n = 10
        obj = KAMA(n)
        streamed = np.array([obj(v) for v in x])
        np.testing.assert_allclose(streamed, KAMA(n)(x),
                                   equal_nan=True, atol=1e-12)

    def test_2d_per_column_independence(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 3)).cumsum(axis=0)
        n = 10
        out_2d = KAMA(n)(X)
        for k in range(3):
            np.testing.assert_allclose(
                out_2d[:, k], KAMA(n)(X[:, k].copy()),
                equal_nan=True, atol=1e-12,
            )

    def test_reset_clears_history(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(40).cumsum()
        obj = KAMA(10)
        first = np.array([obj(v) for v in x])
        obj.reset()
        second = np.array([obj(v) for v in x])
        np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_window_size_below_2_raises(self):
        with pytest.raises(ValueError):
            KAMA(1)

    def test_fast_zero_raises(self):
        with pytest.raises(ValueError):
            KAMA(10, fast=0)

    def test_slow_zero_raises(self):
        with pytest.raises(ValueError):
            KAMA(10, slow=0)
