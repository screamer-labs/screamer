"""
Tests for RollingRSI (Wilder default, Cutler opt-in).

Cross-validation against external references for both methods lives
in tests/test_third_party_alignment.py; this file pins the
internal behaviour without requiring TA-Lib / pandas-ta to be
installed.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import RollingRSI


# ---------------------------------------------------------------------------
# Reference implementations (plain numpy)
# ---------------------------------------------------------------------------

def ref_wilder_rsi(x, n):
    """Wilder's RSI, TA-Lib-aligned:
       - first n samples produce NaN
       - at sample n, seed avg_gain/avg_loss = mean of first n gains
       - thereafter, avg_gain[t] = ((n-1)*avg_gain[t-1] + gain[t]) / n
    """
    out = np.full(len(x), np.nan)
    deltas = np.diff(x)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    if len(x) <= n:
        return out
    avg_gain = gains[:n].mean()
    avg_loss = losses[:n].mean()
    out[n] = 100.0 * avg_gain / (avg_gain + avg_loss) if (avg_gain + avg_loss) else 50.0
    for t in range(n + 1, len(x)):
        g = gains[t - 1]
        l = losses[t - 1]
        avg_gain = ((n - 1) * avg_gain + g) / n
        avg_loss = ((n - 1) * avg_loss + l) / n
        total = avg_gain + avg_loss
        out[t] = 100.0 * avg_gain / total if total else 50.0
    return out


def ref_cutler_rsi(x, n):
    """Cutler's RSI as our implementation defines it: rolling SMA of
    gains/losses including the t=0 "missing-delta" zero. First valid
    at sample n-1."""
    deltas = np.diff(x, prepend=np.nan)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = pd.Series(gains).rolling(n).mean()
    avg_loss = pd.Series(losses).rolling(n).mean()
    total = avg_gain + avg_loss
    return (100.0 * avg_gain / total).to_numpy()


# ---------------------------------------------------------------------------
# Wilder (default)
# ---------------------------------------------------------------------------

class TestWilder:

    @pytest.mark.parametrize("n", [5, 10, 14, 20])
    def test_matches_manual_wilder_reference(self, n):
        rng = np.random.default_rng(n)
        x = rng.standard_normal(200)
        ours = RollingRSI(n)(x)
        ref = ref_wilder_rsi(x, n)
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_warmup_is_nan(self):
        n = 10
        out = RollingRSI(n)(np.arange(20.0))
        assert np.all(np.isnan(out[:n]))
        assert np.all(np.isfinite(out[n:]))

    def test_default_method_is_wilder(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(50)
        np.testing.assert_array_equal(RollingRSI(10)(x),
                                      RollingRSI(10, method="wilder")(x))

    def test_bounded_zero_to_hundred(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(200)
        out = RollingRSI(14)(x)
        finite = out[~np.isnan(out)]
        assert np.all(finite >= 0.0)
        assert np.all(finite <= 100.0)

    def test_monotonic_input_saturates_to_100(self):
        """Strictly increasing input has no losses, so RSI = 100 after warmup."""
        x = np.arange(50.0)
        out = RollingRSI(10)(x)
        np.testing.assert_allclose(out[10:], 100.0, atol=1e-12)

    def test_monotonic_decreasing_saturates_to_0(self):
        x = -np.arange(50.0)
        out = RollingRSI(10)(x)
        np.testing.assert_allclose(out[10:], 0.0, atol=1e-12)

    def test_flat_input_returns_50(self):
        """No gains and no losses -> RSI = 50 by convention."""
        out = RollingRSI(10)(np.full(30, 5.0))
        finite = out[~np.isnan(out)]
        np.testing.assert_allclose(finite, 50.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Cutler (opt-in)
# ---------------------------------------------------------------------------

class TestCutler:

    @pytest.mark.parametrize("n", [5, 10, 14, 20])
    def test_matches_manual_cutler_reference(self, n):
        rng = np.random.default_rng(n + 100)
        x = rng.standard_normal(200)
        ours = RollingRSI(n, method="cutler")(x)
        ref = ref_cutler_rsi(x, n)
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_cutler_first_valid_is_one_earlier_than_wilder(self):
        """Cutler emits at index n-1; Wilder at index n."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(20)
        n = 5
        cutler = RollingRSI(n, method="cutler")(x)
        wilder = RollingRSI(n, method="wilder")(x)
        assert np.isfinite(cutler[n - 1])
        assert np.isnan(wilder[n - 1])
        assert np.isfinite(wilder[n])


# ---------------------------------------------------------------------------
# Cross-method sanity
# ---------------------------------------------------------------------------

class TestCrossMethod:

    def test_wilder_and_cutler_converge_for_stationary_series(self):
        """For a long stationary series, Wilder and Cutler tend toward the
        same value but the gap shrinks only slowly (Wilder has effective
        lookback n / alpha = n^2 / 1 samples). Check the gap is sane."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(2000)
        n = 14
        wilder = RollingRSI(n)(x)
        cutler = RollingRSI(n, method="cutler")(x)
        mask = ~(np.isnan(wilder) | np.isnan(cutler))
        diff = np.abs(wilder[mask] - cutler[mask])
        assert diff.mean() < 5.0   # mean disagreement well under 5 points

    @pytest.mark.parametrize("method", ["wilder", "cutler"])
    def test_scalar_loop_matches_array(self, method):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(50)
        obj = RollingRSI(10, method=method)
        streamed = np.array([obj(v) for v in x])
        np.testing.assert_allclose(
            streamed, RollingRSI(10, method=method)(x),
            equal_nan=True, atol=1e-12,
        )

    @pytest.mark.parametrize("method", ["wilder", "cutler"])
    def test_2d_per_column_independence(self, method):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((40, 3))
        out_2d = RollingRSI(10, method=method)(X)
        for k in range(3):
            np.testing.assert_allclose(
                out_2d[:, k], RollingRSI(10, method=method)(X[:, k].copy()),
                equal_nan=True, atol=1e-12,
            )

    @pytest.mark.parametrize("method", ["wilder", "cutler"])
    def test_reset_clears_history(self, method):
        rng = np.random.default_rng(6)
        x = rng.standard_normal(40)
        obj = RollingRSI(10, method=method)
        first = np.array([obj(v) for v in x])
        obj.reset()
        second = np.array([obj(v) for v in x])
        np.testing.assert_array_equal(first, second)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            RollingRSI(10, method="ema")
