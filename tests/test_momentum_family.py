"""
Tests for the simple momentum / rate-of-change family:

  Momentum(k)  -- alias of Diff(k):  x[t] - x[t-k]
  ROC(k)       -- 100 * (x[t] - x[t-k]) / x[t-k]
  ROCP(k)      -- (x[t] - x[t-k]) / x[t-k]            (alias of Return)
  ROCR(k)      -- x[t] / x[t-k]
  TRIX(span)   -- 1-step rate of change of triple-smoothed EMA

The first four are all O(1), exact arithmetic with no smoothing
subtleties. TRIX inherits the EwMean adjust=True convention from
the moving-average family, so it differs from TA-Lib's TRIX during
warmup -- same convention split as DEMA / TEMA / MACD.

Cross-validation against TA-Lib lives in
tests/test_third_party_alignment.py.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import (
    Momentum, ROC, ROCP, ROCR, TRIX,
    Diff, Return, EwMean,
)


def _walk(n, seed=0):
    return 100 + np.random.default_rng(seed).standard_normal(n).cumsum()


# ---------------------------------------------------------------------------
# Aliases: Momentum and ROCP
# ---------------------------------------------------------------------------

class TestAliases:

    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_momentum_equals_diff(self, k):
        x = _walk(80, seed=k)
        np.testing.assert_array_equal(Momentum(k)(x), Diff(k)(x))

    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_rocp_equals_return(self, k):
        x = _walk(80, seed=k + 100)
        np.testing.assert_array_equal(ROCP(k)(x), Return(k)(x))


# ---------------------------------------------------------------------------
# ROC family formula correctness
# ---------------------------------------------------------------------------

class TestROCFamily:

    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_roc_matches_manual(self, k):
        x = _walk(80, seed=k + 200)
        ours = ROC(k)(x)
        ref = np.full(len(x), np.nan)
        ref[k:] = 100.0 * (x[k:] - x[:-k]) / x[:-k]
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_rocp_matches_manual(self, k):
        x = _walk(80, seed=k + 300)
        ours = ROCP(k)(x)
        ref = np.full(len(x), np.nan)
        ref[k:] = (x[k:] - x[:-k]) / x[:-k]
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_rocr_matches_manual(self, k):
        x = _walk(80, seed=k + 400)
        ours = ROCR(k)(x)
        ref = np.full(len(x), np.nan)
        ref[k:] = x[k:] / x[:-k]
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_roc_is_100_times_rocp(self):
        x = _walk(50, seed=0)
        np.testing.assert_allclose(ROC(5)(x), 100.0 * ROCP(5)(x),
                                   equal_nan=True, atol=1e-12)

    def test_rocr_minus_one_equals_rocp(self):
        x = _walk(50, seed=1)
        np.testing.assert_allclose(ROCR(5)(x) - 1.0, ROCP(5)(x),
                                   equal_nan=True, atol=1e-12)


# ---------------------------------------------------------------------------
# TRIX
# ---------------------------------------------------------------------------

class TestTRIX:

    @pytest.mark.parametrize("span", [5, 10, 14, 20])
    def test_matches_manual_composition(self, span):
        x = _walk(150, seed=span)
        ours = TRIX(span)(x)
        # Manual reference: pandas adjust=True three times, then 100*ROC(1).
        s = pd.Series(x)
        e1 = s.ewm(span=span, adjust=True).mean()
        e2 = e1.ewm(span=span, adjust=True).mean()
        e3 = e2.ewm(span=span, adjust=True).mean()
        ref = 100.0 * e3.pct_change()
        np.testing.assert_allclose(ours, ref.to_numpy(),
                                   equal_nan=True, atol=1e-12)

    def test_first_sample_is_nan(self):
        """No previous ema3 at t=0, so TRIX[0] is NaN."""
        out = TRIX(10)(_walk(20, seed=0))
        assert np.isnan(out[0])
        assert np.all(np.isfinite(out[1:]))

    def test_constant_input_is_zero(self):
        """For a constant input c, every EMA stage returns c, so
        ema3 is constant and TRIX = 0 from t=1 onward."""
        out = TRIX(10)(np.full(30, 5.7))
        # t=0 is NaN (no prev_ema3 yet); from t=1 on, 0/0 from
        # (5.7 - 5.7) / 5.7 = 0. Need to skip NaN at t=0.
        np.testing.assert_allclose(out[1:], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Dispatcher / parity / constructor (one block per class)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,arg", [
    (Momentum, 5),
    (ROC,      5),
    (ROCP,     5),
    (ROCR,     5),
    (TRIX,    10),
])
class TestParity:

    def test_scalar_loop_matches_array(self, cls, arg):
        x = _walk(40, seed=arg)
        obj = cls(arg)
        streamed = np.array([obj(v) for v in x])
        np.testing.assert_allclose(streamed, cls(arg)(x),
                                   equal_nan=True, atol=1e-12)

    def test_reset_clears_history(self, cls, arg):
        x = _walk(30, seed=arg)
        obj = cls(arg)
        first = obj(x)
        obj.reset()
        second = obj(x)
        np.testing.assert_array_equal(first, second)

    def test_2d_per_column_independence(self, cls, arg):
        X = np.column_stack([_walk(30, seed=arg + i) for i in range(3)])
        out_2d = cls(arg)(X)
        for k in range(3):
            np.testing.assert_allclose(
                out_2d[:, k], cls(arg)(X[:, k].copy()),
                equal_nan=True, atol=1e-12,
            )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:

    @pytest.mark.parametrize("cls", [Momentum, ROC, ROCP, ROCR])
    def test_zero_window_raises(self, cls):
        with pytest.raises(ValueError):
            cls(0)

    def test_trix_zero_span_raises(self):
        with pytest.raises(ValueError):
            TRIX(0)
