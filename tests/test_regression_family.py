"""
Tests for the regression / statistical batch:

  RollingAlpha             -- 2->1, intercept of target ~ slope*regressor + alpha
  RollingResidualStd       -- 2->1, std of RollingSpread series
  RollingLinearRegression  -- 2->4, (slope, intercept, r_squared, stderr)
  RollingTSF               -- 1->1, regression vs time, one step ahead
  RollingRank              -- 1->1, pandas-style rolling rank
  RollingPercentile        -- 1->1, RollingRank / w (pandas pct=True)
"""
import numpy as np
import pandas as pd
import pytest

from screamer import (
    RollingAlpha, RollingResidualStd, RollingLinearRegression,
    RollingTSF, RollingRank, RollingPercentile,
    RollingBeta, RollingSpread,
)


def _pair(n, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    x = 0.5 * y + 0.3 * rng.standard_normal(n)
    return y, x


# ---------------------------------------------------------------------------
# RollingAlpha (composition of RollingBeta + RollingMean)
# ---------------------------------------------------------------------------

class TestRollingAlpha:

    @pytest.mark.parametrize("w", [10, 20, 50])
    def test_matches_manual_definition(self, w):
        y, x = _pair(200, seed=w)
        ours = RollingAlpha(w)(y, x)
        beta = RollingBeta(w)(y, x)
        my = pd.Series(y).rolling(w).mean().to_numpy()
        mx = pd.Series(x).rolling(w).mean().to_numpy()
        ref = my - beta * mx
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)


# ---------------------------------------------------------------------------
# RollingResidualStd (composition of RollingSpread + RollingStd)
# ---------------------------------------------------------------------------

class TestRollingResidualStd:

    @pytest.mark.parametrize("w", [10, 20, 50])
    def test_matches_manual_definition(self, w):
        y, x = _pair(200, seed=w + 100)
        ours = RollingResidualStd(w)(y, x)
        spread = RollingSpread(w)(y, x)
        ref = pd.Series(spread).rolling(w).std(ddof=1).to_numpy()
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_no_nan_poisoning_post_warmup(self):
        """Spread emits NaN during its own warmup -- ResidualStd must
        gate against feeding NaN into the std accumulator."""
        y, x = _pair(120, seed=0)
        w = 20
        out = RollingResidualStd(w)(y, x)
        # After Spread+Std warmup, output should be finite somewhere.
        assert np.any(np.isfinite(out))


# ---------------------------------------------------------------------------
# RollingLinearRegression (2->4)
# ---------------------------------------------------------------------------

class TestRollingLinearRegression:

    @pytest.mark.parametrize("w", [10, 20, 50])
    def test_slope_intercept_r2_match_scipy(self, w):
        import scipy.stats
        y, x = _pair(200, seed=w + 200)
        ours = RollingLinearRegression(w)(y, x)
        for t in (w - 1, w + 20, 150, 199):
            res = scipy.stats.linregress(x[t - w + 1:t + 1], y[t - w + 1:t + 1])
            assert ours[t, 0] == pytest.approx(res.slope, abs=1e-10)
            assert ours[t, 1] == pytest.approx(res.intercept, abs=1e-10)
            assert ours[t, 2] == pytest.approx(res.rvalue ** 2, abs=1e-10)

    @pytest.mark.parametrize("w", [10, 20, 50])
    def test_stderr_is_rmse_of_residuals(self, w):
        """Our `stderr` is the standard error of estimate = sqrt(SSE/(n-2))
        (RMSE of residuals), NOT scipy's slope-stderr."""
        y, x = _pair(200, seed=w + 300)
        ours = RollingLinearRegression(w)(y, x)
        t = 150
        slope = ours[t, 0]
        intercept = ours[t, 1]
        residuals = y[t - w + 1:t + 1] - slope * x[t - w + 1:t + 1] - intercept
        manual_rmse = np.sqrt((residuals ** 2).sum() / (w - 2))
        assert ours[t, 3] == pytest.approx(manual_rmse, abs=1e-12)

    def test_output_shape(self):
        y, x = _pair(50, seed=0)
        out = RollingLinearRegression(10)(y, x)
        assert out.shape == (50, 4)

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            RollingLinearRegression(2)


# ---------------------------------------------------------------------------
# RollingTSF
# ---------------------------------------------------------------------------

class TestRollingTSF:

    @pytest.mark.parametrize("w", [5, 10, 14, 20])
    def test_matches_manual_regression_one_ahead(self, w):
        rng = np.random.default_rng(w)
        y = rng.standard_normal(120)
        ours = RollingTSF(w)(y)
        ref = np.full(120, np.nan)
        t_local = np.arange(w, dtype=float)
        for t in range(w - 1, 120):
            y_win = y[t - w + 1:t + 1]
            slope, intercept = np.polyfit(t_local, y_win, 1)
            # Project to local time = w (one step beyond).
            ref[t] = slope * w + intercept
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-10)

    def test_warmup_is_window_minus_one(self):
        out = RollingTSF(10)(np.arange(20.0))
        assert np.all(np.isnan(out[:9]))
        assert np.all(np.isfinite(out[9:]))


# ---------------------------------------------------------------------------
# RollingRank / RollingPercentile
# ---------------------------------------------------------------------------

class TestRollingRank:

    @pytest.mark.parametrize("w", [5, 10, 20])
    def test_matches_pandas(self, w):
        rng = np.random.default_rng(w + 400)
        y = rng.standard_normal(100)
        ours = RollingRank(w)(y)
        ref = pd.Series(y).rolling(w).rank().to_numpy()
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_bounded_one_to_w(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)
        out = RollingRank(20)(y)
        finite = out[~np.isnan(out)]
        assert np.all(finite >= 1.0)
        assert np.all(finite <= 20.0)


class TestRollingPercentile:

    @pytest.mark.parametrize("w", [5, 10, 20])
    def test_matches_pandas(self, w):
        rng = np.random.default_rng(w + 500)
        y = rng.standard_normal(100)
        ours = RollingPercentile(w)(y)
        ref = pd.Series(y).rolling(w).rank(pct=True).to_numpy()
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_equals_rank_over_w(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal(100)
        w = 20
        rank = RollingRank(w)(y)
        pct = RollingPercentile(w)(y)
        np.testing.assert_allclose(pct, rank / w, equal_nan=True, atol=1e-15)


# ---------------------------------------------------------------------------
# Reset / scalar-loop parity
# ---------------------------------------------------------------------------

class TestParity:

    @pytest.mark.parametrize("ctor,is_two", [
        (lambda: RollingAlpha(10),             True),
        (lambda: RollingResidualStd(10),       True),
        (lambda: RollingLinearRegression(10),  True),
        (lambda: RollingTSF(10),               False),
        (lambda: RollingRank(10),              False),
        (lambda: RollingPercentile(10),        False),
    ])
    def test_reset_clears_history(self, ctor, is_two):
        y, x = _pair(40, seed=30)
        obj = ctor()
        if is_two:
            first = obj(y, x)
            obj.reset()
            second = obj(y, x)
        else:
            first = obj(y)
            obj.reset()
            second = obj(y)
        np.testing.assert_array_equal(first, second)
