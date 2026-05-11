"""
Tests for the performance / risk metric batch:

  Drawdown, MaxDrawdown      -- cumulative; compose CumMax/CumMin
  RollingMaxDrawdown(w)      -- worst peak-to-trough loss in last w bars
  RollingSharpe(w, ppy)      -- annualised mean/std ratio
  RollingSortino(w, ppy, t)  -- downside-only variant
  RollingInfoRatio(w, ppy)   -- vs benchmark, 2-input
  RollingCalmar(w, ppy)      -- annualised mean / |rolling max DD|
  RollingHitRate(w)          -- fraction of positive samples
"""
import numpy as np
import pandas as pd
import pytest

from screamer import (
    Drawdown, MaxDrawdown, RollingMaxDrawdown,
    RollingSharpe, RollingSortino, RollingInfoRatio,
    RollingCalmar, RollingHitRate,
)


def _returns(n, seed=0):
    return np.random.default_rng(seed).normal(0.001, 0.02, n)


def _price(n, seed=0):
    return 100 * np.cumprod(1 + _returns(n, seed))


# ---------------------------------------------------------------------------
# Drawdown / MaxDrawdown (cumulative)
# ---------------------------------------------------------------------------

class TestDrawdown:

    def test_matches_pandas_cummax(self):
        p = _price(200, seed=0)
        ours = Drawdown()(p)
        ref = p / pd.Series(p).cummax().to_numpy() - 1
        np.testing.assert_allclose(ours, ref, atol=1e-14)

    def test_always_non_positive(self):
        p = _price(200, seed=1)
        out = Drawdown()(p)
        assert np.all(out <= 1e-14)

    def test_zero_at_new_peak(self):
        p = np.array([100.0, 95.0, 110.0, 105.0])
        out = Drawdown()(p)
        np.testing.assert_allclose(out, [0.0, -0.05, 0.0, -5/110], atol=1e-14)


class TestMaxDrawdown:

    def test_matches_cummin_of_drawdown(self):
        p = _price(200, seed=2)
        ours = MaxDrawdown()(p)
        ref = pd.Series(Drawdown()(p)).cummin().to_numpy()
        np.testing.assert_allclose(ours, ref, atol=1e-14)

    def test_monotonically_non_increasing(self):
        """MaxDrawdown can only go down (or stay flat) over time."""
        p = _price(200, seed=3)
        out = MaxDrawdown()(p)
        assert np.all(np.diff(out) <= 1e-14)


# ---------------------------------------------------------------------------
# RollingMaxDrawdown (windowed)
# ---------------------------------------------------------------------------

class TestRollingMaxDrawdown:

    @pytest.mark.parametrize("w", [10, 20, 50])
    def test_matches_manual_within_window(self, w):
        p = _price(200, seed=w)
        ours = RollingMaxDrawdown(w)(p)
        ref = np.full(200, np.nan)
        for t in range(w - 1, 200):
            win = p[t - w + 1:t + 1]
            peak_w = np.maximum.accumulate(win)
            dd_w = win / peak_w - 1
            ref[t] = dd_w.min()
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-14)

    def test_no_drawdown_for_monotonic(self):
        p = np.arange(100, 200, dtype=float)
        out = RollingMaxDrawdown(10)(p)
        np.testing.assert_array_equal(out[9:], 0.0)

    def test_output_bounded(self):
        p = _price(200, seed=4)
        out = RollingMaxDrawdown(20)(p)
        finite = out[~np.isnan(out)]
        assert np.all(finite <= 1e-14)
        assert np.all(finite > -1.0 - 1e-14)


# ---------------------------------------------------------------------------
# Sharpe / Sortino / InfoRatio / Calmar / HitRate
# ---------------------------------------------------------------------------

class TestRollingSharpe:

    @pytest.mark.parametrize("w,ppy", [(10, 1.0), (20, 252), (50, 252)])
    def test_matches_pandas_composition(self, w, ppy):
        r = _returns(200, seed=w)
        ours = RollingSharpe(w, ppy)(r)
        s = pd.Series(r)
        ref = np.sqrt(ppy) * s.rolling(w).mean() / s.rolling(w).std(ddof=1)
        np.testing.assert_allclose(ours, ref.to_numpy(), equal_nan=True, atol=1e-12)

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            RollingSharpe(1, 252)
        with pytest.raises(ValueError):
            RollingSharpe(10, -1.0)


class TestRollingSortino:

    @pytest.mark.parametrize("w", [10, 20, 50])
    def test_matches_manual_reference(self, w):
        r = _returns(200, seed=w + 100)
        ppy = 252
        ours = RollingSortino(w, ppy, target=0.0)(r)
        out = np.full(200, np.nan)
        for t in range(w - 1, 200):
            win = r[t - w + 1:t + 1]
            mean = win.mean()
            downside = np.where(win < 0, win, 0.0)
            rms = np.sqrt(np.mean(downside ** 2))
            if rms > 0:
                out[t] = np.sqrt(ppy) * mean / rms
        np.testing.assert_allclose(ours, out, equal_nan=True, atol=1e-12)

    def test_target_shifts_baseline(self):
        """Higher target -> lower mean - target -> lower Sortino."""
        r = _returns(100, seed=5)
        s_low = RollingSortino(20, 252, target=0.0)(r)
        s_high = RollingSortino(20, 252, target=0.01)(r)
        finite_low = s_low[~np.isnan(s_low)]
        finite_high = s_high[~np.isnan(s_high)]
        # On most samples, higher target gives lower Sortino.
        assert np.mean(finite_high < finite_low) > 0.7


class TestRollingInfoRatio:

    def test_matches_manual_reference(self):
        r = _returns(200, seed=6)
        b = _returns(200, seed=7)
        ours = RollingInfoRatio(20, 252)(r, b)
        excess = pd.Series(r - b)
        ref = np.sqrt(252) * excess.rolling(20).mean() / excess.rolling(20).std(ddof=1)
        np.testing.assert_allclose(ours, ref.to_numpy(), equal_nan=True, atol=1e-12)

    def test_zero_when_returns_equal_benchmark(self):
        r = _returns(60, seed=8)
        out = RollingInfoRatio(20, 252)(r, r)
        # Excess = 0 always -> std = 0 -> NaN.
        assert np.all(np.isnan(out))


class TestRollingCalmar:

    def test_sane_values(self):
        r = _returns(200, seed=9)
        out = RollingCalmar(50, 252)(r)
        # Just sanity: shape + finite where defined.
        assert out.shape == (200,)
        # Some finite values exist after warmup.
        assert np.sum(np.isfinite(out)) > 0


class TestRollingHitRate:

    @pytest.mark.parametrize("w", [5, 10, 20])
    def test_matches_pandas(self, w):
        r = _returns(100, seed=w + 10)
        ours = RollingHitRate(w)(r)
        ref = pd.Series(r > 0).rolling(w).mean().to_numpy()
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-15)

    def test_bounded_zero_to_one(self):
        r = _returns(200, seed=11)
        out = RollingHitRate(20)(r)
        finite = out[~np.isnan(out)]
        assert np.all(finite >= 0.0)
        assert np.all(finite <= 1.0)

    def test_all_positive_gives_one(self):
        out = RollingHitRate(5)(np.arange(1, 20.0))
        np.testing.assert_array_equal(out[4:], 1.0)

    def test_all_negative_gives_zero(self):
        out = RollingHitRate(5)(-np.arange(1, 20.0))
        np.testing.assert_array_equal(out[4:], 0.0)


# ---------------------------------------------------------------------------
# Reset / scalar-loop parity
# ---------------------------------------------------------------------------

class TestParity:

    @pytest.mark.parametrize("ctor,is_two_input", [
        (lambda: Drawdown(),                       False),
        (lambda: MaxDrawdown(),                    False),
        (lambda: RollingMaxDrawdown(10),           False),
        (lambda: RollingSharpe(10, 252),           False),
        (lambda: RollingSortino(10, 252),          False),
        (lambda: RollingInfoRatio(10, 252),        True),
        (lambda: RollingCalmar(10, 252),           False),
        (lambda: RollingHitRate(10),               False),
    ])
    def test_reset_clears_history(self, ctor, is_two_input):
        r = _returns(40, seed=20)
        obj = ctor()
        if is_two_input:
            b = _returns(40, seed=21)
            first = obj(r, b)
            obj.reset()
            second = obj(r, b)
        else:
            first = obj(r if isinstance(obj, (RollingSharpe, RollingSortino,
                                              RollingCalmar, RollingHitRate)) else _price(40, seed=20))
            obj.reset()
            second = obj(r if isinstance(obj, (RollingSharpe, RollingSortino,
                                               RollingCalmar, RollingHitRate)) else _price(40, seed=20))
        np.testing.assert_array_equal(first, second)
