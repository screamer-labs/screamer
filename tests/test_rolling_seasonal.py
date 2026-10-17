import math

import numpy as np
import pytest

from devtools.baselines.RollingSeasonalMean import RollingSeasonalMean_numpy
from devtools.baselines.RollingSeasonalStd import RollingSeasonalStd_numpy
from devtools.baselines.RollingSeasonalZscore import RollingSeasonalZscore_numpy
from screamer import RollingSeasonalMean, RollingSeasonalStd, RollingSeasonalZscore, RollingZscore
from tests.regime_helpers import assert_regimes_agree

X = np.array([1.0, 10.0, 3.0, 20.0, 5.0, 30.0])
S2 = math.sqrt(2.0)
S50 = math.sqrt(50.0)


def test_mean_hand_example():
    out = np.asarray(RollingSeasonalMean(period=2, window_size=2)(X))
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [2.0, 15.0, 4.0, 25.0], rtol=1e-12)


def test_std_and_zscore_hand_example():
    sd = np.asarray(RollingSeasonalStd(period=2, window_size=2)(X))
    np.testing.assert_allclose(sd[2:], [S2, S50, S2, S50], rtol=1e-12)
    z = np.asarray(RollingSeasonalZscore(period=2, window_size=2)(X))
    np.testing.assert_allclose(z[2:], [1 / S2] * 4, rtol=1e-12)


def test_expanding_and_zero():
    exp = np.asarray(RollingSeasonalMean(period=2, window_size=2, start_policy="expanding")(X))
    np.testing.assert_allclose(exp, [1.0, 10.0, 2.0, 15.0, 4.0, 25.0], rtol=1e-12)
    sd = np.asarray(RollingSeasonalStd(period=2, window_size=2, start_policy="expanding")(X))
    assert np.isnan(sd[:2]).all()
    zero = np.asarray(RollingSeasonalMean(period=2, window_size=2, start_policy="zero")(X))
    np.testing.assert_allclose(zero[:3], [0.5, 5.0, 2.0], rtol=1e-12)


def test_nan_advances_the_slot_clock():
    x = np.array([1.0, 10.0, np.nan, 20.0, 5.0, 30.0])
    out = np.asarray(RollingSeasonalMean(period=2, window_size=2, start_policy="expanding")(x))
    assert math.isnan(out[2])
    np.testing.assert_allclose(out[[0, 1, 3, 4, 5]], [1.0, 10.0, 15.0, 3.0, 25.0], rtol=1e-12)


def test_period_one_is_the_rolling_zscore():
    x = np.random.default_rng(0).standard_normal(200)
    np.testing.assert_allclose(np.asarray(RollingSeasonalZscore(period=1, window_size=20)(x)),
                               np.asarray(RollingZscore(20)(x)), rtol=1e-9, equal_nan=True)


@pytest.mark.parametrize("cls,ref", [(RollingSeasonalMean, RollingSeasonalMean_numpy),
                                     (RollingSeasonalStd, RollingSeasonalStd_numpy),
                                     (RollingSeasonalZscore, RollingSeasonalZscore_numpy)])
@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_matches_baseline(cls, ref, start_policy):
    rng = np.random.default_rng(3)
    t = np.arange(24 * 40)
    x = 1000 + 5 * np.sin(2 * np.pi * t / 24) + rng.standard_normal(len(t))
    x[[7, 300, 301, 650]] = np.nan
    got = np.asarray(cls(period=24, window_size=10, start_policy=start_policy)(x))
    want = ref(24, 10, start_policy)(x)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("cls", [RollingSeasonalMean, RollingSeasonalStd, RollingSeasonalZscore])
def test_regimes_agree(cls):
    x = np.random.default_rng(4).standard_normal(300)
    x[[3, 150]] = np.nan
    assert_regimes_agree(lambda: cls(period=7, window_size=5), x)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        RollingSeasonalMean(period=0)
    with pytest.raises(ValueError):
        RollingSeasonalStd(window_size=1)
    RollingSeasonalMean(window_size=1)            # one value has a mean
