import math

import numpy as np
import pytest

from devtools.baselines.RollingVarianceRatio import RollingVarianceRatio_numpy
from screamer import RollingVarianceRatio
from tests.regime_helpers import assert_regimes_agree

ALT = np.array([1.0, -1.0, 1.0, -1.0, 1.0])


def test_hand_example():
    # mu = 0.2, sum (r - mu)^2 = 4.8, the four 2-sums are 0: ratio = (0.64 / 4.8) / 1.2 = 1/9
    ratio = np.asarray(RollingVarianceRatio(5, lag=2)(ALT))
    assert np.isnan(ratio[:4]).all()
    np.testing.assert_allclose(ratio[4], 1 / 9, rtol=1e-12)
    # delta_1 = 5 * 4 * 0.9216 / 4.8^2 = 0.8 = theta: z = sqrt(5) (1/9 - 1) / sqrt(0.8) = -20/9
    z = np.asarray(RollingVarianceRatio(5, lag=2, output="zscore")(ALT))
    np.testing.assert_allclose(z[4], -20 / 9, rtol=1e-12)


def test_random_walk_ratio_is_near_one():
    r = np.random.default_rng(0).standard_normal(20_000)
    vr = np.asarray(RollingVarianceRatio(2000, lag=4)(r))
    assert abs(np.nanmean(vr) - 1.0) < 0.02


def test_trending_and_mean_reverting_returns():
    rng = np.random.default_rng(1)
    e = rng.standard_normal(5000)
    ar = np.zeros(5000)
    for t in range(1, 5000):
        ar[t] = 0.3 * ar[t - 1] + e[t]
    assert np.nanmean(np.asarray(RollingVarianceRatio(500, lag=5)(ar))) > 1.2
    assert np.nanmean(np.asarray(RollingVarianceRatio(500, lag=5)(np.diff(e)))) < 0.8


def test_constant_window_is_nan():
    # 0.5 is exact in binary, so the window mean is exactly 0.5 and the dispersion exactly 0
    out = np.asarray(RollingVarianceRatio(10, lag=2)(np.full(30, 0.5)))
    assert np.isnan(out).all()
    out = np.asarray(RollingVarianceRatio(10, lag=2, output="zscore")(np.full(30, 0.5)))
    assert np.isnan(out).all()


@pytest.mark.parametrize("output", ["ratio", "zscore"])
@pytest.mark.parametrize("lag", [2, 3, 7])
@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_matches_baseline(output, lag, start_policy):
    r = np.random.default_rng(lag).standard_normal(400)
    r[[3, 120, 121]] = np.nan
    got = np.asarray(RollingVarianceRatio(60, lag=lag, output=output, start_policy=start_policy)(r))
    want = RollingVarianceRatio_numpy(60, lag, output, start_policy)(r)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_expanding_starts_after_two_lags():
    r = np.random.default_rng(2).standard_normal(20)
    out = np.asarray(RollingVarianceRatio(10, lag=3, start_policy="expanding")(r))
    assert np.isnan(out[:6]).all() and np.isfinite(out[6:]).all()


@pytest.mark.parametrize("output", ["ratio", "zscore"])
def test_regimes_agree(output):
    r = np.random.default_rng(3).standard_normal(300)
    r[77] = np.nan
    assert_regimes_agree(lambda: RollingVarianceRatio(50, lag=4, output=output), r)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        RollingVarianceRatio(10, lag=1)
    with pytest.raises(ValueError):
        RollingVarianceRatio(8, lag=4)                        # needs window_size > 2 lag
    with pytest.raises(ValueError, match="ratio.*zscore"):
        RollingVarianceRatio(10, lag=2, output="z")
