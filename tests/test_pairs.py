"""Pairs-trading operators: KalmanRegression, RollingTLSBeta, RollingADF, BandSignal."""
import numpy as np
import pytest

import screamer as s

rng = np.random.default_rng(31)


def _pair(n=1500, beta=1.1, phi=0.95):
    b = 4.0 + np.cumsum(rng.standard_normal(n)) * 0.01
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.standard_normal() * 0.004
    return 0.3 + beta * b + e, b


def _kalman_numpy(y, x, delta, obs_var, prior_var=1.0):
    theta = np.zeros(2)
    P = np.eye(2) * prior_var
    Vw = delta / (1 - delta) * np.eye(2)
    out = np.full((len(y), 4), np.nan)
    for t in range(len(y)):
        h = np.array([1.0, x[t]])
        R = P + Vw
        q = h @ R @ h + obs_var
        e = y[t] - h @ theta
        k = R @ h / q
        theta = theta + k * e
        A = np.eye(2) - np.outer(k, h)
        P = A @ R @ A.T + np.outer(k, k) * obs_var
        out[t, :2] = theta[1], theta[0]
        if t >= 1:
            out[t, 2:] = e, np.sqrt(q)
    return out


def test_kalman_matches_numpy_filter():
    y, x = _pair(400)
    ours = s.KalmanRegression(delta=1e-4, obs_var=1e-4)(y, x)
    np.testing.assert_allclose(ours, _kalman_numpy(y, x, 1e-4, 1e-4), rtol=1e-8, atol=1e-10)
    zero = s.KalmanRegression(1e-6, 1e-4, prior_var=0.0)(y, x)
    np.testing.assert_allclose(zero, _kalman_numpy(y, x, 1e-6, 1e-4, 0.0), rtol=1e-8, atol=1e-10)


def test_kalman_tracks_the_hedge_ratio_and_skips_nan():
    y, x = _pair(3000, beta=1.25)
    out = s.KalmanRegression(delta=1e-8, obs_var=1.6e-4)(y, x)
    assert abs(np.median(out[1000:, 0]) - 1.25) < 0.05
    z = out[:, 2] / out[:, 3]
    assert 0.5 < np.nanstd(z[500:]) < 1.5
    y2 = y.copy()
    y2[500] = np.nan
    gap = s.KalmanRegression(1e-8, 1.6e-4)(y2, x)
    assert np.isnan(gap[500]).all()
    np.testing.assert_allclose(gap[501:], s.KalmanRegression(1e-8, 1.6e-4)(np.delete(y, 500), np.delete(x, 500))[500:])


def test_tls_is_symmetric_and_between_the_ols_slopes():
    y, x = _pair(800)
    tls = s.RollingTLSBeta(200)(y, x)
    inv = s.RollingTLSBeta(200)(x, y)
    ok = np.isfinite(tls)
    assert ok.sum() == 601
    np.testing.assert_allclose(tls[ok] * inv[ok], 1.0, rtol=1e-10)
    ols_yx = s.RollingBeta(200)(y, x)
    ols_xy = s.RollingBeta(200)(x, y)
    lo, hi = np.minimum(ols_yx, 1 / ols_xy), np.maximum(ols_yx, 1 / ols_xy)
    assert np.all((tls[ok] >= lo[ok] - 1e-12) & (tls[ok] <= hi[ok] + 1e-12))


def test_adf_separates_a_mean_reverting_series_from_a_random_walk():
    n = 3000
    walk = np.cumsum(rng.standard_normal(n))
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.8 * ar[t - 1] + rng.standard_normal()
    a = s.RollingADF(250, 1)(ar)
    w = s.RollingADF(250, 1)(walk)
    assert np.nanmedian(a) < -3.5
    assert np.nanmedian(w) > -2.86
    assert np.isnan(a[:251]).all() and np.isfinite(a[251])


def test_adf_lags_and_argument_checks():
    y = np.cumsum(rng.standard_normal(300))
    for lags in (0, 3):
        out = s.RollingADF(60, lags)(y)
        assert np.flatnonzero(np.isfinite(out))[0] == 60 + lags
    with pytest.raises(ValueError):
        s.RollingADF(4, 1)
    with pytest.raises(ValueError):
        s.RollingADF(100, -1)


def test_band_signal_known_path():
    z = [0, 1, 2.1, 1, 0.5, -0.1, -2.5, -1, 0.2, 3.5, 4.5, 3, 1, 0.1, 2.2, np.nan, 1, -0.5, 2.5]
    out = s.BandSignal(2.0, 0.0, 4.0)(np.array(z))
    want = [0, 0, -1, -1, -1, 0, 1, 1, 0, -1, 0, 0, 0, 0, 0, np.nan, 0, 0, -1]
    np.testing.assert_array_equal(out, want)
    # A jump from one side to the other flips the position in one step.
    np.testing.assert_array_equal(s.BandSignal(2.0, 0.5)(np.array([2.5, -2.5])), [-1, 1])
    # A negative exit holds a short until the spread overshoots.
    np.testing.assert_array_equal(s.BandSignal(2.0, -0.5)(np.array([2.5, 0.0, -0.4, -0.6])),
                                  [-1, -1, -1, 0])


def test_band_signal_argument_checks():
    for kw in ({"entry": 0.0}, {"exit": 2.0}, {"exit": -2.0}, {"stop": 1.5}):
        with pytest.raises(ValueError):
            s.BandSignal(**kw)
