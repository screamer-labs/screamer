"""The robust (Huber) update of KalmanFilter and KalmanRegression.

An innovation beyond `huber` standard deviations updates the state with the
observation variance R / w, w = huber / |z|; the default huber = inf is the
ordinary filter, bit for bit.
"""
import math

import numpy as np
import pytest

import screamer as s

rng = np.random.default_rng(2026)


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def _kf_series(n=3000):
    z = np.cumsum(rng.standard_t(2, n))
    z[::97] = np.nan
    z[n // 2] += 1e4
    return z


def _pair(n=2000):
    x = 4.0 + np.cumsum(rng.standard_normal(n)) * 0.01
    y = 0.3 + 1.1 * x + rng.standard_t(3, n) * 0.004
    y[::131] = np.nan
    y[n // 2] += 0.5
    return y, x


def _kalman_regression_numpy(y, x, delta, obs_var, prior_var=1.0, huber=math.inf):
    """The robust filter written out with numpy matrices."""
    theta = np.zeros(2)
    P = np.eye(2) * prior_var
    Vw = delta / (1 - delta) * np.eye(2)
    out = np.full((len(y), 4), np.nan)
    n = 0
    for t in range(len(y)):
        if np.isnan(y[t]) or np.isnan(x[t]):
            continue
        h = np.array([1.0, x[t]])
        R = P + Vw
        q = h @ R @ h + obs_var
        e = y[t] - h @ theta
        az = abs(e) / math.sqrt(q)
        v = obs_var * az / huber if az > huber else obs_var
        qv = h @ R @ h + v
        k = R @ h / qv
        theta = theta + k * e
        A = np.eye(2) - np.outer(k, h)
        P = A @ R @ A.T + np.outer(k, k) * v
        n += 1
        out[t, :2] = theta[1], theta[0]
        if n >= 2:
            out[t, 2:] = e, math.sqrt(q)
    return out


# ---- huber = inf is the ordinary filter -------------------------------------

@pytest.mark.parametrize("args", [(1e-3, 1.0), (0.1, 2.0, 5.0, 3.0), (1e-5, 0.5, 0.0, math.inf)])
def test_kalman_filter_inf_is_the_ordinary_filter(args):
    z = _kf_series()
    plain = s.KalmanFilter(*args)(z)
    assert np.array_equal(_bits(s.KalmanFilter(*args, huber=math.inf)(z)), _bits(plain))
    # A threshold no innovation reaches changes nothing either.
    assert np.array_equal(_bits(s.KalmanFilter(*args, huber=1e300)(z)), _bits(plain))


@pytest.mark.parametrize("args", [(), (1e-4, 0.01), (1e-8, 1.0, 0.0)])
def test_kalman_regression_inf_is_the_ordinary_filter(args):
    y, x = _pair()
    plain = s.KalmanRegression(*args)(y, x)
    assert np.array_equal(_bits(s.KalmanRegression(*args, huber=math.inf)(y, x)), _bits(plain))
    assert np.array_equal(_bits(s.KalmanRegression(*args, huber=1e300)(y, x)), _bits(plain))


# ---- the weight on one outlier, by hand --------------------------------------

def test_kalman_filter_outlier_step_by_hand():
    # P_pred = 0.5 + 0.5 = 1, Q = P_pred + R = 2, z = 10 -> |z| / sqrt(Q) = 7.07.
    q, r, x0, p0, h = 0.5, 1.0, 0.0, 0.5, 2.0
    obs = 10.0
    p_pred = p0 + q
    az = abs(obs - x0) / math.sqrt(p_pred + r)
    r_eff = r / (h / az)
    k = p_pred / (p_pred + r_eff)
    x1 = x0 + k * (obs - x0)
    p1 = (1 - k) * p_pred

    kf = s.KalmanFilter(q, r, x0, p0, huber=h)
    assert kf(obs) == pytest.approx(x1, rel=1e-15)
    plain = s.KalmanFilter(q, r, x0, p0)(obs)
    assert plain == pytest.approx(5.0, rel=1e-15)
    assert abs(x1) < abs(plain)
    # The next step uses the robust posterior variance p1.
    z2 = 2.4
    p_pred2 = p1 + q
    x2 = x1 + p_pred2 / (p_pred2 + r) * (z2 - x1)
    assert abs(z2 - x1) / math.sqrt(p_pred2 + r) <= h
    assert kf(z2) == pytest.approx(x2, rel=1e-14)


def test_kalman_filter_outlier_influence_is_bounded():
    # As the outlier grows the step tends to P_pred * huber * sqrt(Q) / R.
    q, r, h = 0.5, 1.0, 2.0
    steps = [s.KalmanFilter(q, r, 0.0, 0.5, huber=h)(z) for z in (1e2, 1e4, 1e8)]
    limit = 1.0 * h * math.sqrt(2.0) / r
    assert steps[0] < steps[1] < steps[2] < limit
    assert steps[2] == pytest.approx(limit, rel=1e-7)


def test_kalman_filter_matches_the_numpy_filter():
    from devtools.baselines.KalmanFilter import KalmanFilter_numpy
    z = np.cumsum(rng.standard_t(2, 3000))
    for h in (1.5, 3.0):
        ours = s.KalmanFilter(0.1, 1.0, 0.0, 1.0, huber=h)(z)
        ref = KalmanFilter_numpy(0.1, 1.0, 0.0, 1.0, huber=h)(z)
        np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12)


def test_kalman_regression_matches_the_numpy_filter():
    y, x = _pair()
    for h in (2.0, 3.0):
        ours = s.KalmanRegression(1e-6, 1.6e-5, huber=h)(y, x)
        ref = _kalman_regression_numpy(y, x, 1e-6, 1.6e-5, huber=h)
        np.testing.assert_allclose(ours, ref, rtol=1e-7, atol=1e-10)


def test_kalman_regression_outlier_step():
    # A clean pair with one bad print: the robust filter agrees with the plain
    # one up to it, reports the same spread and spread_std at it, and moves the
    # hedge ratio less there.
    n = 400
    x = 4.0 + np.cumsum(rng.standard_normal(n)) * 0.01
    y = 0.3 + 1.1 * x + rng.standard_normal(n) * 1e-3
    y[300] += 5.0
    plain = s.KalmanRegression(1e-6, 1e-6)(y, x)
    robust = s.KalmanRegression(1e-6, 1e-6, huber=10.0)(y, x)
    z = plain[:, 2] / plain[:, 3]
    assert np.nanmax(np.abs(z[:300])) < 10.0 < abs(z[300])
    assert np.array_equal(_bits(robust[:300]), _bits(plain[:300]))
    assert np.array_equal(_bits(robust[300, 2:]), _bits(plain[300, 2:]))
    before = plain[299, 0]
    assert abs(robust[300, 0] - before) < abs(plain[300, 0] - before)


def test_small_innovations_are_unaffected():
    z = np.cumsum(rng.standard_normal(2000))
    kf = s.KalmanFilter(1.0, 1.0)
    out = kf(z)
    # Every standardized innovation is under 10, so huber = 10 never triggers.
    assert np.array_equal(_bits(s.KalmanFilter(1.0, 1.0, huber=10.0)(z)), _bits(out))
    y, x = _pair()
    y = 0.3 + 1.1 * x + rng.standard_normal(len(x)) * 0.004
    plain = s.KalmanRegression(1e-6, 1.6e-5)(y, x)
    zr = plain[:, 2] / plain[:, 3]
    assert np.nanmax(np.abs(zr)) < 50.0
    robust = s.KalmanRegression(1e-6, 1.6e-5, huber=50.0)(y, x)
    assert np.array_equal(_bits(robust), _bits(plain))


# ---- batch == streaming -------------------------------------------------------

def test_batch_equals_streaming():
    z = _kf_series(500)
    kf = s.KalmanFilter(0.1, 1.0, huber=2.5)
    batch = kf(z)
    kf.reset()
    stream = np.array([kf(v) for v in z])
    assert np.array_equal(_bits(stream), _bits(batch))

    y, x = _pair(500)
    kr = s.KalmanRegression(1e-6, 1.6e-5, huber=2.0)
    batch = kr(y, x)
    kr.reset()
    stream = np.array([kr(a, b) for a, b in zip(y, x)])
    assert np.array_equal(_bits(stream), _bits(batch))


# ---- validation ---------------------------------------------------------------

@pytest.mark.parametrize("op", [lambda h: s.KalmanFilter(huber=h), lambda h: s.KalmanRegression(huber=h)])
def test_huber_must_be_positive(op):
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="huber must be > 0"):
            op(bad)
    for bad in (math.nan, -math.inf):
        with pytest.raises(ValueError, match="huber must be finite or inf"):
            op(bad)
    op(math.inf)
    op(1e-300)
