"""The robust (Huber) update of KalmanFilter and KalmanRegression.

An innovation beyond `huber` standard deviations updates the state with the
observation variance R / w, w = huber / |z|, on every step that has a
prediction; the default huber = inf is the ordinary filter, bit for bit.
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
        v = obs_var * az / huber if n >= 1 and az > huber else obs_var
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
    # The first observation equals the state (no innovation); the second is an
    # outlier: P_pred = P_1 + q, Q = P_pred + R, |z| = 10 / sqrt(Q) > huber.
    q, r, x0, p0, h = 0.1, 1.0, 0.0, 0.1, 2.0
    p_pred = p0 + q
    p1 = (1 - p_pred / (p_pred + r)) * p_pred
    p_pred = p1 + q
    az = 10.0 / math.sqrt(p_pred + r)
    k = p_pred / (p_pred + r / (h / az))
    x2 = k * 10.0
    p2 = (1 - k) * p_pred

    kf = s.KalmanFilter(q, r, x0, p0, huber=h)
    assert kf(0.0) == 0.0
    assert kf(10.0) == pytest.approx(x2, rel=1e-14)
    plain = s.KalmanFilter(q, r, x0, p0)
    plain(0.0)
    assert x2 < plain(10.0)
    # The next step uses the robust posterior variance p2.
    z3 = x2 + 0.5
    p_pred3 = p2 + q
    x3 = x2 + p_pred3 / (p_pred3 + r) * (z3 - x2)
    assert kf(z3) == pytest.approx(x3, rel=1e-14)


def test_kalman_filter_first_observation_is_not_weighted():
    # The first observation has only the prior to compare with.
    for var in (1.0, math.inf):
        assert s.KalmanFilter(0.1, 1.0, 0.0, var, huber=2.0)(50.0) == s.KalmanFilter(0.1, 1.0, 0.0, var)(50.0)


def test_kalman_filter_outlier_influence_is_bounded():
    # As the outlier grows the step tends to P_pred * huber * sqrt(Q) / R.
    q, r, h = 0.1, 1.0, 2.0
    p_pred = 0.2
    p_pred = (1 - p_pred / (p_pred + r)) * p_pred + q
    steps = []
    for z in (1e2, 1e4, 1e8):
        kf = s.KalmanFilter(q, r, 0.0, 0.1, huber=h)
        kf(0.0)
        steps.append(kf(z))
    limit = p_pred * h * math.sqrt(p_pred + r) / r
    assert steps[0] < steps[1] < steps[2] < limit
    assert steps[2] == pytest.approx(limit, rel=1e-7)


def test_kalman_filter_infinite_observation_is_skipped():
    # After the first observation an infinite one has weight 0: the state stays
    # and its variance grows to P_pred.
    kf = s.KalmanFilter(0.1, 1.0, 2.0, 0.1, huber=2.0)
    assert kf(2.0) == 2.0
    assert kf(math.inf) == 2.0
    assert kf(-math.inf) == 2.0
    ref = s.KalmanFilter(0.1, 1.0, 2.0, 0.1)
    ref(2.0)
    p1 = (1 - 0.2 / 1.2) * 0.2
    assert kf(2.5) == s.KalmanFilter(0.1, 1.0, 2.0, p1 + 0.2)(2.5)
    y, x = _pair(300)
    y = np.nan_to_num(y, nan=1.0)
    out = s.KalmanRegression(1e-6, 1e-4, huber=2.0)(np.append(y, math.inf), np.append(x, x[-1]))
    assert out[-1, 0] == out[-2, 0] and out[-1, 1] == out[-2, 1]
    assert out[-1, 2] == math.inf and np.isfinite(out[-1, 3])


# ---- warm-up: a robust filter needs a prior that agrees with the data --------

def test_kalman_filter_warm_up():
    g = np.random.default_rng(3)
    z = 100.0 + g.standard_normal(300)
    # A diffuse prior: the first observation sets the state, and the robust
    # filter tracks the ordinary one from the start.
    plain = s.KalmanFilter(0.01, 1.0, 0.0, math.inf)(z)
    robust = s.KalmanFilter(0.01, 1.0, 0.0, math.inf, huber=2.0)(z)
    assert robust[0] == plain[0] == z[0]
    np.testing.assert_allclose(robust, plain, atol=1.0)
    np.testing.assert_allclose(robust[20:], plain[20:], atol=0.5)
    # The default prior (state 0, variance 1) contradicts data near 100: the
    # robust filter takes those observations for outliers and lags.
    plain = s.KalmanFilter(0.01, 1.0)(z)
    robust = s.KalmanFilter(0.01, 1.0, huber=2.0)(z)
    assert robust[0] == plain[0]
    assert robust[9] < plain[9] - 20.0
    assert abs(robust[-1] - plain[-1]) < 0.5


def test_kalman_filter_outlier_after_convergence_has_the_expected_weight():
    # A constant level gives zero innovations, so P follows the plain recursion.
    q, r, h, n = 0.01, 1.0, 3.0, 200
    z = np.full(n + 1, 100.0)
    z[n] += 50.0
    out = s.KalmanFilter(q, r, 0.0, math.inf, huber=h)(z)
    assert (out[:n] == 100.0).all()
    p = r
    for _ in range(n - 1):
        p_pred = p + q
        p = (1 - p_pred / (p_pred + r)) * p_pred
    p_pred = p + q
    w = h / (50.0 / math.sqrt(p_pred + r))
    expected = 100.0 + p_pred / (p_pred + r / w) * 50.0
    assert out[n] == pytest.approx(expected, rel=1e-14)
    plain = s.KalmanFilter(q, r, 0.0, math.inf)(z)[n]
    assert out[n] - 100.0 < (plain - 100.0) / 4


def test_kalman_regression_down_weights_an_outlier_on_log_prices():
    g = np.random.default_rng(0)
    n = 1000
    x = 4.0 + np.cumsum(g.standard_normal(n)) * 0.01
    y = 0.5 + 1.1 * x + g.standard_normal(n) * 0.004
    for prior_var in (1.0, 100.0):
        plain = s.KalmanRegression(1e-6, 1.6e-5, prior_var)(y, x)
        robust = s.KalmanRegression(1e-6, 1.6e-5, prior_var, huber=2.0)(y, x)
        np.testing.assert_allclose(robust[300:, 0], plain[300:, 0], atol=0.005)
        y2 = y.copy()
        y2[800] += 0.1
        p2 = s.KalmanRegression(1e-6, 1.6e-5, prior_var)(y2, x)
        r2 = s.KalmanRegression(1e-6, 1.6e-5, prior_var, huber=2.0)(y2, x)
        assert abs(r2[800, 0] - r2[799, 0]) < abs(p2[800, 0] - p2[799, 0]) / 3
        np.testing.assert_allclose(r2[800], _kalman_regression_numpy(y2, x, 1e-6, 1.6e-5, prior_var, 2.0)[800],
                                   rtol=1e-4)


# ---- huber = inf gives the outputs recorded before the change ----------------

def test_huber_inf_gives_the_recorded_outputs():
    # Recorded from the build before huber was added (commit 222b052).
    g = np.random.default_rng(7)
    n = 5000
    z = np.cumsum(g.standard_t(2, n))
    z[::97] = np.nan
    z[1234] = 1e6
    x = 100 + np.cumsum(g.normal(0, 1, n))
    y = 3 + 1.5 * x + g.standard_t(2, n)
    y[::131] = np.nan
    y[2000] += 500
    idx = [2, 10, 1234, 1235, 2000, 4999]
    for args, expected in RECORDED_KF.items():
        got = s.KalmanFilter(*args, huber=math.inf)(z)[idx].tolist()
        assert got == expected, args
    for args, expected in RECORDED_KR.items():
        got = s.KalmanRegression(*args, huber=math.inf)(y, x)[idx].tolist()
        assert got == expected, args


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
    plain = s.KalmanRegression(1e-8, 1e-6)(y, x)
    robust = s.KalmanRegression(1e-8, 1e-6, huber=10.0)(y, x)
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


RECORDED_KF = {
    (0.001, 1.0): [-0.27501707046132484, -1.22452648787555, 31050.38179702113, 30081.333829611835, -13.548642538752858, -11.636186791296938],
    (0.1, 2.0, 5.0, 3.0): [0.8643644064724839, -1.7413097894244953, 199935.18301037038, 159931.75444121013, -12.849828985417235, -1.159421903516333],
    (1e-05, 0.5, 0.0, math.inf): [-0.4121289023439973, -1.325543503295453, 4378.540215004757, 4358.636099110445, 119.001465861052, -70.98635400415932],
    (0.0, 1.0): [-0.27475177765309616, -1.204739670150702, 745.0226954097252, 744.3470560871431, 444.83924287918546, 161.2606558495498],
}
RECORDED_KR = {
    (): [
        [1.5191116651769418, -0.778423873943302, -1.5619561542156362, 0.10947742155907712],
        [1.5542479910768248, -2.914121136247698, 1.3389243662752222, 0.10949278866825249],
        [1.4758909122701696, 5.186905822202761, 0.22709705471390862, 0.11991894368312951],
        [1.5214158796018546, 5.205389242450937, 5.471482517219954, 0.1197472892945884],
        [6.4350206647005965, 46.066864070785925, 498.4121173634161, 0.09144551954075471],
        [1.2299679346897987, 25.582783122713106, 0.7576569510558784, 0.09653125828986518],
    ],
    (0.0001, 0.01): [
        [1.5101028124476634, 0.005677902059246339, -1.561820264422039, 1.007993792944043],
        [1.5263959953635642, -0.024256636884695596, 1.4501210850104087, 1.0124989163019924],
        [1.5171300253318982, 0.6028364662904921, 0.1886085918476681, 1.124126078148885],
        [1.5657290258088885, 0.6072166066140183, 5.461561553260822, 1.1229860165732828],
        [7.485917267963251, 13.527868888203308, 498.22381017144824, 0.8182615828910959],
        [1.3662851284158999, 13.932794956973902, 0.7733751216773896, 0.8717542442725263],
    ],
    (1e-08, 1.0, 0.0): [
        [0.0004544657813493678, 4.544537229006468e-06, 150.67808790057566, 1.0000995936411692],
        [0.008461455933122197, 8.407027165417316e-05, 152.3242523077413, 1.0005010365765976],
        [1.5267388079192104, 0.015952796409912866, -0.48869160504270326, 1.0058211450390213],
        [1.5272531105117637, 0.015955015628739043, 4.978037764916365, 1.0058049292795685],
        [1.5925923436752378, 0.016488302293264572, 500.23624164725305, 1.0043188692420995],
        [1.5321185306703238, 0.017048617376045756, -0.33597608915246724, 1.004271898315714],
    ],
    (0.3, 5.0, 10.0): [
        [1.5098734450877773, 0.015013359270195068, -1.5547316051700477, 65.41151309427687],
        [1.5261277769486996, 0.015082799681859365, 1.4650659213041592, 65.71187195187888],
        [1.5215611461894758, 0.11006414855741412, 0.18429445431161184, 73.06916545382404],
        [1.5705260284820894, 0.11102024770858236, 5.46071204417575, 72.99904526029066],
        [7.698050149754936, 2.9800977035131595, 498.2164175291797, 52.863084145443906],
        [1.4784275382242584, 4.299500465842948, 0.6356148872412462, 56.38871921753992],
    ],
}
