import numpy as np
import pytest

from devtools.baselines.RollingHMM import RollingHMM_numpy
from screamer import RollingHMM
from tests.regime_helpers import assert_regimes_agree

A_TRUE = [[0.99, 0.01], [0.02, 0.98]]


def _chain(T, rng):
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        stay = A_TRUE[s[t - 1]][s[t - 1]]
        s[t] = s[t - 1] if rng.random() < stay else 1 - s[t - 1]
    return s


def _gaussian_series(seed=1, T=4000):
    rng = np.random.default_rng(seed)
    s = _chain(T, rng)
    return rng.standard_normal(T) * np.where(s == 1, 2.0, 0.5), s


def _ar1_series(seed=2, T=4000):
    rng = np.random.default_rng(seed)
    s = _chain(T, rng)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = (0.6 if s[t] == 1 else -0.4) * x[t - 1] + rng.standard_normal()
    return x, s


def test_gaussian_recovers_the_states_and_variances():
    x, s = _gaussian_series()
    out = np.asarray(RollingHMM(window_size=2000, refit_every=500)(x))
    assert np.isnan(out[:1999]).all() and np.isfinite(out[1999:]).all()
    o, st = out[2000:], s[2000:]
    assert np.mean((o[:, 0] > 0.5) == (st == 1)) > 0.9
    assert 1.7 < np.median(o[o[:, 0] > 0.9, 2]) < 2.3          # high-variance state: sigma 2
    assert 0.4 < np.median(o[o[:, 0] < 0.1, 2]) < 0.7          # low-variance state: sigma 0.5
    assert np.all(np.abs(o[:, 1]) < 0.2)                       # both means near 0


def test_ar1_identifies_the_persistent_state():
    x, s = _ar1_series()
    out = np.asarray(RollingHMM(window_size=2000, refit_every=500, emission="ar1")(x))
    o, st, xt = out[2000:], s[2000:], x[2000:]
    assert np.mean((o[:, 0] > 0.5) == (st == 1)) > 0.8
    hi, lo = o[:, 0] > 0.9, o[:, 0] < 0.1
    assert abs(np.polyfit(xt[hi], o[hi, 1], 1)[0] - 0.6) < 0.1     # state 1: phi 0.6
    assert abs(np.polyfit(xt[lo], o[lo, 1], 1)[0] + 0.4) < 0.1     # state 0: phi -0.4


@pytest.mark.parametrize("emission", ["gaussian", "ar1"])
def test_matches_baseline(emission):
    x, _ = _ar1_series(seed=5, T=700) if emission == "ar1" else _gaussian_series(seed=5, T=700)
    x[[50, 400]] = np.nan
    args = dict(window_size=200, refit_every=60, emission=emission, max_iter=40, tol=1e-8)
    got = np.asarray(RollingHMM(**args)(x))
    want = RollingHMM_numpy(**args)(x)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8, equal_nan=True)


@pytest.mark.parametrize("emission", ["gaussian", "ar1"])
def test_outputs_never_use_future_values(emission):
    x, _ = _gaussian_series(seed=7, T=900)
    t0 = 600
    y = x.copy()
    y[t0:] = 10.0 * x[t0:] + 3.0
    op = lambda: RollingHMM(window_size=300, refit_every=50, emission=emission)
    np.testing.assert_array_equal(np.asarray(op()(x))[:t0], np.asarray(op()(y))[:t0])


def test_nan_is_skipped():
    x, _ = _gaussian_series(seed=8, T=500)
    holed = np.insert(x, 300, np.nan)
    a = np.asarray(RollingHMM(window_size=100, refit_every=30)(x))
    b = np.asarray(RollingHMM(window_size=100, refit_every=30)(holed))
    assert np.isnan(b[300]).all()
    np.testing.assert_array_equal(np.delete(b, 300, axis=0), a)


def test_constant_window_stays_finite():
    x = np.r_[np.full(150, 0.25), np.random.default_rng(9).standard_normal(100)]
    for emission in ("gaussian", "ar1"):
        out = np.asarray(RollingHMM(window_size=100, refit_every=20, emission=emission)(x))
        assert np.isfinite(out[99:]).all()


@pytest.mark.parametrize("emission", ["gaussian", "ar1"])
def test_regimes_agree(emission):
    x, _ = _gaussian_series(seed=10, T=260)
    x[[20, 150]] = np.nan
    assert_regimes_agree(lambda: RollingHMM(window_size=50, refit_every=7, emission=emission), x)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        RollingHMM(window_size=9)
    with pytest.raises(ValueError):
        RollingHMM(refit_every=0)
    with pytest.raises(ValueError):
        RollingHMM(max_iter=0)
    with pytest.raises(ValueError):
        RollingHMM(tol=-1.0)
    with pytest.raises(ValueError, match="gaussian.*ar1"):
        RollingHMM(emission="ar2")


@pytest.mark.parametrize("emission", ["gaussian", "ar1"])
def test_zero_and_large_inputs_stay_finite(emission):
    rng = np.random.default_rng(11)
    for x in (np.zeros(150), rng.standard_normal(150) * 1e100, rng.standard_normal(150) * 1e-100):
        out = np.asarray(RollingHMM(window_size=50, refit_every=7, emission=emission)(x))
        assert np.isfinite(out[49:]).all()
