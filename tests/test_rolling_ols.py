"""RollingOLS and ExpandingOLS: least squares of y on k regressors.

Checked against numpy.linalg.lstsq on every window (devtools/baselines/
RollingOLS.py), for k = 1..4, with and without an intercept, rolling and
expanding; pred uses only earlier rows; NaN rows are skipped; a singular window
gives NaN; batch equals per-event; k = 1 agrees with RollingLinearRegression;
and the input forms of a multi-input operator work for every input count.
"""
import numpy as np
import pytest

import screamer as s
from devtools.baselines.RollingOLS import ols_rows


def _data(k, T=300, seed=0, level=0.0):
    rng = np.random.default_rng(seed + 10 * k)
    X = rng.standard_normal((T, k)) + level
    beta = rng.standard_normal(k)
    y = 0.7 + X @ beta + 0.3 * rng.standard_normal(T)
    return y, X


def _close(got, ref, rtol=1e-9):
    got, ref = np.asarray(got), np.asarray(ref)
    assert got.shape == ref.shape
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    m = ~np.isnan(ref)
    scale = np.maximum(np.abs(ref[m]), 1.0)
    assert np.max(np.abs(got[m] - ref[m]) / scale, initial=0.0) <= rtol


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("window", [None, 8, 40])
def test_matches_lstsq(k, intercept, window):
    y, X = _data(k)
    if window is not None and window < k + 2:
        pytest.skip("window below the minimum")
    op = (s.ExpandingOLS(k, intercept=intercept) if window is None
          else s.RollingOLS(window, k, intercept=intercept))
    out = op(y, *X.T)
    assert out.shape == (len(y), k + (4 if intercept else 3))
    _close(out, ols_rows(y, X.T, window, intercept))


def test_warmup():
    y, X = _data(2)
    out = s.RollingOLS(10, 2)(y, *X.T)
    assert np.isnan(out[:9, 2:]).all() and np.isfinite(out[9:, 2:]).all()
    assert np.isnan(out[:10, :2]).all() and np.isfinite(out[10:, :2]).all()
    out = s.ExpandingOLS(2, intercept=False)(y, *X.T)   # p + 1 = 3 rows
    assert np.isnan(out[:2, 2:]).all() and np.isfinite(out[2:, 2:]).all()
    assert np.isnan(out[:3, :2]).all() and np.isfinite(out[3:, :2]).all()


def test_pred_uses_the_previous_fit():
    y, X = _data(3)
    out = s.RollingOLS(25, 3)(y, *X.T)
    b = out[:-1, 3:]
    pred = b[:, 0] + np.sum(b[:, 1:] * X[1:], axis=1)
    np.testing.assert_allclose(out[1:, 0], pred, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out[:, 1], y - out[:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("make", [lambda: s.RollingOLS(20, 2), lambda: s.ExpandingOLS(2)])
def test_perturbing_y_t_leaves_pred_t(make):
    y, X = _data(2)
    base = make()(y, *X.T)
    for t in (30, 150, 299):
        y2 = y.copy()
        y2[t] += 100.0
        out = make()(y2, *X.T)
        assert np.array_equal(out[:t], base[:t], equal_nan=True)
        assert out[t, 0] == base[t, 0]
        assert out[t, 1] == pytest.approx(base[t, 1] + 100.0, rel=1e-12)
        assert not np.array_equal(out[t, 3:], base[t, 3:])


@pytest.mark.parametrize("make", [lambda: s.RollingOLS(15, 2), lambda: s.ExpandingOLS(2)])
def test_nan_rows_are_skipped(make):
    y, X = _data(2)
    bad = np.zeros(len(y), bool)
    bad[[5, 40, 41, 100, 200]] = True
    y2, X2 = y.copy(), X.copy()
    y2[[5, 100]] = np.nan
    X2[[40, 41], 0] = np.nan
    X2[200, 1] = np.nan
    out = make()(y2, *X2.T)
    assert np.isnan(out[bad]).all()
    clean = make()(y[~bad], *X[~bad].T)
    assert np.array_equal(out[~bad], clean, equal_nan=True)


def test_singular_window_gives_nan_and_recovers():
    rng = np.random.default_rng(3)
    T = 120
    x1 = rng.standard_normal(T)
    x2 = rng.standard_normal(T)
    x2[30:60] = 2.0                 # constant: collinear with the intercept
    y = 1.0 + x1 - x2 + 0.1 * rng.standard_normal(T)
    out = s.RollingOLS(10, 2)(y, x1, x2)
    assert np.isnan(out[39:60, 2:]).all()            # windows inside the constant run
    assert np.isnan(out[40:61, :2]).all()            # and the next pred
    assert np.isfinite(out[60:, 2:]).all()
    # A duplicated regressor is singular with or without an intercept.
    for icpt in (True, False):
        out = s.ExpandingOLS(2, intercept=icpt)(y, x1, x1)
        assert np.isnan(out[:, 2:]).all()
    # Without an intercept, a constant regressor is a valid intercept column.
    ones = np.ones(T)
    a = s.RollingOLS(10, 2, intercept=False)(y, ones, x1)
    b = s.RollingOLS(10, 1)(y, x1)
    np.testing.assert_allclose(a[:, 3:], b[:, 3:], rtol=1e-9, atol=1e-12)


def test_uncentered_r2_without_intercept():
    y, X = _data(2)
    out = s.RollingOLS(30, 2, intercept=False)(y, *X.T)
    t = 100
    A, v = X[t - 29:t + 1], y[t - 29:t + 1]
    b = np.linalg.lstsq(A, v, rcond=None)[0]
    e = v - A @ b
    assert out[t, 2] == pytest.approx(1 - e @ e / (v @ v), rel=1e-12)


def test_k1_matches_rolling_linear_regression():
    y, X = _data(1, T=400)
    out = s.RollingOLS(30, 1)(y, X[:, 0])
    ref = s.RollingLinearRegression(30)(y, X[:, 0])   # slope, intercept, r_squared, stderr
    np.testing.assert_allclose(out[:, 4], ref[:, 0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out[:, 3], ref[:, 1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out[:, 2], ref[:, 2], rtol=1e-10, atol=1e-12)


def _centered_slopes(y, X):
    """lstsq slopes of a window taken about its first row (exact differences
    for values within a factor 2), then centered: free of the level."""
    A, v = X - X[0], y - y[0]
    return np.linalg.lstsq(A - A.mean(0), v - v.mean(), rcond=None)[0]


@pytest.mark.parametrize("level", [0.0, 1e3, 1e6, 1e9])
def test_slopes_do_not_depend_on_the_level(level):
    """Columns enter about an origin near their mean, so the slopes keep their
    digits at any level, also right after a spike leaves the window and in an
    expanding fit whose level drifts."""
    y, X = _data(3, T=600, level=level)
    y = y - y.mean() + level
    y[200] += 1e4
    X[300, 1] += 1e4
    out = s.RollingOLS(50, 3)(y, *X.T)
    for t in list(range(49, 200)) + list(range(250, 300)) + list(range(350, 600)):
        ref = _centered_slopes(y[t - 49:t + 1], X[t - 49:t + 1])
        np.testing.assert_allclose(out[t, 4:], ref, rtol=1e-12, atol=1e-13)
    drift = np.cumsum(np.full(600, 0.05))[:, None]
    Xd, yd = X + drift, y + 3.0 * drift[:, 0]
    out = s.ExpandingOLS(3)(yd, *Xd.T)
    for t in (100, 199, 400, 599):
        ref = _centered_slopes(yd[:t + 1], Xd[:t + 1])
        np.testing.assert_allclose(out[t, 4:], ref, rtol=1e-11, atol=1e-12)


def test_long_run_does_not_drift():
    y, X = _data(4, T=20000)
    out = s.RollingOLS(100, 4)(y, *X.T)
    ref = ols_rows(y[18900:], X[18900:].T, 100, True)[100:]   # pred from row 18999 on
    _close(out[19000:], ref, rtol=1e-9)


def test_batch_equals_streaming_and_columns():
    y, X = _data(3)
    y[[50, 120]] = np.nan
    for make in (lambda: s.RollingOLS(20, 3), lambda: s.ExpandingOLS(3, intercept=False)):
        batch = make()(y, *X.T)
        op = make()
        stream = np.array([op(*map(float, row)) for row in np.c_[y, X]])
        assert np.array_equal(batch, stream, equal_nan=True)
        # (T, 2) inputs: two columns, each the 1-D result; a packed (T, 4) array.
        Y2 = np.c_[y, y[::-1]]
        X2 = [np.c_[x, x[::-1]] for x in X.T]
        out2 = make()(Y2, *X2)
        assert out2.shape == (len(y), 2, batch.shape[1])
        assert np.array_equal(out2[:, 0], batch, equal_nan=True)
        assert np.array_equal(out2[:, 1], make()(y[::-1], *X[::-1].T), equal_nan=True)
        assert np.array_equal(make()(np.c_[y, X]), batch, equal_nan=True)


def test_input_forms():
    y, X = _data(2, T=40)
    rows = [tuple(map(float, r)) for r in np.c_[y, X]]
    batch = s.RollingOLS(10, 2)(y, *X.T)
    lst = s.RollingOLS(10, 2)(rows)
    assert isinstance(lst, list) and isinstance(lst[0], tuple) and len(lst[0]) == 6
    assert np.array_equal(np.array(lst), batch, equal_nan=True)
    it = list(s.RollingOLS(10, 2)(iter(y), iter(X[:, 0]), iter(X[:, 1])))
    assert np.array_equal(np.array(it, dtype=float), batch, equal_nan=True)
    with pytest.raises(ValueError, match="^RollingOLS expects 3 inputs, got 2$"):
        s.RollingOLS(10, 2)(y, X[:, 0])
    op = s.RollingOLS(10, 2)
    op(y, *X.T)
    op.reset()
    assert np.array_equal(np.array([op(*r) for r in rows]), batch, equal_nan=True)


def test_in_a_pipeline():
    y, X = _data(2, T=60)
    a, b, c = s.Input("y"), s.Input("x1"), s.Input("x2")
    p = s.Pipeline(inputs=[a, b, c], outputs=[s.RollingOLS(10, 2)(s.CombineLatest()(a, b, c))])
    values, index = p(y, X[:, 0], X[:, 1])
    assert np.array_equal(values, s.RollingOLS(10, 2)(y, *X.T), equal_nan=True)


def test_sixteen_regressors():
    y, X = _data(16, T=200)
    out = s.RollingOLS(40, 16)(y, *X.T)
    assert out.shape == (200, 20)
    _close(out, ols_rows(y, X.T, 40, True))


def test_settings():
    with pytest.raises(ValueError, match="n_regressors must be between 1 and 16"):
        s.RollingOLS(40, 17)
    with pytest.raises(ValueError, match="^RollingOLS: n_regressors must be at least 1, got 0$"):
        s.RollingOLS(40, 0)
    with pytest.raises(ValueError, match="window_size must be at least n_regressors"):
        s.RollingOLS(3, 2)
    s.RollingOLS(3, 2, intercept=False)
    s.RollingOLS(4, 2)
    with pytest.raises(TypeError, match="^RollingOLS: intercept must be a bool, got int$"):
        s.RollingOLS(10, 2, 1)
    with pytest.raises(TypeError, match="^ExpandingOLS: intercept must be a bool, got str$"):
        s.ExpandingOLS(2, "yes")
    assert s.ExpandingOLS(2, np.False_)(1.0, 2.0, 3.0)[0] != 0   # NaN, a numpy bool is a bool
    assert repr(s.RollingOLS(10, 2)) == "RollingOLS(window_size=10, n_regressors=2)"
    assert repr(s.RollingOLS(10, 2, np.True_)) == "RollingOLS(window_size=10, n_regressors=2, intercept=True)"
    assert repr(s.ExpandingOLS(2, intercept=np.array(False))) == "ExpandingOLS(n_regressors=2, intercept=False)"


@pytest.mark.parametrize("intercept", [True, False])
def test_long_random_walk_at_level_1e5(intercept):
    """Prices near 1e5 with unit moves over 2e5 rows, checked at the end.
    With an intercept the slopes keep their digits at this level. Without one
    the raw normal equations square the condition number of the regressors,
    so the coefficients keep about 16 - 2 log10(cond) digits: the tolerance
    is a small multiple of eps * cond^2."""
    rng = np.random.default_rng(5)
    T, k, w = 200_000, 3, 250
    X = 1e5 + np.cumsum(rng.standard_normal((T, k)), axis=0)
    y = 2.0 * (X[:, 0] - 1e5) - (X[:, 1] - 1e5) + 0.5 * X[:, 2] + rng.standard_normal(T)
    off = 4 if intercept else 3
    for out, lo in ((s.RollingOLS(w, k, intercept=intercept)(y, *X.T), T - w),
                    (s.ExpandingOLS(k, intercept=intercept)(y, *X.T), 0)):
        A, v = X[lo:], y[lo:]
        if intercept:
            ref = _centered_slopes(v, A)
            np.testing.assert_allclose(out[-1, off:], ref, rtol=1e-11)
        else:
            ref = np.linalg.lstsq(A, v, rcond=None)[0]
            cond = np.linalg.cond(A)
            err = np.max(np.abs(out[-1, off:] - ref) / np.abs(ref))
            assert err <= 100 * np.finfo(float).eps * cond ** 2, (err, cond)


@pytest.mark.parametrize("intercept", [True, False])
def test_an_infinity_leaves_the_window(intercept):
    """An infinite input enters the window (it is not NaN): the fit is NaN while
    it is inside, and the exact refit when it leaves restores the fit."""
    y, X = _data(2, T=200)
    w = 20
    bad_x, bad_y = X.copy(), y.copy()
    bad_x[100, 0] = np.inf
    bad_y[150] = -np.inf
    out = s.RollingOLS(w, 2, intercept=intercept)(bad_y, *bad_x.T)
    ref = s.RollingOLS(w, 2, intercept=intercept)(y, *X.T)
    assert np.isnan(out[100:100 + w, 2:]).all()
    assert np.isnan(out[150:150 + w, 2:]).all()
    for rows in (slice(100 + w, 150), slice(150 + w, 200)):
        np.testing.assert_allclose(out[rows, 2:], ref[rows, 2:], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(out[100 + w + 1:150, :2], ref[100 + w + 1:150, :2],
                               rtol=1e-12, atol=1e-12)
