"""The exponentially weighted ratios of a series that stops varying.

Once a series stops varying, its weighted variance decays as (1 - alpha)^n
while the sums keep the rounding of its running mean, so a ratio with that
variance in the denominator ends in rounding noise (EwBeta returned 1, 2 or
-1 on a regressor that had been constant for 19000 rows at alpha = 0.1). The
operators that divide by it give NaN once the variance is below kEwFlat^2 =
(2^24 eps)^2 times a peak of it that decays by (1 - alpha)^(1/8) per row
(include/screamer/detail/ew_flat.h): about 45 / alpha rows after the series
stops, 414 to 434 rows at alpha = 0.1 in these tests.

The references are exact weighted moments, from sums kept in 300-digit
decimals.
"""
from decimal import Decimal, getcontext

import numpy as np
import pytest

import screamer as s

ALPHA = 0.1
N1 = 1000                 # varying rows before the series stops
ONSET = (380, 470)        # NaN from a row in this range after it stops


def _dec(v):
    return Decimal(float(v))


def _exact(x, y, t, alpha=ALPHA):
    """Exact weighted moments of x (and y) over rows 0..t, with the weights
    (1 - alpha)^(t - i) the operators use."""
    getcontext().prec = 300
    a = _dec(1.0 - alpha)
    W = W2 = Sx = Sy = Sxx = Syy = Sxy = Sx3 = Sx4 = Decimal(0)
    for i in range(t + 1):
        xi = _dec(x[i])
        yi = _dec(y[i]) if y is not None else Decimal(0)
        W = W * a + 1
        W2 = W2 * a * a + 1
        Sx = Sx * a + xi
        Sy = Sy * a + yi
        Sxx = Sxx * a + xi * xi
        Syy = Syy * a + yi * yi
        Sxy = Sxy * a + xi * yi
        Sx3 = Sx3 * a + xi ** 3
        Sx4 = Sx4 * a + xi ** 4
    mx, my = Sx / W, Sy / W
    q2, q3, q4 = Sxx / W, Sx3 / W, Sx4 / W
    m2 = q2 - mx * mx
    m3 = q3 - 3 * mx * q2 + 2 * mx ** 3
    m4 = q4 - 4 * mx * q3 + 6 * mx * mx * q2 - 3 * mx ** 4
    vy = Syy / W - my * my
    cxy = Sxy / W - mx * my
    n = W * W / W2
    return dict(x=_dec(x[t]), y=_dec(y[t]) if y is not None else None,
                mx=mx, my=my, m2=m2, m3=m3, m4=m4, vy=vy, cxy=cxy, n=n)


def _exact_value(name, x, y, t):
    """The exact output of `name` at row t (x the first input, y the second)."""
    e = _exact(x, y, t)
    if name in ("EwBeta", "EwAlpha", "EwSpread"):
        beta = e["cxy"] / e["vy"]
        if name == "EwBeta":
            return float(beta)
        if name == "EwAlpha":
            return float(e["mx"] - beta * e["my"])
        return float(e["x"] - beta * e["y"])
    if name == "EwCorr":
        return float(e["cxy"] / (e["m2"] * e["vy"]).sqrt())
    n, m2 = e["n"], e["m2"]
    if name == "EwZscore":
        return float((e["x"] - e["mx"]) / (m2 * n / (n - 1)).sqrt())
    if name == "EwSkew":
        g1 = e["m3"] / (m2 * m2.sqrt())
        return float((n * (n - 1)).sqrt() / (n - 2) * g1)
    if name == "EwKurt":
        g2 = e["m4"] / (m2 * m2)
        return float((n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 - 3 * (n - 1)))
    raise ValueError(name)


# (case, operator, how the flat series f and a varying series r feed it).
# EwKyleLambda is EwBeta with its inputs swapped, EwAlpha, EwSpread and
# EwResidualStd use EwBeta's slope.
CASES = {
    "EwBeta": ("EwBeta", lambda f, r: (r, f)),
    "EwKyleLambda": ("EwKyleLambda", lambda f, r: (f, r)),
    "EwAlpha": ("EwAlpha", lambda f, r: (r, f)),
    "EwSpread": ("EwSpread", lambda f, r: (r, f)),
    "EwResidualStd": ("EwResidualStd", lambda f, r: (r, f)),
    "EwCorr first flat": ("EwCorr", lambda f, r: (f, r)),
    "EwCorr second flat": ("EwCorr", lambda f, r: (r, f)),
    "EwZscore": ("EwZscore", lambda f, r: (f,)),
    "EwSkew": ("EwSkew", lambda f, r: (f,)),
    "EwKurt": ("EwKurt", lambda f, r: (f,)),
}
# Cases with an exact reference (_exact_value), as (operator, x, y) of f, r.
EXACT = {
    "EwBeta": lambda f, r: (r, f),
    "EwAlpha": lambda f, r: (r, f),
    "EwSpread": lambda f, r: (r, f),
    "EwCorr": lambda f, r: (f, r),
    "EwZscore": lambda f, r: (f, None),
    "EwSkew": lambda f, r: (f, None),
    "EwKurt": lambda f, r: (f, None),
}


def _run(case, f, r):
    name, feed = CASES[case]
    return getattr(s, name)(alpha=ALPHA)(*feed(f, r))


def _stops(level, seed, n_flat=3000):
    rng = np.random.default_rng(seed)
    f = np.r_[rng.standard_normal(N1), np.full(n_flat, level)]
    r = rng.standard_normal(len(f))
    return f, r


@pytest.mark.parametrize("level", [3.7, 1.0, 0.0, -5.0])
@pytest.mark.parametrize("case", list(CASES))
def test_series_that_stops_varying_gives_nan(case, level):
    for seed in range(3):
        f, r = _stops(level, seed)
        out = _run(case, f, r)
        assert np.isfinite(out[5:N1 + ONSET[0]]).all(), (seed, case)
        nan = np.flatnonzero(np.isnan(out[N1:]))
        assert ONSET[0] <= nan[0] <= ONSET[1], (seed, nan[0])
        # It stays NaN.
        assert np.isnan(out[N1 + nan[0]:]).all(), seed


@pytest.mark.parametrize("level", [3.7, 1.0, 0.0, -5.0])
@pytest.mark.parametrize("name", ["EwBeta", "EwAlpha", "EwSpread"])
def test_flat_target_is_not_flagged(name, level):
    """Only the regressor's variance is a denominator: a flat target gives a
    slope that decays toward 0, exact, and no NaN."""
    f, r = _stops(level, 0, n_flat=1500)
    out = getattr(s, name)(alpha=ALPHA)(f, r)
    assert np.isfinite(out[5:]).all()
    t = len(f) - 1
    ref = _exact_value(name, f, r, t)
    assert out[t] == pytest.approx(ref, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("level", [3.7, -5.0])
@pytest.mark.parametrize("name", list(EXACT))
def test_exact_up_to_the_nan(name, level):
    """Before the NaN the ratios match the exact weighted value: to 1e-9 a
    hundred rows after the series stops, and still to 5 digits (or 1e-12
    absolute for a ratio that decays to 0) on the row before the NaN."""
    f, r = _stops(level, 1, n_flat=1000)
    x, y = EXACT[name](f, r)
    out = getattr(s, name)(alpha=ALPHA)(*((x,) if y is None else (x, y)))
    last = N1 + np.flatnonzero(np.isnan(out[N1:]))[0] - 1
    for t, rel in ((N1 + 100, 1e-9), (last, 1e-5)):
        ref = _exact_value(name, x, y, t)
        assert out[t] == pytest.approx(ref, rel=rel, abs=1e-12), (t, out[t], ref)


@pytest.mark.parametrize("level", [3.7, 0.0])
@pytest.mark.parametrize("name", list(EXACT))
def test_recovers_when_the_series_varies_again(name, level):
    """The first row on which the series varies again is finite and exact;
    the old peak has decayed far below the new variance."""
    f, r = _stops(level, 2, n_flat=1500)
    rng = np.random.default_rng(7)
    f2 = np.r_[f, level + rng.standard_normal(50)]
    r2 = np.r_[r, rng.standard_normal(50)]
    x, y = EXACT[name](f2, r2)
    out = getattr(s, name)(alpha=ALPHA)(*((x,) if y is None else (x, y)))
    t0 = len(f)
    assert np.isnan(out[t0 - 1])
    assert np.isfinite(out[t0:]).all()
    for t in (t0, t0 + 1, len(f2) - 1):
        ref = _exact_value(name, x, y, t)
        assert out[t] == pytest.approx(ref, rel=1e-9, abs=1e-12), t


def _first_finite_after_last_nan(v, start):
    nan = np.flatnonzero(np.isnan(v[start:]))
    return start + (nan[-1] + 1 if len(nan) else 0)


@pytest.mark.parametrize("case", ["EwBeta", "EwCorr first flat", "EwCorr second flat",
                                  "EwZscore", "EwSkew", "EwKurt"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_outlier_is_forgotten(case, seed):
    """One huge tick in returns of spread 0.01 raises the decaying peak. A 1e10
    tick (variance about 1e19 against 1e-4) gives NaN until the peak has
    decayed below the variance over kEwFlat^2, after about
    8 ln(1e23 kEwFlat^2) / alpha, some 1070 rows at alpha = 0.1, and none
    after that. A 1e8 tick stays below that bound: no NaN at all."""
    rng = np.random.default_rng(seed)
    T, at = 4000, 1000
    for tick, bound in ((1e10, 1300), (1e8, 0)):
        f = 0.01 * rng.standard_normal(T)
        f[at] = tick
        r = 0.01 * rng.standard_normal(T)
        out = _run(case, f, r)
        assert np.isfinite(out[5:at]).all()
        back = _first_finite_after_last_nan(out, at)
        assert back <= at + bound, (tick, back - at)
        assert np.isfinite(out[back:]).all()
        if tick == 1e8:
            assert np.isfinite(out[5:]).all()
        else:
            assert np.isnan(out[at:]).any()


@pytest.mark.parametrize("case", list(CASES))
def test_slowly_varying_series_is_not_flat(case):
    """A series at 3.7 that moves by 1e-3, slowly, is never flagged: from the
    start, or after unit noise."""
    t = np.arange(6000)
    slow = 3.7 + 1e-3 * np.sin(2 * np.pi * t / 400)
    rng = np.random.default_rng(3)
    r = rng.standard_normal(len(t))
    for f in (slow, np.r_[rng.standard_normal(N1), slow[N1:]]):
        out = _run(case, f, r)
        assert np.isfinite(out[5:]).all()


@pytest.mark.parametrize("case", list(CASES))
def test_constant_from_the_start_is_nan(case):
    """A series that never varies has variance 0: NaN on every row, as pandas
    gives for ewm().corr() with a constant."""
    f = np.full(200, 3.7)
    r = np.random.default_rng(4).standard_normal(200)
    assert np.isnan(_run(case, f, r)).all()


@pytest.mark.parametrize("case", list(CASES))
def test_batch_equals_stream_on_a_flat_tail(case):
    f, r = _stops(3.7, 5, n_flat=800)
    batch = _run(case, f, r)
    name, feed = CASES[case]
    op = getattr(s, name)(alpha=ALPHA)
    cols = feed(f, r)
    stream = np.array([op(*(c[i] for c in cols)) for i in range(len(f))])
    np.testing.assert_array_equal(batch, stream)
