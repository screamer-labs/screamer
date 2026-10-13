"""Regressions stay accurate when the level is large next to the spread.

The sibling of ``test_moment_stability.py`` for the fits (tracker rows C7 and
C9): the time-index regressions (RollingPoly1, RollingPoly2, RollingTSF,
ExpandingSlope), the AR(1) fit RollingOU and the RollingADF statistic. Each
test feeds ``level + noise`` and compares the operator with its documented
estimator evaluated in exact rational arithmetic on the same float inputs.

The error is measured against the natural size of each output (``scale``):
the window's standard deviation for fitted values and the OU mean, sd_y / sd_t
for a slope against time and sd_y / sd_t^2 for a curvature, the value itself
for the OU noise std, and 1 for the dimensionless mrr and ADF statistic.

Also here: RollingZscore's start_policy (row C8), and batch == per-event for
every operator changed with them.
"""
import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import screamer as s
from tests.regime_helpers import assert_bitwise

LEVELS = [0.0, 1e3, 1e6]
POLICIES = ["strict", "expanding", "zero"]
N = 400
W = 50
TOL = 1e-9
EVENT_TOL = 1e-11


def noise(level, seed=0, n=N):
    return level + np.random.default_rng(seed).standard_normal(n)


def walk(level, seed=5, n=N):
    e = np.random.default_rng(seed).standard_normal(n)
    return level + np.cumsum(e) + 0.2 * np.arange(n)


def ar1(level, seed=3, phi=0.9, n=N):
    e = np.random.default_rng(seed).standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return level + x


def event_series(kind, level=0.0, n=N, seed=4):
    """Unit noise with a spike of 1e4 sd at n/3, or a level jump of 1e5 sd at
    n/3, in either direction."""
    x = noise(0.0, seed, n)
    k = n // 3
    if kind == "spike":
        x[k] += 1e4
    elif kind == "spike_down":
        x[k] -= 1e4
    elif kind == "jump":
        x[k:] += 1e5
    elif kind == "jump_down":
        x[k:] -= 1e5
    return level + x


def check(out, ref, scale, tol=TOL):
    if ref is None:  # undefined: the operator gives NaN
        assert math.isnan(out), out
        return
    err = abs(float(out) - float(ref)) / scale
    assert err <= tol, (float(out), float(ref), err)


def sd(vals):
    n = len(vals)
    m = sum(vals) / n
    return math.sqrt(float(sum((v - m) ** 2 for v in vals) / (n - 1)))


def windows(x, policy="strict", w=W, first=2):
    xs = [Fr(float(v)) for v in x]
    for t in range(len(xs)):
        if policy == "strict" and t < w - 1:
            continue
        win = xs[max(0, t - w + 1):t + 1]
        if policy == "zero":
            win = [Fr(0)] * (w - len(win)) + win
        if len(win) < first:
            continue
        yield t, win


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


# --- time-index regressions --------------------------------------------------

def line_fit(ys):
    """Exact least-squares line against t = 0..n-1: (mean, slope, tbar, sd_t)."""
    n = len(ys)
    tb = Fr(n - 1, 2)
    yb = sum(ys) / n
    stt = sum((t - tb) ** 2 for t in range(n))
    sty = sum((t - tb) * (y - yb) for t, y in enumerate(ys))
    return yb, sty / stt, tb, math.sqrt(float(stt / (n - 1)))


def quad_fit(ys):
    """Exact least-squares (a, b, c) of y = a t^2 + b t + c on t = 0..n-1."""
    S = [[Fr(0)] * 3 for _ in range(3)]
    r = [Fr(0)] * 3
    for t, y in enumerate(ys):
        v = [Fr(t * t), Fr(t), Fr(1)]
        for i in range(3):
            r[i] += v[i] * y
            for j in range(3):
                S[i][j] += v[i] * v[j]
    return solve(S, r)


def poly_refs(win):
    """name -> (exact value, scale) for every time-regression output."""
    n = len(win)
    sdy = sd(win)
    yb, b, tb, sdt = line_fit(win)
    refs = {
        "Poly1.endpoint": (yb + b * (n - 1 - tb), sdy),
        "Poly1.slope": (b, sdy / sdt),
        "TSF": (yb + b * (n - tb), sdy),
    }
    if n >= 3:
        a2, b2, c2 = quad_fit(win)
        tl = n - 1
        refs["Poly2.endpoint"] = (a2 * tl * tl + b2 * tl + c2, sdy)
        refs["Poly2.slope"] = (2 * a2 * tl + b2, sdy / sdt)
        refs["Poly2.curvature"] = (2 * a2, sdy / sdt ** 2)
    return refs


def poly_outputs(x, policy, w=W):
    out = {
        "Poly1.endpoint": s.RollingPoly1(w, 0, policy)(x),
        "Poly1.slope": s.RollingPoly1(w, 1, policy)(x),
        "Poly2.endpoint": s.RollingPoly2(w, 0, policy)(x),
        "Poly2.slope": s.RollingPoly2(w, 1, policy)(x),
        "Poly2.curvature": s.RollingPoly2(w, 2, policy)(x),
    }
    if policy == "strict":
        out["TSF"] = s.RollingTSF(w)(x)
    return out


def check_polys(x, policy, w=W, tol=TOL, every=7):
    outs = poly_outputs(x, policy, w)
    for t, win in list(windows(x, policy, w, first=4))[::every]:
        for name, (ref, scale) in poly_refs(win).items():
            if name in outs:
                check(outs[name][t], ref, scale, tol)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("level", LEVELS)
def test_time_regressions_are_stable_at_large_level(level, policy):
    check_polys(noise(level), policy)


@pytest.mark.parametrize("level", LEVELS)
def test_time_regressions_on_a_trend(level):
    check_polys(walk(level), "strict")


@pytest.mark.parametrize("kind", ["spike", "spike_down", "jump", "jump_down"])
def test_time_regressions_after_spike_or_jump(kind):
    # Every step from just before the event until it has left the window.
    x = event_series(kind)
    k = N // 3
    outs = poly_outputs(x, "strict")
    for t, win in windows(x, "strict"):
        if k - 2 <= t <= k + W + 5:
            for name, (ref, scale) in poly_refs(win).items():
                check(outs[name][t], ref, scale, EVENT_TOL)


def expanding_slope_ref(xs, t):
    _, b, _, sdt = line_fit(xs[:t + 1])
    return b, sd(xs[:t + 1]) / sdt


@pytest.mark.parametrize("kind", ["noise", "walk", "jump", "spike"])
@pytest.mark.parametrize("level", LEVELS)
def test_expanding_slope_is_stable_at_large_level(level, kind):
    if kind == "noise":
        x = noise(level)
    elif kind == "walk":
        x = walk(level)
    else:
        x = event_series(kind, level)
    out = s.ExpandingSlope()(x)
    xs = [Fr(float(v)) for v in x]
    for t in range(3, N, 7):
        ref, scale = expanding_slope_ref(xs, t)
        check(out[t], ref, scale)


# --- RollingOU (C7) ----------------------------------------------------------

def ou_refs(win, x_t):
    """name -> (exact value, scale) for the AR(1) fit of the window's pairs."""
    us, vs = win[:-1], win[1:]
    m = len(us)
    ub, vb = sum(us) / m, sum(vs) / m
    suu = sum((u - ub) ** 2 for u in us)
    suv = sum((u - ub) * (v - vb) for u, v in zip(us, vs))
    svv = sum((v - vb) ** 2 for v in vs)
    if suu == 0:  # u constant: no fit
        return {o: (None, 1.0) for o in OU_OUTPUTS}
    b = suv / suu
    a = vb - b * ub
    mu = a / (1 - b)
    sdu = math.sqrt(float(suu / m))
    # The mean's own size above the data: an error in b moves it by about
    # (mu - mean u) * db / (1 - b).
    mscale = max(sdu, abs(float(mu - ub)))
    refs = {"mrr": (1 - b, 1.0), "mean": (mu, mscale), "relmean": (mu - x_t, mscale)}
    if m >= 3:
        std = math.sqrt(float((svv - b * suv) / (m - 2)))
        refs["std"] = (std, std)
    return refs


OU_OUTPUTS = ["mrr", "mean", "relmean", "std"]


def check_ou(x, policy, w=W, tol=TOL, every=7, steps=None):
    outs = {o: s.RollingOU(w, o, policy)(x) for o in OU_OUTPUTS}
    for t, win in list(windows(x, policy, w, first=4))[::every]:
        if steps is not None and t not in steps:
            continue
        for name, (ref, scale) in ou_refs(win, win[-1]).items():
            check(outs[name][t], ref, scale, tol)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("level", LEVELS)
def test_rolling_ou_matches_the_ar1_fit(level, policy):
    check_ou(ar1(level), policy)


def test_rolling_ou_after_jump():
    x = ar1(0.0)
    k = N // 3
    x[k:] += 1e5
    check_ou(x, "strict", tol=EVENT_TOL, every=1, steps=range(k - 2, k + W + 5))


@pytest.mark.parametrize("level", [1e3, 1e6])
def test_rolling_ou_level_shift(level):
    """mean moves by the level; mrr, relmean and std do not."""
    x = ar1(0.0)
    for o in OU_OUTPUTS:
        base = s.RollingOU(W, o)(x)
        moved = s.RollingOU(W, o)(x + level)
        if o == "mean":
            moved = moved - level
        ok = ~np.isnan(base)
        assert np.array_equal(ok, ~np.isnan(moved))
        np.testing.assert_allclose(moved[ok], base[ok], rtol=1e-9, atol=1e-9 * max(1.0, level * 1e-6))


def test_rolling_ou_warmup():
    x = ar1(0.0, n=20)
    for o in OU_OUTPUTS:
        strict = s.RollingOU(10, o)(x)
        assert np.all(np.isnan(strict[:9])) and np.all(np.isfinite(strict[9:]))
        exp = s.RollingOU(10, o, "expanding")(x)
        # mrr and mean from 2 pairs (3 samples), std from 3 pairs.
        first = 3 if o == "std" else 2
        assert np.all(np.isnan(exp[:first])) and np.all(np.isfinite(exp[first:]))
        zero = s.RollingOU(10, o, "zero")(x)
        padded = s.RollingOU(10, o, "zero")(np.concatenate([np.zeros(9), x]))[9:]
        np.testing.assert_allclose(zero, padded, rtol=1e-12, atol=1e-12)


def test_rolling_ou_validation():
    # One pair has no slope: window_size must be 3 or more.
    for w in (0, 1, 2):
        with pytest.raises(ValueError, match=f"^RollingOU: window_size must be at least 3, got {w}$"):
            s.RollingOU(w)
    with pytest.raises(ValueError):
        s.RollingOU(10, output="bogus")


def test_rolling_ou_std_needs_three_pairs():
    """window_size 3 gives 2 pairs: the residual has no degrees of freedom."""
    x = ar1(0.0, n=40)
    for p in POLICIES:
        assert np.all(np.isnan(s.RollingOU(3, "std", p)(x)))
        assert np.any(np.isfinite(s.RollingOU(3, "mrr", p)(x)))
    assert np.all(np.isfinite(s.RollingOU(4, "std")(x)[3:]))


def test_nan_residual_gives_nan_std():
    """An infinity in the window makes the residual NaN: the noise estimate is
    NaN with the other outputs, not 0."""
    x = ar1(0.0, n=60)
    x[30] = np.inf
    std = s.RollingOU(10, "std")(x)
    mrr = s.RollingOU(10, "mrr")(x)
    both_nan = np.isnan(mrr) & ~np.isnan(std)
    assert not both_nan.any(), np.flatnonzero(both_nan)
    assert np.isnan(std[30])
    y = noise(0.0, n=60, seed=2)
    out = np.asarray(s.RollingLinearRegression(10)(y, x))
    slope, stderr = out[:, 0], out[:, 3]
    assert np.isnan(stderr[30])
    assert not (np.isnan(slope) & ~np.isnan(stderr)).any()


# --- RollingADF --------------------------------------------------------------

def adf_ref(ys, lags):
    rows = []
    for t in range(lags + 1, len(ys)):
        row = [Fr(1), ys[t - 1]] + [ys[t - 1 - i] - ys[t - 2 - i] for i in range(lags)]
        rows.append((row, ys[t] - ys[t - 1]))
    d = lags + 2
    XtX = [[sum(r[i] * r[j] for r, _ in rows) for j in range(d)] for i in range(d)]
    Xty = [sum(r[i] * y for r, y in rows) for i in range(d)]
    beta = solve(XtX, Xty)
    rss = sum((y - sum(bi * ri for bi, ri in zip(beta, r))) ** 2 for r, y in rows)
    inv11 = solve(XtX, [Fr(int(i == 1)) for i in range(d)])[1]
    return float(beta[1]) / math.sqrt(float(rss / (len(rows) - d) * inv11))


@pytest.mark.parametrize("lags", [0, 2])
@pytest.mark.parametrize("level", LEVELS)
def test_rolling_adf_is_stable_at_large_level(level, lags):
    w = 60
    x = ar1(level, seed=7, phi=0.8)
    out = s.RollingADF(w, lags)(x)
    xs = [Fr(float(v)) for v in x]
    for t in range(w + lags, N, 11):
        ref = adf_ref(xs[t - w - lags:t + 1], lags)
        check(out[t], ref, max(1.0, abs(ref)))


def test_rolling_adf_after_jump():
    w, lags = 60, 1
    x = ar1(0.0, seed=7, phi=0.8)
    k = N // 3
    x[k:] += 1e5
    out = s.RollingADF(w, lags)(x)
    xs = [Fr(float(v)) for v in x]
    # From the step the jump's difference (1e5 sd in dy and its lag) has
    # left the window. While it is inside, one row dominates X'X and the
    # normal equations lose digits to it whatever the origin.
    for t in range(k + w + lags, k + 2 * w, 3):
        ref = adf_ref(xs[t - w - lags:t + 1], lags)
        check(out[t], ref, max(1.0, abs(ref)), EVENT_TOL)


# --- RollingZscore start_policy (C8) -----------------------------------------

@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("level", [0.0, 1e3])
def test_rolling_zscore_start_policy_matches_composition(policy, level):
    x = noise(level, n=120)
    w = 20
    z = s.RollingZscore(w, policy)(x)
    ref = (x - s.RollingMean(w, policy)(x)) / s.RollingStd(w, policy)(x)
    assert np.array_equal(np.isnan(z), np.isnan(ref))
    # The composition subtracts two level-sized values, so it carries the
    # level's rounding.
    np.testing.assert_allclose(z, ref, rtol=0, atol=1e-12 * max(1.0, level))


def test_rolling_zscore_start_policy_warmup():
    x = noise(0.0, n=30)
    w = 10
    assert np.all(np.isnan(s.RollingZscore(w)(x)[:w - 1]))
    exp = s.RollingZscore(w, "expanding")(x)
    assert np.isnan(exp[0]) and np.all(np.isfinite(exp[1:]))
    assert np.all(np.isfinite(s.RollingZscore(w, "zero")(x)))


# --- batch == per-event ------------------------------------------------------

def changed_ops():
    ops = {}
    for p in POLICIES:
        for d in (0, 1):
            ops[f"Poly1({d},{p})"] = lambda d=d, p=p: s.RollingPoly1(7, d, p)
        for d in (0, 1, 2):
            ops[f"Poly2({d},{p})"] = lambda d=d, p=p: s.RollingPoly2(7, d, p)
        for o in OU_OUTPUTS:
            ops[f"OU({o},{p})"] = lambda o=o, p=p: s.RollingOU(7, o, p)
        ops[f"Zscore({p})"] = lambda p=p: s.RollingZscore(7, p)
    ops["TSF"] = lambda: s.RollingTSF(7)
    ops["ExpandingSlope"] = lambda: s.ExpandingSlope()
    ops["ADF"] = lambda: s.RollingADF(12, 1)
    return ops


def hard_input():
    x = noise(1e6, n=300, seed=9)
    x[[20, 21, 90, 150]] = np.nan
    x[60] += 1e6
    x[120:] += 1e5
    x[200] = np.inf
    x[260:] -= 3e5
    return x


@pytest.mark.parametrize("name", list(changed_ops()))
def test_batch_equals_per_event(name):
    make = changed_ops()[name]
    x = hard_input()
    batch = np.asarray(make()(x))
    op = make()
    stream = np.array([op(float(v)) for v in x])
    assert_bitwise(batch, stream)
    # Columns of a 2-D input match the 1-D run.
    two = np.asarray(make()(np.stack([x, x[::-1]], axis=1)))
    assert_bitwise(two[:, 0], batch)
    assert_bitwise(two[:, 1], np.asarray(make()(x[::-1].copy())))
