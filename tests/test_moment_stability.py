"""Moment operators stay accurate when the mean is large next to the spread.

Each test feeds ``level + noise`` (unit spread) and compares the operator with
its documented estimator evaluated in exact rational arithmetic on the same
float inputs. Raw power sums of x lose about ``(level / spread)^p`` in relative
precision for the p-th moment, so at level 1e6 a raw-sum kurtosis has no
correct digits; a stable update keeps the error near machine precision.

The error is measured against the natural size of each statistic (``scale``):
the value itself for variances and standard deviations, ``sd_x * sd_y`` for a
covariance, ``sd_x / sd_y`` for a slope, ``sd_x`` for a spread, and 1 for the
dimensionless skewness, kurtosis, correlation and z-score.
"""
import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import screamer as s

LEVELS = [0.0, 1e3, 1e6]
N = 400
W = 50
SPAN = 50
TOL = 1e-9


def fsqrt(q):
    return math.sqrt(float(q))


def signed_sqrt(q_sq, sign_of):
    return math.copysign(fsqrt(q_sq), float(sign_of))


def series(level, seed=0, kind="noise"):
    """level plus unit noise, or plus a drifting walk ("walk"), whose mean moves
    by many standard deviations, so the expanding and weighted operators move
    their origin."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(N)
    if kind == "walk":
        e = np.cumsum(e) + 0.2 * np.arange(N)
    return level + e


def pair(level, seed=1, kind="noise"):
    rng = np.random.default_rng(seed)
    e1, e2 = rng.standard_normal(N), rng.standard_normal(N)
    if kind == "walk":
        e1 = np.cumsum(e1) + 0.2 * np.arange(N)
    return level + 0.8 * e1 + 0.6 * e2, level + e1   # (target, regressor)

KINDS = ["noise", "walk"]


# --- exact window statistics -------------------------------------------------

def window_sums(xs):
    """n and exact central sums M2, M3, M4 of a list of Fractions."""
    n = len(xs)
    m = sum(xs) / n
    return n, m, [None, None] + [sum((x - m) ** p for x in xs) for p in (2, 3, 4)]


def ref_var(n, M):
    return M[2] / (n - 1)


def ref_skew(n, M):
    var = ref_var(n, M)
    c0 = Fr(n, (n - 1) * (n - 2))
    return signed_sqrt(c0 * c0 * M[3] ** 2 / var ** 3, M[3])


def ref_kurt(n, M):
    var = ref_var(n, M)
    return float(Fr(n * (n + 1)) * M[4] / ((n - 1) * (n - 2) * (n - 3) * var * var)
                 - Fr(3 * (n - 1) ** 2, (n - 2) * (n - 3)))


def check(out, ref, scale, tol=TOL):
    assert abs(out - ref) <= tol * scale, (out, ref, abs(out - ref) / scale)


def windows(x, policy="strict", w=W):
    """(t, exact window) pairs the way the rolling operators define them."""
    xs = [Fr(float(v)) for v in x]
    for t in range(len(xs)):
        if policy == "strict" and t < w - 1:
            continue
        win = xs[max(0, t - w + 1):t + 1]
        if policy == "zero":
            win = [Fr(0)] * (w - len(win)) + win
        if len(win) < 5:
            continue
        yield t, win


# --- rolling, one input ------------------------------------------------------

UNI = {
    "RollingVar": lambda n, M, xt, m: (float(ref_var(n, M)), float(ref_var(n, M))),
    "RollingStd": lambda n, M, xt, m: (fsqrt(ref_var(n, M)), fsqrt(ref_var(n, M))),
    "RollingSkew": lambda n, M, xt, m: (ref_skew(n, M), 1.0),
    "RollingKurt": lambda n, M, xt, m: (ref_kurt(n, M), 1.0),
    "RollingZscore": lambda n, M, xt, m: (signed_sqrt((xt - m) ** 2 / ref_var(n, M), xt - m), 1.0),
}


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("name", list(UNI))
def test_rolling_uni_is_stable_at_large_mean(name, level):
    x = series(level)
    out = getattr(s, name)(W)(x)
    for t, win in list(windows(x))[::7]:
        n, m, M = window_sums(win)
        ref, scale = UNI[name](n, M, win[-1], m)
        check(out[t], ref, scale)


@pytest.mark.parametrize("policy", ["expanding", "zero"])
@pytest.mark.parametrize("name", ["RollingVar", "RollingStd", "RollingSkew", "RollingKurt"])
def test_rolling_uni_start_policies(name, policy):
    level = 1e3
    x = series(level)[:120]
    out = getattr(s, name)(W, start_policy=policy)(x)
    for t, win in windows(x, policy):
        n, m, M = window_sums(win)
        ref, scale = UNI[name](n, M, win[-1], m)
        # Zero padding makes the spread of the level size, a well-conditioned case.
        check(out[t], ref, scale)


@pytest.mark.parametrize("level", LEVELS)
def test_bollinger_bands_is_stable_at_large_mean(level):
    x = series(level)
    out = s.BollingerBands(W, 2.0)(x)
    for t, win in list(windows(x))[::7]:
        n, m, M = window_sums(win)
        sd = fsqrt(ref_var(n, M))
        check(out[t, 1], float(m), sd)
        check(out[t, 2] - out[t, 1], 2.0 * sd, sd)
        check(out[t, 1] - out[t, 0], 2.0 * sd, sd)


@pytest.mark.parametrize("level", LEVELS)
def test_rolling_sigma_clip_mean_std_at_large_mean(level):
    # Without bounds nothing is clipped and the outputs are the window's mean
    # and standard deviation, the truncation correction being the identity.
    x = series(level)
    mean = s.RollingSigmaClip(W, output="mean")(x)
    std = s.RollingSigmaClip(W, output="std")(x)
    for t, win in list(windows(x))[::7]:
        n, m, M = window_sums(win)
        sd = fsqrt(ref_var(n, M))
        check(mean[t], float(m), sd)
        check(std[t], sd, sd)


# --- rolling, two inputs -----------------------------------------------------

def bi_ref(xw, yw):
    n = len(xw)
    mx, my = sum(xw) / n, sum(yw) / n
    cxx = sum((a - mx) ** 2 for a in xw)
    cyy = sum((b - my) ** 2 for b in yw)
    cxy = sum((a - mx) * (b - my) for a, b in zip(xw, yw))
    return n, mx, my, cxx, cyy, cxy


@pytest.mark.parametrize("level", LEVELS)
def test_rolling_pair_ops_are_stable_at_large_mean(level):
    x, y = pair(level)
    cov = s.RollingCov(W)(x, y)
    corr = s.RollingCorr(W)(x, y)
    beta = s.RollingBeta(W)(x, y)
    spread = s.RollingSpread(W)(x, y)
    alpha = s.RollingAlpha(W)(x, y)
    tls = s.RollingTLSBeta(W)(x, y)
    lr = s.RollingLinearRegression(W)(x, y)
    X = [Fr(float(v)) for v in x]
    Y = [Fr(float(v)) for v in y]
    for t in range(W - 1, N, 7):
        n, mx, my, cxx, cyy, cxy = bi_ref(X[t - W + 1:t + 1], Y[t - W + 1:t + 1])
        sdx, sdy = fsqrt(cxx / (n - 1)), fsqrt(cyy / (n - 1))
        b = cxy / cyy
        check(cov[t], float(cxy / (n - 1)), sdx * sdy)
        check(corr[t], signed_sqrt(cxy ** 2 / (cxx * cyy), cxy), 1.0)
        check(beta[t], float(b), sdx / sdy)
        check(lr[t, 0], float(b), sdx / sdy)
        check(lr[t, 2], float(cxy ** 2 / (cxx * cyy)), 1.0)
        check(lr[t, 3], fsqrt((cxx - b * cxy) / (n - 2)), sdx)
        # A spread or an intercept subtracts slope * level from a level, so
        # the slope's error reaches it multiplied by the level.
        lvl = sdx + abs(level) * sdx / sdy
        check(spread[t], float(X[t] - b * Y[t]), lvl)
        check(lr[t, 1], float(mx - b * my), lvl)
        check(alpha[t], float(mx - b * my), lvl)
        d = cxx - cyy
        ref_tls = (float(d) + fsqrt(d * d + 4 * cxy * cxy)) / (2 * float(cxy))
        check(tls[t], ref_tls, max(1.0, abs(ref_tls)))


# --- expanding ---------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("level", LEVELS)
def test_expanding_moments_are_stable_at_large_mean(level, kind):
    x = series(level, kind=kind)
    outs = {
        "var": s.ExpandingVar()(x), "std": s.ExpandingStd()(x),
        "skew": s.ExpandingSkew()(x), "kurt": s.ExpandingKurt()(x),
    }
    xs = [Fr(float(v)) for v in x]
    S = [Fr(0)] * 5
    for t, xf in enumerate(xs):
        S = [S[0] + 1] + [S[p] + xf ** p for p in range(1, 5)]
        if t < 4 or t % 7:
            continue
        n = t + 1
        m = S[1] / n
        M = [None, None,
             S[2] - n * m * m,
             S[3] - 3 * m * S[2] + 2 * n * m ** 3,
             S[4] - 4 * m * S[3] + 6 * m * m * S[2] - 3 * n * m ** 4]
        var = ref_var(n, M)
        check(outs["var"][t], float(var), float(var))
        check(outs["std"][t], fsqrt(var), fsqrt(var))
        check(outs["skew"][t], ref_skew(n, M), 1.0)
        check(outs["kurt"][t], ref_kurt(n, M), 1.0)


# --- exponentially weighted --------------------------------------------------

class Dy:
    """An exact dyadic rational n / 2**e; cheaper than Fraction for long recursions."""
    __slots__ = ("n", "e")

    def __init__(self, n=0, e=0):
        self.n, self.e = n, e

    @staticmethod
    def of(v):
        f = Fr(float(v))
        e = f.denominator.bit_length() - 1
        return Dy(f.numerator, e)

    def __add__(self, o):
        if self.e >= o.e:
            return Dy(self.n + (o.n << (self.e - o.e)), self.e)
        return Dy((self.n << (o.e - self.e)) + o.n, o.e)

    def __mul__(self, o):
        return Dy(self.n * o.n, self.e + o.e)

    def fr(self):
        return Fr(self.n, 1 << self.e)


def ew_decays():
    alpha = 2.0 / (SPAN + 1.0)
    return Dy.of(1.0 - alpha), Dy.of((1.0 - alpha) * (1.0 - alpha))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("level", LEVELS)
def test_ew_moments_are_stable_at_large_mean(level, kind):
    x = series(level, kind=kind)
    outs = {name: getattr(s, name)(span=SPAN)(x)
            for name in ("EwVar", "EwStd", "EwSkew", "EwKurt", "EwZscore")}
    a, a2 = ew_decays()
    one = Dy(1, 0)
    Sw = Sww = Dy()
    S = [Dy()] * 5
    for t, v in enumerate(x):
        xd = Dy.of(v)
        Sw, Sww = a * Sw + one, a2 * Sww + one
        p = [one, xd, xd * xd, xd * xd * xd, xd * xd * xd * xd]
        S = [Dy()] + [a * S[k] + p[k] for k in range(1, 5)]
        if t < 5 or t % 7:
            continue
        w, ww, R = Sw.fr(), Sww.fr(), [q.fr() for q in S]
        ne = w * w / ww
        m = R[1] / w
        m2 = R[2] / w - m * m
        m3 = R[3] / w - 3 * m * R[2] / w + 2 * m ** 3
        m4 = R[4] / w - 4 * m * R[3] / w + 6 * m * m * R[2] / w - 3 * m ** 4
        var = m2 * ne / (ne - 1)
        check(outs["EwVar"][t], float(var), float(var))
        check(outs["EwStd"][t], fsqrt(var), fsqrt(var))
        check(outs["EwSkew"][t], signed_sqrt(ne * (ne - 1) / (ne - 2) ** 2 * m3 * m3 / m2 ** 3, m3), 1.0)
        ku = (ne - 1) / ((ne - 2) * (ne - 3)) * ((ne + 1) * m4 / (m2 * m2) - 3 * (ne - 1))
        check(outs["EwKurt"][t], float(ku), 1.0)
        z = xd.fr() - m
        check(outs["EwZscore"][t], signed_sqrt(z * z / var, z), 1.0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("level", LEVELS)
def test_ew_pair_ops_are_stable_at_large_mean(level, kind):
    x, y = pair(level, kind=kind)
    outs = {name: getattr(s, name)(span=SPAN)(x, y)
            for name in ("EwCov", "EwCorr", "EwBeta", "EwSpread")}
    a, a2 = ew_decays()
    one = Dy(1, 0)
    D = [Dy()] * 7
    for t in range(N):
        X, Y = Dy.of(x[t]), Dy.of(y[t])
        D = [a * D[0] + one, a2 * D[1] + one, a * D[2] + X, a * D[3] + Y,
             a * D[4] + X * X, a * D[5] + Y * Y, a * D[6] + X * Y]
        if t < 5 or t % 7:
            continue
        w, ww, sx, sy, sxx, syy, sxy = [q.fr() for q in D]
        ne = w * w / ww
        mx, my = sx / w, sy / w
        cxx, cyy, cxy = sxx / w - mx * mx, syy / w - my * my, sxy / w - mx * my
        k = ne / (ne - 1)
        sdx, sdy = fsqrt(cxx * k), fsqrt(cyy * k)
        b = cxy / cyy
        check(outs["EwCov"][t], float(cxy * k), sdx * sdy)
        check(outs["EwCorr"][t], signed_sqrt(cxy ** 2 / (cxx * cyy), cxy), 1.0)
        check(outs["EwBeta"][t], float(b), sdx / sdy)
        check(outs["EwSpread"][t], float(X.fr() - b * Y.fr()), sdx + abs(level) * sdx / sdy)


# --- details that the rewrite fixed ------------------------------------------

def test_ew_zscore_first_output_is_nan():
    # One sample has no spread (N_eff = 1); a rounding residue gave 0 before.
    for level in LEVELS:
        assert np.isnan(s.EwZscore(span=SPAN)(level + series(0.0))[0])


def test_rolling_linear_regression_zero_policy_streams_zero_filled():
    # A fresh operator fed one event at a time: before the fix it did not
    # reset in its constructor, so it treated the window as empty (NaN for the
    # first events) until its first reset(); batch calls reset first and were
    # right.
    x, y = pair(0.0)
    op = s.RollingLinearRegression(10, start_policy="zero")
    streamed = np.array([op(float(a), float(b)) for a, b in zip(x[:30], y[:30])])
    batch = s.RollingLinearRegression(10, start_policy="zero")(x[:30], y[:30])
    np.testing.assert_array_equal(streamed, batch)
    assert np.all(np.isfinite(streamed[2:]))


def test_rolling_linear_regression_zero_policy_starts_zero_filled():
    x, y = pair(0.0)
    op = s.RollingLinearRegression(10, start_policy="zero")
    fresh = op(x[:30], y[:30])
    op.reset()
    np.testing.assert_array_equal(fresh, op(x[:30], y[:30]))
    padded = s.RollingLinearRegression(10)(np.r_[np.zeros(9), x[:30]], np.r_[np.zeros(9), y[:30]])
    np.testing.assert_allclose(fresh, padded[9:], rtol=1e-12, atol=1e-12)


# --- the original report -----------------------------------------------------

def test_rolling_kurt_unmoved_by_last_bit_input_changes():
    # On a price series near 1000, flipping the last bit of the inputs moved
    # the raw-sum RollingKurt by 3.4e-3 of its scale.
    rng = np.random.default_rng(3)
    x = 1000.0 + np.cumsum(rng.standard_normal(5000)) * 0.1
    y = np.nextafter(x, np.inf)
    a, b = s.RollingKurt(50)(x), s.RollingKurt(50)(y)
    ok = np.isfinite(a)
    assert np.max(np.abs(a[ok] - b[ok])) < 1e-6


# --- spikes and level jumps in the rolling window ----------------------------
#
# The rolling sums move their origin when a far sample leaves or the mean
# drifts, not only when the window wraps, so the outputs stay at the stationary
# accuracy right after a spike leaves or a level changes.

EVENT_TOL = 1e-11
EVENT_WINDOWS = [50, 1000]


def event_series(kind, n, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    k = n // 3
    if kind == "spike":
        x[k] += 1e4
    elif kind == "spike_down":
        x[k] -= 1e4
    elif kind == "jump":
        x[k:] += 1e5
    elif kind == "jump_down":
        x[k:] -= 1e5
    return x


class ExactWindows:
    """Exact sums of x^p (and x*y) over any window, from exact prefix sums."""

    def __init__(self, x, y=None):
        self.X = [Fr(float(v)) for v in x]
        self.P = {p: self._prefix([v ** p for v in self.X]) for p in (1, 2, 3, 4)}
        if y is not None:
            self.Y = [Fr(float(v)) for v in y]
            self.Q = {p: self._prefix([v ** p for v in self.Y]) for p in (1, 2)}
            self.XY = self._prefix([a * b for a, b in zip(self.X, self.Y)])

    @staticmethod
    def _prefix(vals):
        out = [Fr(0)]
        for v in vals:
            out.append(out[-1] + v)
        return out

    def central(self, a, b):
        n = b - a
        S = [n] + [self.P[p][b] - self.P[p][a] for p in (1, 2, 3, 4)]
        m = S[1] / n
        M2 = S[2] - S[1] * m
        M3 = S[3] - 3 * m * S[2] + 2 * n * m ** 3
        M4 = S[4] - 4 * m * S[3] + 6 * m * m * S[2] - 3 * n * m ** 4
        return n, m, [None, None, M2, M3, M4]

    def co_central(self, a, b):
        n = b - a
        sx, sy = self.P[1][b] - self.P[1][a], self.Q[1][b] - self.Q[1][a]
        cxx = self.P[2][b] - self.P[2][a] - sx * sx / n
        cyy = self.Q[2][b] - self.Q[2][a] - sy * sy / n
        cxy = self.XY[b] - self.XY[a] - sx * sy / n
        return n, sx / n, sy / n, cxx, cyy, cxy


@pytest.mark.parametrize("w", EVENT_WINDOWS)
@pytest.mark.parametrize("kind", ["spike", "spike_down", "jump", "jump_down"])
def test_rolling_uni_after_spike_or_jump(kind, w):
    n_total = 3 * w + 200
    x = event_series(kind, n_total)
    E = ExactWindows(x)
    outs = {name: getattr(s, name)(w)(x) for name in UNI}
    bb = s.BollingerBands(w, 2.0)(x)
    worst = 0.0
    for t in range(w - 1, n_total):
        n, m, M = E.central(t - w + 1, t + 1)
        for name, fn in UNI.items():
            ref, scale = fn(n, M, E.X[t], m)
            # Kurtosis reaches about n with a spike in the window: relative.
            worst = max(worst, abs(outs[name][t] - ref) / max(scale, abs(ref)))
        sd = fsqrt(ref_var(n, M))
        worst = max(worst, abs(bb[t, 1] - float(m)) / max(sd, abs(float(m)) * 1e-16 / EVENT_TOL))
        worst = max(worst, abs((bb[t, 2] - bb[t, 1]) - 2 * sd) / max(sd, abs(float(m)) * 1e-16 / EVENT_TOL))
    assert worst < EVENT_TOL, worst


@pytest.mark.parametrize("w", EVENT_WINDOWS)
@pytest.mark.parametrize("kind", ["spike", "jump"])
def test_rolling_pair_after_spike_or_jump_in_both_inputs(kind, w):
    n_total = 3 * w + 200
    rng = np.random.default_rng(7)
    e = rng.standard_normal(n_total)
    y = event_series(kind, n_total)
    x = 0.8 * y + 0.6 * e
    E = ExactWindows(x, y)
    cov = s.RollingCov(w)(x, y)
    corr = s.RollingCorr(w)(x, y)
    beta = s.RollingBeta(w)(x, y)
    tls = s.RollingTLSBeta(w)(x, y)
    lr = s.RollingLinearRegression(w)(x, y)
    worst = 0.0
    for t in range(w - 1, n_total):
        n, mx, my, cxx, cyy, cxy = E.co_central(t - w + 1, t + 1)
        sdx, sdy = fsqrt(cxx / (n - 1)), fsqrt(cyy / (n - 1))
        b = cxy / cyy
        d = cxx - cyy
        ref_tls = (float(d) + fsqrt(d * d + 4 * cxy * cxy)) / (2 * float(cxy))
        errs = [
            abs(cov[t] - float(cxy / (n - 1))) / (sdx * sdy),
            abs(corr[t] - signed_sqrt(cxy ** 2 / (cxx * cyy), cxy)),
            abs(beta[t] - float(b)) / (sdx / sdy),
            abs(lr[t, 0] - float(b)) / (sdx / sdy),
            abs(lr[t, 2] - float(cxy ** 2 / (cxx * cyy))),
            # stderr = sqrt(SST - SSR) cancels when r^2 is near 1, whatever
            # the origin; its error is relative to sd_x^2 / stderr.
            abs(lr[t, 3] - fsqrt(se := (cxx - b * cxy) / (n - 2))) / (sdx * sdx / fsqrt(se)),
            abs(tls[t] - ref_tls) / max(1.0, abs(ref_tls)),
        ]
        worst = max(worst, max(errs))
    assert worst < EVENT_TOL, worst


def test_rolling_window_recovers_after_an_infinity_leaves():
    x = series(1e3)[:200]
    x[60] = np.inf
    out = s.RollingVar(20)(x)
    ref = s.RollingVar(20)(x[81:])
    # Once the infinity has left (step 80 is its last), the output is finite
    # and equals a fresh operator's on the same window.
    np.testing.assert_allclose(out[100:], ref[100 - 81:], rtol=1e-12)
    assert np.all(np.isfinite(out[80:]))
