"""Tests for the Expanding* whole-history statistic family.

Each Expanding* functor accumulates over the entire history seen since the
last ``reset()`` (no window, no start_policy). The moment stats mirror the
ddof / bias conventions of their Rolling* analogues, which coincide with
pandas' ``.expanding()`` defaults (var/std ddof=1, skew = bias-corrected G1,
kurt = Fisher excess with bias correction).
"""
import numpy as np
import pandas as pd

from screamer import (
    ExpandingMean, ExpandingVar, ExpandingStd, ExpandingSkew, ExpandingKurt,
    ExpandingSlope, ExpandingSum, ExpandingMax, ExpandingMin, ExpandingProd,
)


def _pd(x, method):
    return getattr(pd.Series(x).expanding(), method)().to_numpy()


def test_expanding_moments_match_pandas():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    np.testing.assert_allclose(np.asarray(ExpandingMean()(x)), _pd(x, "mean"), rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(np.asarray(ExpandingVar()(x)),  _pd(x, "var"),  rtol=1e-9,  equal_nan=True)
    np.testing.assert_allclose(np.asarray(ExpandingStd()(x)),  _pd(x, "std"),  rtol=1e-9,  equal_nan=True)
    np.testing.assert_allclose(np.asarray(ExpandingSkew()(x)), _pd(x, "skew"), rtol=1e-8,  equal_nan=True)
    np.testing.assert_allclose(np.asarray(ExpandingKurt()(x)), _pd(x, "kurt"), rtol=1e-8,  equal_nan=True)


def test_expanding_aliases_match_pandas():
    rng = np.random.default_rng(1)
    x = rng.normal(size=100)
    np.testing.assert_allclose(np.asarray(ExpandingSum()(x)), _pd(x, "sum"), rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(np.asarray(ExpandingMax()(x)), _pd(x, "max"), rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(np.asarray(ExpandingMin()(x)), _pd(x, "min"), rtol=1e-10, equal_nan=True)
    # numpy cumprod as the reference for prod (pandas .expanding().apply is slow)
    np.testing.assert_allclose(np.asarray(ExpandingProd()(x)), np.cumprod(x), rtol=1e-10, equal_nan=True)


def test_expanding_slope_matches_ols():
    rng = np.random.default_rng(2)
    x = rng.normal(size=50)
    out = np.asarray(ExpandingSlope()(x))
    # Reference: OLS slope of y[0..t] against t = 0..t, for each t.
    ref = np.full_like(x, np.nan)
    for t in range(1, len(x)):
        tt = np.arange(t + 1)
        ref[t] = np.polyfit(tt, x[: t + 1], 1)[0]
    np.testing.assert_allclose(out, ref, rtol=1e-8, equal_nan=True)


def test_expanding_moment_warmup_nans():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    # mean valid from n=1; var/std from n=2; skew from n=3; kurt from n=4.
    assert np.isfinite(np.asarray(ExpandingMean()(x))).all()
    v = np.asarray(ExpandingVar()(x));  assert np.isnan(v[0]) and np.isfinite(v[1:]).all()
    s = np.asarray(ExpandingSkew()(x)); assert np.isnan(s[:2]).all() and np.isfinite(s[2:]).all()
    k = np.asarray(ExpandingKurt()(x)); assert np.isnan(k[:3]).all() and np.isfinite(k[3:]).all()


def test_reset_restarts_accumulation():
    x = np.array([1.0, 2.0, 3.0])
    m = ExpandingMean()
    _ = m(x)
    m.reset()
    # After reset the accumulator restarts: first fresh sample is its own mean.
    assert float(np.asarray(m(np.array([10.0]))).reshape(-1)[0]) == 10.0


def test_nan_ignore_recovers():
    x = np.array([1.0, np.nan, 3.0])
    out = np.asarray(ExpandingMean()(x))
    assert out[0] == 1.0
    assert np.isnan(out[1])
    # NaN skipped in state: mean of {1, 3} = 2.0
    assert out[2] == 2.0
