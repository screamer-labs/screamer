"""
Tests for WMA: linearly-weighted moving average.

The newest sample carries weight w, the oldest weight 1, so

    WMA[t] = (1*x[t-w+1] + 2*x[t-w+2] + ... + w*x[t]) / (w(w+1)/2)

Computed in O(1) per step via the identity

    W[t] - W[t-1] = w*x[t] - S[t-1]

where S[t-1] is the simple rolling sum of the previous window.
Validated against three explicit per-step reference implementations
(one per start_policy) and against pandas .rolling().apply().
"""
import numpy as np
import pandas as pd
import pytest

from screamer import WMA


# ---------------------------------------------------------------------------
# Reference implementations: brute-force, one per start policy.
# ---------------------------------------------------------------------------

def ref_wma_strict(x, w):
    out = np.full(len(x), np.nan)
    weights = np.arange(1, w + 1, dtype=float)
    denom = weights.sum()
    for i in range(w - 1, len(x)):
        out[i] = np.dot(x[i - w + 1:i + 1], weights) / denom
    return out


def ref_wma_expanding(x, w):
    out = np.empty(len(x))
    for i in range(len(x)):
        n = min(i + 1, w)
        weights = np.arange(1, n + 1, dtype=float)
        out[i] = np.dot(x[i - n + 1:i + 1], weights) / weights.sum()
    return out


def ref_wma_zero(x, w):
    """Zero-pad missing past, divide by full weight sum."""
    out = np.empty(len(x))
    weights = np.arange(1, w + 1, dtype=float)
    denom = weights.sum()
    for i in range(len(x)):
        v = np.zeros(w)
        start = max(0, i - w + 1)
        end = i + 1
        v[w - (end - start):] = x[start:end]
        out[i] = np.dot(v, weights) / denom
    return out


# ---------------------------------------------------------------------------
# Reference parity (O(1) recurrence vs explicit per-window dot product)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w", [3, 5, 10, 20])
def test_strict_matches_reference(w):
    rng = np.random.default_rng(w)
    x = rng.standard_normal(80)
    np.testing.assert_allclose(WMA(w, "strict")(x), ref_wma_strict(x, w),
                               equal_nan=True, atol=1e-12)


@pytest.mark.parametrize("w", [3, 5, 10, 20])
def test_expanding_matches_reference(w):
    rng = np.random.default_rng(w + 100)
    x = rng.standard_normal(80)
    np.testing.assert_allclose(WMA(w, "expanding")(x), ref_wma_expanding(x, w),
                               atol=1e-12)


@pytest.mark.parametrize("w", [3, 5, 10, 20])
def test_zero_matches_reference(w):
    rng = np.random.default_rng(w + 200)
    x = rng.standard_normal(80)
    np.testing.assert_allclose(WMA(w, "zero")(x), ref_wma_zero(x, w),
                               atol=1e-12)


def test_default_policy_is_strict():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(40)
    np.testing.assert_array_equal(WMA(5)(x), WMA(5, "strict")(x))


# ---------------------------------------------------------------------------
# Pandas parity
# ---------------------------------------------------------------------------

def test_matches_pandas_rolling_apply():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(80)
    w = 10
    ref = pd.Series(x).rolling(w).apply(
        lambda v: np.dot(v, np.arange(1, w + 1)) / (w * (w + 1) / 2),
        raw=True,
    ).to_numpy()
    np.testing.assert_allclose(WMA(w)(x), ref, equal_nan=True, atol=1e-12)


# ---------------------------------------------------------------------------
# Algebraic / sanity checks
# ---------------------------------------------------------------------------

def test_constant_input_is_constant():
    """For constant input c, WMA = c whenever a full window is available.
    Strict: post-warmup only. Expanding: all samples (the partial-window
    weighted average of a constant is the constant). Zero: post-warmup
    only (warmup is dampened by the implicit zero-pad)."""
    c = 3.7
    w = 5
    x = np.full(30, c)
    np.testing.assert_allclose(WMA(w, "strict")(x)[w - 1:], c, atol=1e-12)
    np.testing.assert_allclose(WMA(w, "expanding")(x), c, atol=1e-12)
    np.testing.assert_allclose(WMA(w, "zero")(x)[w - 1:], c, atol=1e-12)


def test_linear_input_matches_explicit():
    """For x[t] = t, WMA after warmup equals (sum k*(t-w+k))/(sum k)."""
    w = 5
    x = np.arange(20, dtype=float)
    out = WMA(w)(x)
    weights = np.arange(1, w + 1, dtype=float)
    for t in range(w - 1, len(x)):
        expected = np.dot(x[t - w + 1:t + 1], weights) / weights.sum()
        assert out[t] == pytest.approx(expected, abs=1e-12)


def test_strict_warmup_is_nan():
    out = WMA(7)(np.arange(10, dtype=float))
    assert np.all(np.isnan(out[:6]))
    assert np.all(np.isfinite(out[6:]))


def test_zero_warmup_is_dampened():
    """Under "zero" policy the warmup output is biased toward zero
    (missing past treated as zeros). Specifically: at t=0,
    WMA = w * x[0] / (w(w+1)/2) = 2 * x[0] / (w+1)."""
    w = 5
    obj = WMA(w, "zero")
    assert obj(1.0) == pytest.approx(2.0 / (w + 1))


def test_expanding_first_sample_equals_input():
    """Under "expanding" policy at t=0 there's only one sample with
    weight 1, divisor 1 -- so WMA[0] = x[0]."""
    obj = WMA(5, "expanding")
    assert obj(1.7) == 1.7


def test_warmup_to_post_warmup_continuity():
    """At t=w-1 the warmup branch and the post-warmup branch must
    produce the same value (algorithm relies on this transition)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(20)
    w = 6
    for policy in ("strict", "expanding", "zero"):
        out = WMA(w, policy)(x)
        # The first valid sample under all policies (post-warmup) must
        # match the strict reference (pure rolling form).
        assert out[w - 1] == pytest.approx(ref_wma_strict(x, w)[w - 1],
                                            abs=1e-12)


def test_reset_clears_history():
    w = 5
    obj = WMA(w)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(20)
    first = np.array([obj(v) for v in x])
    obj.reset()
    second = np.array([obj(v) for v in x])
    np.testing.assert_array_equal(first, second)


def test_scalar_loop_matches_array():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(40)
    w = 7
    obj = WMA(w)
    streamed = np.array([obj(v) for v in x])
    np.testing.assert_allclose(streamed, WMA(w)(x), equal_nan=True, atol=1e-12)


def test_2d_per_column_independence():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((30, 3))
    w = 6
    out_2d = WMA(w)(X)
    for k in range(X.shape[1]):
        np.testing.assert_allclose(
            out_2d[:, k], WMA(w)(X[:, k].copy()),
            equal_nan=True, atol=1e-12,
        )
