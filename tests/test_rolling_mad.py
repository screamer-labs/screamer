"""
Tests for RollingMad = mean(|x - rolling_mean|).

No closed-form O(1) algorithm exists (the moving mean re-evaluates
every window's abs-deviations); RollingMad runs O(W) per step,
matching pandas. Validated against pandas .apply(...) and a manual
numpy reference.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import RollingMad, RollingMean


def _manual_mad(x, w):
    """Plain-numpy reference: NaN warmup + mean(|x - mean|) per window."""
    out = np.full(len(x), np.nan)
    for i in range(w - 1, len(x)):
        v = x[i - w + 1:i + 1]
        out[i] = np.mean(np.abs(v - v.mean()))
    return out


@pytest.mark.parametrize("w", [3, 5, 10, 20])
def test_matches_manual_numpy_reference(w):
    rng = np.random.default_rng(w)
    x = rng.standard_normal(80)
    np.testing.assert_allclose(RollingMad(w)(x), _manual_mad(x, w),
                               equal_nan=True, atol=1e-12)


@pytest.mark.parametrize("w", [5, 10, 20])
def test_matches_pandas_apply(w):
    rng = np.random.default_rng(w + 10)
    x = rng.standard_normal(60)
    ref = pd.Series(x).rolling(w).apply(
        lambda v: np.mean(np.abs(v - v.mean())), raw=True
    ).to_numpy()
    np.testing.assert_allclose(RollingMad(w)(x), ref, equal_nan=True, atol=1e-12)


def test_constant_input_is_zero():
    out = RollingMad(5)(np.full(20, 3.7))
    # Warmup is NaN under strict policy; rest is exactly 0.
    np.testing.assert_array_equal(np.isnan(out[:4]), [True] * 4)
    np.testing.assert_array_equal(out[4:], 0.0)


def test_arithmetic_progression():
    """For x = a, a+1, ..., a+w-1 the MAD has a closed form: w / 4 for even w,
    (w - 1/w) / 4 for odd w. Just verify the value is non-negative and
    constant after warmup."""
    x = np.arange(15, dtype=float)
    w = 5
    out = RollingMad(w)(x)
    # After warmup, MAD is constant (the relative pattern is the same).
    np.testing.assert_allclose(out[w:], out[w - 1], atol=1e-12)
    assert out[w - 1] > 0


def test_strict_warmup_is_nan():
    out = RollingMad(7)(np.arange(10, dtype=float))
    assert np.all(np.isnan(out[:6]))
    assert np.all(np.isfinite(out[6:]))


def test_expanding_policy():
    """Expanding policy emits MAD over the partial window during warmup."""
    x = np.arange(8, dtype=float)
    w = 5
    out = RollingMad(w, "expanding")(x)
    # All outputs should be finite (no NaN warmup).
    assert np.all(np.isfinite(out))
    # First sample: just one value, MAD = 0.
    assert out[0] == 0.0
    # After full window, must agree with strict policy.
    strict = RollingMad(w, "strict")(x)
    np.testing.assert_allclose(out[w - 1:], strict[w - 1:], atol=1e-12)


def test_scalar_loop_matches_array():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(40)
    w = 7
    obj = RollingMad(w)
    streamed = np.array([obj(v) for v in x])
    np.testing.assert_allclose(streamed, RollingMad(w)(x),
                               equal_nan=True, atol=1e-12)


def test_2d_per_column_independence():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 3))
    w = 6
    out_2d = RollingMad(w)(X)
    for k in range(X.shape[1]):
        np.testing.assert_allclose(
            out_2d[:, k], RollingMad(w)(X[:, k].copy()),
            equal_nan=True, atol=1e-12,
        )


def test_mad_is_non_negative():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(100)
    out = RollingMad(7)(x)
    finite = out[~np.isnan(out)]
    assert np.all(finite >= 0)
