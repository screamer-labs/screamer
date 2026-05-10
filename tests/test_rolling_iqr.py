"""
Tests for RollingIqr = q75 - q25.

The dedicated implementation uses a single OST queried twice per step,
saving a factor of 2 in memory and inserts vs the obvious composition
of two RollingQuantile instances. Both implementations share the same
linear-interpolation quantile formula, so post-warmup they must agree
to floating-point precision.

Note: RollingQuantile currently has a pre-existing bug where it
returns -2 (sentinel) instead of NaN during warmup, so the composition
reference is only valid for samples >= window_size - 1.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import RollingIqr, RollingQuantile


@pytest.mark.parametrize("w", [5, 10, 20])
def test_matches_pandas(w):
    rng = np.random.default_rng(w)
    x = rng.standard_normal(80)
    s = pd.Series(x)
    ref = s.rolling(w).quantile(0.75).to_numpy() - s.rolling(w).quantile(0.25).to_numpy()
    np.testing.assert_allclose(RollingIqr(w)(x), ref, equal_nan=True, atol=1e-12)


@pytest.mark.parametrize("w", [5, 10, 20])
def test_matches_two_rolling_quantile_composition_post_warmup(w):
    """Composition reference: RollingQuantile(0.75) - RollingQuantile(0.25).
    Compared after warmup only, due to a pre-existing sentinel bug in
    RollingQuantile during warmup."""
    rng = np.random.default_rng(w + 100)
    x = rng.standard_normal(80)
    ours = RollingIqr(w)(x)
    ref = RollingQuantile(w, 0.75)(x) - RollingQuantile(w, 0.25)(x)
    np.testing.assert_allclose(ours[w - 1:], ref[w - 1:], atol=1e-12)


def test_strict_warmup_is_nan():
    out = RollingIqr(7)(np.random.default_rng(0).standard_normal(20))
    assert np.all(np.isnan(out[:6]))
    assert np.all(np.isfinite(out[6:]))


def test_iqr_non_negative():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(100)
    out = RollingIqr(7)(x)
    finite = out[~np.isnan(out)]
    assert np.all(finite >= 0)


def test_constant_input_is_zero():
    out = RollingIqr(5)(np.full(20, 3.7))
    np.testing.assert_array_equal(np.isnan(out[:4]), [True] * 4)
    np.testing.assert_array_equal(out[4:], 0.0)


def test_uniform_distribution_roughly_half_of_range():
    """For uniform U(0, 1) samples, IQR converges to roughly 0.5."""
    rng = np.random.default_rng(2)
    x = rng.uniform(0.0, 1.0, size=2000)
    out = RollingIqr(200)(x)
    finite = out[~np.isnan(out)]
    # Sample IQR converges to 0.5; allow generous tolerance.
    assert 0.4 < np.mean(finite) < 0.6


def test_scalar_loop_matches_array():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(40)
    w = 7
    obj = RollingIqr(w)
    streamed = np.array([obj(v) for v in x])
    np.testing.assert_allclose(streamed, RollingIqr(w)(x),
                               equal_nan=True, atol=1e-12)


def test_2d_per_column_independence():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((30, 3))
    w = 6
    out_2d = RollingIqr(w)(X)
    for k in range(X.shape[1]):
        np.testing.assert_allclose(
            out_2d[:, k], RollingIqr(w)(X[:, k].copy()),
            equal_nan=True, atol=1e-12,
        )
