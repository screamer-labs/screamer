"""
Tests for Diff2: the second-order finite difference.

  y[t] = x[t] - 2*x[t-1] + x[t-2]

This is the discrete analogue of the second derivative. Distinct from
Diff(2), which is the lag-2 first difference x[t] - x[t-2].
"""
import numpy as np
import pytest

from screamer import Diff, Diff2


def test_matches_explicit_formula():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(60)
    out = Diff2()(x)
    expected = np.full_like(x, np.nan)
    expected[2:] = x[2:] - 2.0 * x[1:-1] + x[:-2]
    np.testing.assert_allclose(out, expected, equal_nan=True, atol=1e-12)


def test_matches_diff_of_diff():
    """Two Diff(1) operations chained must equal Diff2."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(50)
    chained = Diff(1)(Diff(1)(x))
    np.testing.assert_allclose(Diff2()(x), chained, equal_nan=True, atol=1e-12)


def test_strict_warmup_is_two_nans():
    out = Diff2()(np.array([1.0, 4.0, 9.0, 16.0, 25.0]))
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert np.all(np.isfinite(out[2:]))


def test_quadratic_input_gives_constant_second_diff():
    """If x[t] = a*t^2 + b*t + c, the second-order difference is 2a."""
    a = 3.0
    t = np.arange(20, dtype=float)
    x = a * t * t + 1.5 * t - 7.0
    out = Diff2()(x)
    np.testing.assert_allclose(out[2:], 2.0 * a, atol=1e-12)


def test_linear_input_gives_zero_second_diff():
    """For a linear input the discrete second derivative is zero."""
    t = np.arange(15, dtype=float)
    out = Diff2()(2.5 * t + 0.7)
    np.testing.assert_allclose(out[2:], 0.0, atol=1e-12)


def test_distinct_from_diff_lag_2():
    """Diff2 is NOT the same as Diff(2). Confirm they disagree on a
    series where the difference matters."""
    x = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # impulse
    diff2_out = Diff2()(x)
    diff_lag2 = Diff(2)(x)
    # At t=4 the two formulas diverge: Diff2 = 0 - 2*0 + 1 = 1,
    # Diff(2) = 0 - 0 = 0.
    assert diff2_out[4] != diff_lag2[4]


def test_scalar_loop_matches_array():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(30)
    obj = Diff2()
    streamed = np.array([obj(v) for v in x])
    np.testing.assert_allclose(streamed, Diff2()(x), equal_nan=True, atol=1e-12)


def test_reset_clears_history():
    d = Diff2()
    for v in [1.0, 2.0, 3.0]:
        d(v)
    assert d(4.0) == pytest.approx(0.0)  # constant 2nd diff for arithmetic seq
    d.reset()
    out = [d(v) for v in [1.0, 2.0, 3.0]]
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == pytest.approx(0.0)


def test_2d_per_column_independence():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 3))
    out_2d = Diff2()(X)
    for k in range(X.shape[1]):
        np.testing.assert_allclose(
            out_2d[:, k], Diff2()(X[:, k].copy()),
            equal_nan=True, atol=1e-12,
        )
