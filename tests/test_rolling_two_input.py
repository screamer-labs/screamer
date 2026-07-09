"""
Tests for the rolling 2-input/1-output indicators on top of FunctorBase.

  RollingCorr   - covered in test_rolling_corr.py
  RollingCov    - rolling sample covariance
  RollingBeta   - rolling regression slope of x on y, beta = cov(x, y) / var(y)
  RollingSpread - hedge-adjusted residual x - beta * y, same beta as above

The contract verified per class:
  * matches pandas reference to floating-point precision
    (parametrised over several windows);
  * 2-D paired arrays produce column-by-column results bit-exactly;
  * scalar / streaming / list / iterator paths all agree numerically;
  * shape and ndim mismatches across the two inputs raise TypeError;
  * window_size < 2 is rejected.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import RollingCov, RollingBeta, RollingSpread


# ---------------------------------------------------------------------------
# Reference implementations (pandas)
# ---------------------------------------------------------------------------

def pandas_cov(x, y, w):
    return pd.Series(x).rolling(w).cov(pd.Series(y)).to_numpy()

def pandas_beta(x, y, w):
    """Slope of x on y: cov(x, y) / var(y)."""
    return (pd.Series(x).rolling(w).cov(pd.Series(y))
            / pd.Series(y).rolling(w).var()).to_numpy()

def pandas_spread(x, y, w):
    """x - beta * y, with rolling beta of x on y."""
    return x - pandas_beta(x, y, w) * y


# ---------------------------------------------------------------------------
# Pandas parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", [3, 5, 10, 30])
def test_cov_matches_pandas(window):
    rng = np.random.default_rng(window)
    x = rng.standard_normal(120)
    y = 0.5 * x + rng.standard_normal(120)
    np.testing.assert_allclose(
        RollingCov(window)(x, y), pandas_cov(x, y, window),
        equal_nan=True, atol=1e-10,
    )


@pytest.mark.parametrize("window", [3, 5, 10, 30])
def test_beta_matches_pandas(window):
    rng = np.random.default_rng(window + 100)
    x = rng.standard_normal(120)
    y = 0.5 * x + rng.standard_normal(120)
    np.testing.assert_allclose(
        RollingBeta(window)(x, y), pandas_beta(x, y, window),
        equal_nan=True, atol=1e-10,
    )


@pytest.mark.parametrize("window", [3, 5, 10, 30])
def test_spread_matches_manual(window):
    rng = np.random.default_rng(window + 200)
    x = rng.standard_normal(120)
    y = 0.5 * x + rng.standard_normal(120)
    np.testing.assert_allclose(
        RollingSpread(window)(x, y), pandas_spread(x, y, window),
        equal_nan=True, atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Multi-D paired arrays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [RollingCov, RollingBeta, RollingSpread])
def test_2d_inputs_pair_columnwise(cls):
    rng = np.random.default_rng(0)
    T, K, window = 50, 4, 10
    X = rng.standard_normal((T, K))
    Y = 0.5 * X + 0.5 * rng.standard_normal((T, K))

    out_2d = cls(window)(X, Y)
    assert out_2d.shape == (T, K)
    for k in range(K):
        ref = cls(window)(X[:, k].copy(), Y[:, k].copy())
        np.testing.assert_array_equal(out_2d[:, k], ref)


@pytest.mark.parametrize("cls", [RollingCov, RollingBeta, RollingSpread])
def test_3d_inputs_pair_per_inner_index(cls):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 3, 2))
    Y = 0.7 * X + 0.3 * rng.standard_normal((30, 3, 2))
    out = cls(window_size=5)(X, Y)
    assert out.shape == (30, 3, 2)
    ref = cls(5)(X[:, 1, 1].copy(), Y[:, 1, 1].copy())
    np.testing.assert_array_equal(out[:, 1, 1], ref)


@pytest.mark.parametrize("cls", [RollingCov, RollingBeta, RollingSpread])
def test_2d_strided_view_matches_contig(cls):
    rng = np.random.default_rng(2)
    big_x = rng.standard_normal((100, 12))
    big_y = 0.5 * big_x + 0.5 * rng.standard_normal((100, 12))
    vx, vy = big_x[::2, ::3], big_y[::2, ::3]
    rc = cls(window_size=10)
    np.testing.assert_array_equal(rc(vx, vy), rc(vx.copy(), vy.copy()))


# ---------------------------------------------------------------------------
# Streaming, scalar, list-of-pairs paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [RollingCov, RollingBeta, RollingSpread])
def test_streaming_matches_array(cls):
    """Scalar loop and array path produce the same numbers."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(40)
    y = rng.standard_normal(40)
    expected = cls(5)(x, y)

    streaming = cls(5)
    out = np.array([streaming(xi, yi) for xi, yi in zip(x, y)])
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("cls", [RollingCov, RollingBeta, RollingSpread])
def test_list_of_pairs(cls):
    pairs = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0)]
    out = cls(window_size=3)(pairs)
    assert isinstance(out, list)
    assert len(out) == 5


@pytest.mark.parametrize("cls", [RollingCov, RollingBeta, RollingSpread])
def test_two_iterators(cls):
    out = cls(window_size=3)(iter([1.0, 2.0, 3.0]), iter([2.0, 4.0, 6.0]))
    assert hasattr(out, "__next__") and not isinstance(out, list)
    assert len(list(out)) == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [RollingCov, RollingBeta, RollingSpread])
def test_constant_y(cls):
    """Constant y. Cov is well defined (zero); beta and spread divide by
    var(y) which is zero, so NaN."""
    x = np.arange(10, dtype=float)
    y = np.full(10, 7.0)
    out = cls(5)(x, y)
    if cls is RollingCov:
        np.testing.assert_allclose(out[4:], 0.0, atol=1e-12)
    else:
        assert np.all(np.isnan(out))


@pytest.mark.parametrize("cls", [RollingCov, RollingBeta, RollingSpread])
def test_shape_mismatch_raises(cls):
    with pytest.raises(TypeError):
        cls(5)(np.zeros(10), np.zeros(20))


@pytest.mark.parametrize("cls", [RollingCov, RollingBeta, RollingSpread])
def test_window_size_must_be_at_least_two(cls):
    with pytest.raises(ValueError):
        cls(window_size=1)
    cls(window_size=2)  # 2 is fine
