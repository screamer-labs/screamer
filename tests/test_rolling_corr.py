"""
Tests for RollingCorr (Plan D: first real FunctorBase<_, 2, 1> indicator).

The standard auto-discovery harness in tests/param_cases.py only knows how
to feed a single 1D array per class, so RollingCorr is excluded from it
(see _ROLLING_AUTO_EXCLUDE). These tests cover RollingCorr directly:

  * the multi-input dispatcher (scalar args, two parallel arrays, list of
    pairs, two parallel iterators)
  * Pearson correlation correctness against numpy on aligned windows
  * pandas rolling.corr() parity for the steady-state samples
  * NaN warmup semantics
  * idempotency / state isolation between calls
  * constant-input edge case (zero variance -> NaN)
"""
import math
import numpy as np
import pandas as pd
import pytest

from screamer import RollingCorr


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------

def test_identical_inputs_give_correlation_one():
    x = np.arange(20, dtype=float) + 0.1
    out = RollingCorr(window_size=5)(x, x)
    # First 4 samples are warmup -> NaN. Rest are exactly 1.
    assert np.all(np.isnan(out[:4]))
    np.testing.assert_allclose(out[4:], 1.0, atol=1e-12)


def test_anti_correlated_inputs_give_minus_one():
    x = np.arange(20, dtype=float)
    y = -x + 100.0  # affine transform with negative slope
    out = RollingCorr(window_size=5)(x, y)
    np.testing.assert_allclose(out[4:], -1.0, atol=1e-12)


def test_constant_input_returns_nan():
    """One series with zero variance -> correlation undefined."""
    x = np.arange(10, dtype=float)
    y = np.full(10, 7.0)  # constant
    out = RollingCorr(window_size=5)(x, y)
    # Whole non-warmup range is NaN because var(y)=0.
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("window", [3, 5, 10, 20])
def test_steady_state_matches_numpy_corrcoef(window):
    rng = np.random.default_rng(window)
    n = 80
    x = rng.standard_normal(n)
    y = 0.5 * x + rng.standard_normal(n)
    out = RollingCorr(window_size=window)(x, y)

    # For each post-warmup index i, the value should match numpy's
    # Pearson correlation over x[i-window+1 : i+1] vs y[i-window+1 : i+1].
    for i in range(window - 1, n):
        ref = np.corrcoef(x[i - window + 1 : i + 1],
                          y[i - window + 1 : i + 1])[0, 1]
        assert math.isclose(out[i], ref, abs_tol=1e-10), (
            f"i={i} window={window} got {out[i]} expected {ref}"
        )


@pytest.mark.parametrize("window", [3, 5, 10])
def test_matches_pandas_rolling_corr(window):
    """Pandas is the canonical reference; this is an end-to-end parity check."""
    rng = np.random.default_rng(window + 100)
    x = rng.standard_normal(120)
    y = rng.standard_normal(120)
    out = RollingCorr(window_size=window)(x, y)
    ref = pd.Series(x).rolling(window).corr(pd.Series(y)).to_numpy()

    # Both should produce NaN for the first window-1 samples and matching
    # values afterwards. Pandas may produce -0.0 vs 0.0 or similar minor
    # differences -- 1e-10 is a comfortable tolerance.
    np.testing.assert_allclose(out, ref, equal_nan=True, atol=1e-10)


# ---------------------------------------------------------------------------
# Dispatcher coverage (verifies the FunctorBase<_, 2, 1> input pathways)
# ---------------------------------------------------------------------------

def test_two_scalar_args():
    """Calling with two positional scalars returns a single number."""
    rc = RollingCorr(window_size=3)
    # First call: 1 sample -> NaN.
    r0 = rc(1.0, 2.0)
    assert math.isnan(r0)
    r1 = rc(2.0, 4.0)  # still warming up under strict policy
    assert math.isnan(r1)
    r2 = rc(3.0, 6.0)  # 3 samples now -- y = 2x exactly -> corr = 1
    assert math.isclose(r2, 1.0, abs_tol=1e-12)


def test_list_of_pairs():
    """Streaming format: list of (x, y) tuples."""
    pairs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]
    rc = RollingCorr(window_size=3)
    out = rc(pairs)
    # First two are NaN (warmup), last three are 1.0.
    assert isinstance(out, list)
    assert math.isnan(out[0]) and math.isnan(out[1])
    for v in out[2:]:
        assert math.isclose(v, 1.0, abs_tol=1e-12)


def test_two_iterators():
    """Two parallel iterables (generators) must produce a list of doubles."""
    rc = RollingCorr(window_size=3)
    x_iter = iter([1.0, 2.0, 3.0, 4.0, 5.0])
    y_iter = iter([2.0, 4.0, 6.0, 8.0, 10.0])
    out = rc(x_iter, y_iter)
    assert isinstance(out, list)
    assert math.isnan(out[0]) and math.isnan(out[1])
    for v in out[2:]:
        assert math.isclose(v, 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def test_two_calls_on_same_instance_yield_identical_results():
    """The array path should reset state at start and end of each call."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(30)
    y = rng.standard_normal(30)
    rc = RollingCorr(window_size=5)
    out1 = rc(x, y)
    out2 = rc(x, y)
    np.testing.assert_array_equal(out1, out2)


def test_explicit_reset_returns_to_warmup():
    rc = RollingCorr(window_size=3)
    rc(1.0, 2.0)
    rc(2.0, 4.0)
    rc(3.0, 6.0)
    # After 3 samples we're past warmup; next call returns a real number.
    assert not math.isnan(rc(4.0, 8.0))
    rc.reset()
    # After reset we're back in warmup.
    assert math.isnan(rc(5.0, 10.0))


# ---------------------------------------------------------------------------
# Multi-dimensional inputs (axis 0 = time, other axes = parallel paired streams)
# ---------------------------------------------------------------------------

def test_2d_inputs_pair_columnwise():
    """RollingCorr(X, Y) on (T, K) arrays must equal running RollingCorr
    on each column pair independently."""
    rng = np.random.default_rng(0)
    T, K, window = 50, 4, 10
    X = rng.standard_normal((T, K))
    Y = 0.5 * X + 0.5 * rng.standard_normal((T, K))

    out_2d = RollingCorr(window)(X, Y)
    assert out_2d.shape == (T, K)

    for k in range(K):
        ref = RollingCorr(window)(X[:, k].copy(), Y[:, k].copy())
        np.testing.assert_array_equal(
            out_2d[:, k], ref,
            err_msg=f"column {k} of 2D output differs from per-column 1D"
        )


def test_3d_inputs_pair_per_inner_index():
    """N-D paired arrays: every (j, k) inner index is an independent stream."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 3, 2))
    Y = 0.7 * X + 0.3 * rng.standard_normal((30, 3, 2))
    out = RollingCorr(window_size=5)(X, Y)
    assert out.shape == (30, 3, 2)
    # Spot-check one inner cell
    ref = RollingCorr(5)(X[:, 1, 1].copy(), Y[:, 1, 1].copy())
    np.testing.assert_array_equal(out[:, 1, 1], ref)


def test_2d_strided_view_matches_contig_copy():
    rng = np.random.default_rng(2)
    big_x = rng.standard_normal((100, 12))
    big_y = 0.5 * big_x + 0.5 * rng.standard_normal((100, 12))
    vx, vy = big_x[::2, ::3], big_y[::2, ::3]
    assert not vx.flags.c_contiguous

    rc = RollingCorr(window_size=10)
    out_view = rc(vx, vy)
    out_copy = rc(vx.copy(), vy.copy())
    np.testing.assert_array_equal(out_view, out_copy)


def test_shape_mismatch_raises():
    with pytest.raises(TypeError):
        RollingCorr(5)(np.zeros(10), np.zeros(20))


def test_ndim_mismatch_raises():
    with pytest.raises(TypeError):
        RollingCorr(5)(np.zeros(10), np.zeros((10, 3)))


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_window_size_must_be_at_least_two():
    with pytest.raises(ValueError):
        RollingCorr(window_size=1)
    with pytest.raises(ValueError):
        RollingCorr(window_size=0)
    # 2 is fine.
    rc = RollingCorr(window_size=2)
    rc(1.0, 1.0)
