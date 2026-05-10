"""
Tests for Linear2: two-input affine combination, f(x, y) = a*x + b*y + c.

Stateless 2->1 functor. The tests focus on three things:

  1. Per-element correctness of the formula across input modes.
  2. Common idioms it enables (signed difference, weighted blend,
     positive excess) combined with Sign / Relu.
  3. Dispatcher paths (scalar, list-of-pairs, parallel arrays, 2D).
"""
import numpy as np
import pytest

from screamer import Linear2, Sign, Relu, Sigmoid


# ---------------------------------------------------------------------------
# Basic formula
# ---------------------------------------------------------------------------

class TestFormula:

    @pytest.mark.parametrize("a,b,c", [
        (1.0, -1.0, 0.0),    # x - y
        (1.0,  1.0, 0.0),    # x + y
        (0.7,  0.3, 0.0),    # weighted blend
        (2.0, -3.0, 5.0),    # general
        (0.0,  0.0, 1.5),    # constant
    ])
    def test_matches_numpy(self, a, b, c):
        rng = np.random.default_rng(int(a * 100 + b * 10 + c))
        x = rng.standard_normal(50)
        y = rng.standard_normal(50)
        np.testing.assert_allclose(Linear2(a, b, c)(x, y),
                                   a * x + b * y + c, atol=1e-12)

    def test_default_c_is_zero(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(20)
        y = rng.standard_normal(20)
        np.testing.assert_array_equal(Linear2(1.0, -1.0)(x, y),
                                      Linear2(1.0, -1.0, 0.0)(x, y))

    def test_scalar_inputs(self):
        out = Linear2(2.0, 3.0, 1.0)(4.0, 5.0)
        assert out == pytest.approx(2.0 * 4.0 + 3.0 * 5.0 + 1.0)


# ---------------------------------------------------------------------------
# Composition idioms with Sign / Relu / Sigmoid
# ---------------------------------------------------------------------------

class TestComposition:

    def test_sign_of_difference_is_greater_than(self):
        """Sign(x - y) -> +1 if x > y, -1 if x < y, 0 if equal."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 2.0, 1.0])
        out = Sign()(Linear2(1, -1)(x, y))
        np.testing.assert_array_equal(out, [-1.0, 0.0, 1.0])

    def test_relu_of_difference_is_positive_excess(self):
        """Relu(x - y) = max(x - y, 0)."""
        x = np.array([3.0, 1.0, 5.0, 2.0])
        y = np.array([1.0, 3.0, 5.0, 2.0])
        out = Relu()(Linear2(1, -1)(x, y))
        np.testing.assert_array_equal(out, [2.0, 0.0, 0.0, 0.0])

    def test_sigmoid_of_weighted_sum_is_smooth_classifier(self):
        """Sigmoid(a*x + b*y + c) is the canonical logistic mix; verify
        the chain runs and gives sane outputs in [0, 1]."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(40)
        y = rng.standard_normal(40)
        out = Sigmoid()(Linear2(0.5, -0.5, 0.0)(x, y))
        assert np.all(out >= 0.0) and np.all(out <= 1.0)


# ---------------------------------------------------------------------------
# Dispatcher paths
# ---------------------------------------------------------------------------

class TestDispatcher:

    def test_scalar_pair(self):
        assert Linear2(1, -1, 5)(3.0, 1.0) == pytest.approx(7.0)

    def test_list_of_pairs_returns_list(self):
        out = Linear2(1, 1, 0)([(1.0, 2.0), (3.0, 4.0)])
        assert isinstance(out, list)
        assert out == [3.0, 7.0]

    def test_two_parallel_iterables(self):
        out = Linear2(2, -1, 0)(iter([1.0, 2.0, 3.0]), iter([0.5, 1.0, 1.5]))
        assert isinstance(out, list)
        np.testing.assert_allclose(out, [1.5, 3.0, 4.5])

    def test_2d_per_column_independence(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 3))
        Y = rng.standard_normal((30, 3))
        out_2d = Linear2(0.5, -0.5, 1.0)(X, Y)
        for k in range(3):
            np.testing.assert_array_equal(
                out_2d[:, k],
                Linear2(0.5, -0.5, 1.0)(X[:, k].copy(), Y[:, k].copy()),
            )

    def test_shape_mismatch_raises(self):
        with pytest.raises(TypeError):
            Linear2(1, 1, 0)(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# NaN / sanity
# ---------------------------------------------------------------------------

def test_nan_propagates():
    x = np.array([1.0, np.nan, 3.0])
    y = np.array([2.0, 5.0, np.nan])
    out = Linear2(1, 1, 0)(x, y)
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert out[0] == 3.0


def test_reset_is_noop_for_stateless():
    """Linear2 has no state, but reset() should still be safe to call."""
    obj = Linear2(1, 1, 0)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(20)
    y = rng.standard_normal(20)
    first = obj(x, y)
    obj.reset()
    second = obj(x, y)
    np.testing.assert_array_equal(first, second)
