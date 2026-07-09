"""
Tests for the 2D coordinate / vector batch and the 2->2 dispatcher (Plan E).

  Hypot(x, y), Atan2(y, x): 2->1 stateless math.
  Cart2Polar(x, y), Polar2Cart(r, theta): 2->2 stateless conversions.

The polar pair is the first 2->2 consumer in the library and exercises
every path of the new N->M dispatcher. Hypot and Atan2 are 2->1 primitives
that double as validation references for Cart2Polar's two outputs.
"""
import numpy as np
import pytest

from screamer import Hypot, Atan2, Cart2Polar, Polar2Cart


# ---------------------------------------------------------------------------
# 2->1 primitives
# ---------------------------------------------------------------------------

class TestHypot:

    def test_scalar_3_4_5(self):
        assert Hypot()(3.0, 4.0) == 5.0

    def test_arrays_match_numpy(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(50)
        y = rng.standard_normal(50)
        np.testing.assert_allclose(Hypot()(x, y), np.hypot(x, y), atol=1e-12)

    def test_scalar_matches_numpy(self):
        for x, y in [(0.0, 0.0), (1e200, 1e200), (-3.0, 4.0)]:
            assert Hypot()(x, y) == pytest.approx(np.hypot(x, y), abs=1e-12)


class TestAtan2:

    def test_scalar_quadrants(self):
        # numpy.arctan2(y, x) — y first, x second, range (-pi, pi].
        assert Atan2()(0.0,  1.0) == pytest.approx(0.0)
        assert Atan2()(1.0,  0.0) == pytest.approx(np.pi / 2)
        assert Atan2()(0.0, -1.0) == pytest.approx(np.pi)
        assert Atan2()(-1.0, 0.0) == pytest.approx(-np.pi / 2)

    def test_arrays_match_numpy(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal(50)
        x = rng.standard_normal(50)
        np.testing.assert_allclose(Atan2()(y, x), np.arctan2(y, x), atol=1e-12)


# ---------------------------------------------------------------------------
# 2->2 polar pair
# ---------------------------------------------------------------------------

class TestCart2Polar:

    def test_scalar_returns_tuple(self):
        out = Cart2Polar()(3.0, 4.0)
        assert isinstance(out, tuple)
        assert out[0] == pytest.approx(5.0)
        assert out[1] == pytest.approx(np.arctan2(4.0, 3.0))

    def test_1d_arrays_shape_and_values(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(40)
        y = rng.standard_normal(40)
        out = Cart2Polar()(x, y)
        assert out.shape == (40, 2)
        np.testing.assert_allclose(out[:, 0], np.hypot(x, y), atol=1e-12)
        np.testing.assert_allclose(out[:, 1], np.arctan2(y, x), atol=1e-12)

    def test_outputs_match_2x_2to1(self):
        """Cart2Polar's two outputs should agree with Hypot and Atan2 element-wise."""
        rng = np.random.default_rng(3)
        x, y = rng.standard_normal(30), rng.standard_normal(30)
        polar = Cart2Polar()(x, y)
        np.testing.assert_array_equal(polar[:, 0], Hypot()(x, y))
        np.testing.assert_array_equal(polar[:, 1], Atan2()(y, x))

    def test_list_of_pairs(self):
        out = Cart2Polar()([(3.0, 4.0), (0.0, 1.0), (-1.0, 0.0)])
        assert isinstance(out, list) and len(out) == 3
        assert out[0] == pytest.approx((5.0, np.arctan2(4.0, 3.0)))
        assert out[1] == pytest.approx((1.0, np.pi / 2))
        assert out[2] == pytest.approx((1.0, np.pi))

    def test_iterables(self):
        out = Cart2Polar()(iter([3.0, 0.0]), iter([4.0, 1.0]))
        assert hasattr(out, "__next__") and not isinstance(out, list)
        rows = list(out)
        assert len(rows) == 2
        assert rows[0] == pytest.approx((5.0, np.arctan2(4.0, 3.0)))

    def test_2d_per_column_pairing(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((20, 3))
        Y = rng.standard_normal((20, 3))
        out = Cart2Polar()(X, Y)
        assert out.shape == (20, 3, 2)
        for k in range(3):
            np.testing.assert_array_equal(
                out[:, k, :], Cart2Polar()(X[:, k].copy(), Y[:, k].copy()),
            )


class TestPolar2Cart:

    def test_scalar_returns_tuple(self):
        out = Polar2Cart()(5.0, np.arctan2(4.0, 3.0))
        assert out[0] == pytest.approx(3.0)
        assert out[1] == pytest.approx(4.0)

    def test_1d_arrays(self):
        rng = np.random.default_rng(5)
        r = np.abs(rng.standard_normal(30))
        theta = rng.uniform(-np.pi, np.pi, 30)
        out = Polar2Cart()(r, theta)
        np.testing.assert_allclose(out[:, 0], r * np.cos(theta), atol=1e-12)
        np.testing.assert_allclose(out[:, 1], r * np.sin(theta), atol=1e-12)


class TestRoundtrip:

    def test_cart_polar_cart_identity(self):
        """Polar2Cart(Cart2Polar(x, y)) == (x, y) up to floating point."""
        rng = np.random.default_rng(6)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        polar = Cart2Polar()(x, y)
        back = Polar2Cart()(polar[:, 0], polar[:, 1])
        np.testing.assert_allclose(back[:, 0], x, atol=1e-12)
        np.testing.assert_allclose(back[:, 1], y, atol=1e-12)

    def test_polar_cart_polar_identity(self):
        """Going the other way around also commutes (with theta in (-pi, pi])."""
        rng = np.random.default_rng(7)
        r = np.abs(rng.standard_normal(100)) + 1e-6   # avoid r==0 (theta undefined)
        theta = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3, 100)
        cart = Polar2Cart()(r, theta)
        back = Cart2Polar()(cart[:, 0], cart[:, 1])
        np.testing.assert_allclose(back[:, 0], r, atol=1e-12)
        np.testing.assert_allclose(back[:, 1], theta, atol=1e-12)


# ---------------------------------------------------------------------------
# Plan E dispatcher: shape validation, error paths
# ---------------------------------------------------------------------------

class TestDispatcherErrors:

    def test_shape_mismatch_raises(self):
        with pytest.raises(TypeError):
            Cart2Polar()(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_dim_mismatch_raises(self):
        with pytest.raises(TypeError):
            Cart2Polar()(np.array([1.0, 2.0]),
                          np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_mixed_kinds_raises(self):
        with pytest.raises(TypeError):
            Cart2Polar()(1.0, np.array([1.0, 2.0]))

    def test_nan_propagates(self):
        out = Cart2Polar()(np.array([1.0, np.nan, 3.0]),
                           np.array([2.0, 5.0, np.nan]))
        # NaN in either coordinate produces NaN in both outputs.
        assert np.isnan(out[1, 0]) and np.isnan(out[1, 1])
        assert np.isnan(out[2, 0]) and np.isnan(out[2, 1])
        # Finite row is finite.
        assert np.all(np.isfinite(out[0, :]))


# ---------------------------------------------------------------------------
# Stateful 2->2 sanity check using MyFunctor22 (validates per-column reset)
# ---------------------------------------------------------------------------

def test_stateful_2x2_resets_between_columns():
    """The dispatcher must reset() between columns of a 2D batch, otherwise
    state leaks. MyFunctor22 accumulates a sum_state; if reset is honoured
    each column starts from zero."""
    from screamer import MyFunctor22
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Y = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    # call returns (a-b, sum_state); sum_state accumulates a+b.
    out = MyFunctor22()(X, Y)
    # Column 0: x=[1,3,5], y=[1,1,1] -> sum_state = 2, 6, 12
    np.testing.assert_array_equal(out[:, 0, 1], [2.0, 6.0, 12.0])
    # Column 1: x=[2,4,6], y=[2,2,2] -> sum_state = 4, 10, 18
    # If reset wasn't called between columns, this would start at 12 and
    # produce [16, 22, 30] instead.
    np.testing.assert_array_equal(out[:, 1, 1], [4.0, 10.0, 18.0])
