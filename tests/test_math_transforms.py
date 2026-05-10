"""
Tests for the simple element-wise math transforms.

These wrap a single C++ math function via the `Transform<...>` template,
so the only contract worth pinning is per-element parity with the
equivalent numpy ufunc. The polymorphic input dispatch (scalar / array /
iterator / 2-D) is exercised by the broader test_io_size.py and friends
across all `Transform`-based classes.
"""
import numpy as np
import pytest

from screamer import Floor, Ceil, Square, Cube, Sin, Cos, Atan, Round, Asin, Acos, Identity


# (screamer class, numpy reference) pairs
PAIRS = [
    (Floor,    np.floor),
    (Ceil,     np.ceil),
    (Round,    np.round),
    (Square,   lambda x: x * x),
    (Cube,     lambda x: x * x * x),
    (Sin,      np.sin),
    (Cos,      np.cos),
    (Atan,     np.arctan),
    (Identity, lambda x: x.copy() if hasattr(x, "copy") else x),
]


# Inverse trig is bounded to [-1, 1]. Test it on a clipped input where
# numpy doesn't emit "invalid value" warnings, then test out-of-range
# behaviour separately below.
INVERSE_TRIG_PAIRS = [
    (Asin, np.arcsin),
    (Acos, np.arccos),
]


@pytest.mark.parametrize("cls,reference", PAIRS, ids=[c.__name__ for c, _ in PAIRS])
def test_array_matches_numpy(cls, reference):
    rng = np.random.default_rng(0)
    # Cover negatives, zeros, small magnitudes, large magnitudes,
    # values near integer boundaries (relevant for floor/ceil).
    x = np.concatenate([
        rng.standard_normal(50) * 5.0,
        np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
    ])
    out = cls()(x)
    np.testing.assert_allclose(out, reference(x), atol=1e-12)


@pytest.mark.parametrize("cls,reference", PAIRS, ids=[c.__name__ for c, _ in PAIRS])
def test_scalar_matches_numpy(cls, reference):
    obj = cls()
    for v in [-1.7, -0.5, 0.0, 0.5, 1.7, 3.14159]:
        assert obj(v) == pytest.approx(reference(v), abs=1e-12)


@pytest.mark.parametrize("cls,reference", INVERSE_TRIG_PAIRS,
                         ids=[c.__name__ for c, _ in INVERSE_TRIG_PAIRS])
def test_inverse_trig_in_range_matches_numpy(cls, reference):
    x = np.linspace(-1.0, 1.0, 21)
    np.testing.assert_allclose(cls()(x), reference(x), atol=1e-12)


@pytest.mark.parametrize("cls", [Asin, Acos])
def test_inverse_trig_out_of_range_returns_nan(cls):
    """numpy.arcsin / arccos return NaN outside [-1, 1]; match that."""
    x = np.array([-2.0, -1.5, -1.0, 1.0, 1.5, 2.0])
    out = cls()(x)
    expected_nan = np.array([True, True, False, False, True, True])
    np.testing.assert_array_equal(np.isnan(out), expected_nan)


def test_round_uses_bankers_rounding():
    """Round must match numpy.round, which uses round-half-to-even
    (banker's rounding), not round-half-away-from-zero."""
    halves = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    expected = np.array([-2.0, -2.0,  0.0, 0.0, 2.0, 2.0, 4.0, 4.0])
    np.testing.assert_array_equal(Round()(halves), expected)


def test_identity_preserves_nan():
    x = np.array([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf])
    out = Identity()(x)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(x))
    finite = ~np.isnan(x) & np.isfinite(x)
    np.testing.assert_array_equal(out[finite], x[finite])


@pytest.mark.parametrize("cls", [Floor, Ceil, Round, Square, Cube,
                                  Sin, Cos, Atan, Asin, Acos, Identity])
def test_2d_array_per_column_independence(cls):
    """Element-wise transforms have no state, so 2-D and column-by-column
    must agree exactly."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 4))
    out_2d = cls()(X)
    assert out_2d.shape == X.shape
    for k in range(X.shape[1]):
        np.testing.assert_array_equal(out_2d[:, k], cls()(X[:, k].copy()))


def test_square_equals_power_2():
    """Square is a faster equivalent of Power(2). Verify they agree."""
    from screamer import Power
    rng = np.random.default_rng(2)
    x = rng.standard_normal(50) * 3.0
    np.testing.assert_allclose(Square()(x), Power(2.0)(x), atol=1e-12)


def test_cube_equals_power_3():
    from screamer import Power
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50) * 3.0
    np.testing.assert_allclose(Cube()(x), Power(3.0)(x), atol=1e-12)


def test_sin_cos_pythagorean_identity():
    """sin(x)^2 + cos(x)^2 == 1 for every x."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(100) * 10.0
    s = Sin()(x)
    c = Cos()(x)
    np.testing.assert_allclose(s ** 2 + c ** 2, np.ones_like(x), atol=1e-12)


def test_atan_inverse_of_tan_in_principal_range():
    """For x in (-pi/2, pi/2), atan(tan(x)) == x."""
    x = np.linspace(-1.5, 1.5, 50)  # safely inside (-pi/2, pi/2)
    np.testing.assert_allclose(Atan()(np.tan(x)), x, atol=1e-12)
