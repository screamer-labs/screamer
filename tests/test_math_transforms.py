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

from screamer import Floor, Ceil, Square, Cube, Sin, Cos, Atan


# (screamer class, numpy reference) pairs
PAIRS = [
    (Floor,  np.floor),
    (Ceil,   np.ceil),
    (Square, lambda x: x * x),
    (Cube,   lambda x: x * x * x),
    (Sin,    np.sin),
    (Cos,    np.cos),
    (Atan,   np.arctan),
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


@pytest.mark.parametrize("cls", [Floor, Ceil, Square, Cube, Sin, Cos, Atan])
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
