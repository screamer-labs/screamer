"""Dtype-coercion parity on the ndarray input path.

nanobind's generic `nb::ndarray<>` does not coerce dtypes the way pybind11's
`py::array_t<double, forcecast>` used to. `detail::load_elem`
(include/screamer/common/base.h) restores that coercion by hand for every
dtype numpy commonly produces, including bool, float16 and longdouble. These
tests pin that an operator sees the same values, and therefore produces the
same output, regardless of which numeric dtype the input array was created
with.
"""

import numpy as np
import pytest

from screamer import RollingMean


def _reference(x):
    f = RollingMean(3)
    return np.asarray(f(np.asarray(x, dtype=np.float64)))


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.float16,
        np.float32,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longdouble,
    ],
)
def test_rolling_mean_dtype_parity(dtype):
    rng = np.random.default_rng(0)
    if dtype is np.bool_:
        base = rng.integers(0, 2, size=20)
    elif np.issubdtype(dtype, np.unsignedinteger):
        # Unsigned dtypes cannot hold negative values; casting -5 to uint8
        # wraps to 251, not a dtype-coercion bug in screamer.
        base = rng.integers(0, 6, size=20)
    else:
        # Keep magnitudes small so every dtype (float16's ~3 significant
        # digits, int8's +/-127 range) represents the same values exactly.
        base = rng.integers(-5, 6, size=20)

    x_ref = base.astype(np.float64)
    x_dtype = base.astype(dtype)

    expected = _reference(x_ref)
    got = np.asarray(RollingMean(3)(x_dtype))

    # float16 loses precision on the way in (~3 significant digits); the
    # comparison stays exact-input since `base` is drawn from small integers
    # that round-trip exactly through half precision, but atol stays loose to
    # document that float16 is not bit-exact in general.
    atol = 1e-3 if dtype is np.float16 else 1e-12
    np.testing.assert_allclose(got, expected, atol=atol)


def test_rolling_mean_bool_dtype_values():
    # True/False -> 1.0/0.0, matching the old pybind11 forcecast behavior.
    x = np.array([True, False, True, True, False], dtype=bool)
    got = np.asarray(RollingMean(2)(x))
    expected = np.asarray(RollingMean(2)(np.array([1.0, 0.0, 1.0, 1.0, 0.0])))
    np.testing.assert_array_equal(got, expected)


def test_rolling_mean_float16_dtype():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16)
    got = np.asarray(RollingMean(2)(x))
    expected = np.asarray(RollingMean(2)(np.array([1.0, 2.0, 3.0, 4.0, 5.0])))
    np.testing.assert_allclose(got, expected, atol=1e-3)


def test_rolling_mean_longdouble_dtype():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.longdouble)
    got = np.asarray(RollingMean(2)(x))
    expected = np.asarray(RollingMean(2)(np.array([1.0, 2.0, 3.0, 4.0, 5.0])))
    np.testing.assert_allclose(got, expected, atol=1e-12)
