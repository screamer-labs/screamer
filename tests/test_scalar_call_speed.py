"""A scalar call to a multi-input operator takes a fast path (decision D1).

Exactly N Python floats or ints, or NumPy number scalars, call the kernel
directly. Every other argument falls through to the full dispatch, so the
fast path never changes a result or an error.
"""
import timeit

import numpy as np
import pytest

import screamer as s
from screamer.dag import Input, is_node


def _per_call(f, n=100_000):
    return min(timeit.repeat(f, number=n, repeat=5)) / n


def test_two_input_scalar_call_is_close_to_one_input():
    one = s.RollingMean(20)
    two = s.RollingCorr(20)
    t1 = _per_call(lambda: one(1.0))
    t2 = _per_call(lambda: two(1.0, 2.0))
    assert t2 < 0.15e-6 or t2 < 3 * t1, (t1, t2)


def test_numpy_scalars_take_the_fast_path():
    two = s.RollingCorr(20)
    a, b = np.float64(1.0), np.float64(2.0)
    assert _per_call(lambda: two(a, b)) < 0.3e-6
    c, d = np.float32(1.0), np.int64(2)
    assert _per_call(lambda: two(c, d)) < 0.3e-6


def test_fast_path_values_match_slow_path():
    xs, ys = np.random.default_rng(1).random((2, 200))
    a, b = s.RollingCorr(5), s.RollingCorr(5)
    np.testing.assert_array_equal(
        [a(float(x), float(y)) for x, y in zip(xs, ys)], b(xs, ys))


@pytest.mark.parametrize("cast", [
    float, int, np.float64, np.float32, np.float16, np.int64, np.int32,
    np.int8, np.uint16, np.uint64,
])
def test_numpy_and_python_scalar_kinds_match_batch(cast):
    xs = [cast(v) for v in (1, 3, 2, 7, 5, 4, 9, 8)]
    ys = [cast(v) for v in (2, 1, 4, 3, 6, 8, 7, 9)]
    a, b = s.RollingCorr(3), s.RollingCorr(3)
    ref = b(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    np.testing.assert_array_equal([a(x, y) for x, y in zip(xs, ys)], ref)


def test_multi_output_scalar_calls_match_batch():
    xs = np.random.default_rng(2).random(50)
    a, b = s.BollingerBands(5), s.BollingerBands(5)
    got = np.array([a(float(x)) for x in xs])
    np.testing.assert_array_equal(got, b(xs))


def test_mixed_scalar_kinds_in_one_call():
    a, b = s.Add(), s.Add()
    assert a(np.float32(1.5), 2) == b(1.5, 2.0) == 3.5
    assert a(np.int64(3), np.float64(0.25)) == 3.25


def test_non_exact_numbers_keep_their_old_behavior():
    class F(float):
        pass

    add = s.Add()
    assert add(True, 2.0) == 3.0           # bool is not an exact int
    assert add(F(1.5), 2.0) == 3.5         # a float subclass
    assert add(np.array(1.0), 2.0) == 3.0  # a 0-d array reads as a scalar
    with pytest.raises(TypeError):
        add(10 ** 400, 1.0)                # an int too large for a double
    with pytest.raises(TypeError):
        add(1.0)                           # wrong count
    with pytest.raises(TypeError):
        add(1.0, "2")


def test_pipeline_node_arguments_still_build_a_node():
    a, b = Input("a"), Input("b")
    for n in (s.RollingCorr(20)(a, b), s.BollingerBands(5)(a)):
        assert is_node(n)
