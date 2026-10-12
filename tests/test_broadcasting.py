"""Multi-input operators broadcast over assets; a size-1 field axis is consumed."""
import numpy as np
import pytest

import screamer as s

rng = np.random.default_rng(0)
T, K = 200, 3
X = rng.normal(size=(T, K)).cumsum(axis=0) + 100
y = rng.normal(size=T).cumsum() + 100


def test_one_d_broadcasts_against_assets():
    got = s.RollingCorr(20)(X, y)
    want = s.RollingCorr(20)(X, np.repeat(y[:, None], K, axis=1))
    np.testing.assert_array_equal(got, want)
    assert got.shape == (T, K)


def test_column_broadcasts_against_assets():
    np.testing.assert_array_equal(s.Add()(X, y[:, None]), X + y[:, None])


def test_multi_output_engine_with_one_price():
    target = np.sign(rng.normal(size=(T, K)))
    got = s.BacktestPriceTarget()(target, y)
    want = s.BacktestPriceTarget()(target, np.repeat(y[:, None], K, axis=1))
    assert got.shape == (T, K, 4)
    np.testing.assert_array_equal(got, want)


def test_mismatched_assets_raise():
    with pytest.raises(ValueError, match="do not broadcast"):
        s.Add()(np.zeros((T, 3)), np.zeros((T, 4)))


def test_mismatched_time_raises():
    with pytest.raises(ValueError, match="same length along time"):
        s.Add()(np.zeros(T), np.zeros(T - 1))


def test_packed_three_d():
    packed = np.stack([X, X + 1.0], axis=-1)                    # (T, K, 2)
    np.testing.assert_array_equal(s.Sub()(packed), np.full((T, K), -1.0))


def test_unit_field_consumed_one_input():
    np.testing.assert_array_equal(s.RollingMean(5)(X[:, :, None]), s.RollingMean(5)(X))


def test_unit_field_consumed_multi_output():
    assert s.BollingerBands(5)(X[:, :, None]).shape == (T, K, 3)


def test_unit_field_consumed_functorbase_one_input():
    np.testing.assert_array_equal(s.IsNan()(X[:, :, None]), s.IsNan()(X))


def test_unit_field_consumed_multi_input():
    np.testing.assert_array_equal(s.Add()(X[:, :, None], y), s.Add()(X, y))


def test_t_by_1_is_one_asset():
    assert s.RollingMean(5)(X[:, :1]).shape == (T, 1)
    assert s.BollingerBands(5)(X[:, :1]).shape == (T, 1, 3)


def test_broadcast_reads_any_dtype_and_strides():
    Xi = np.round(X).astype(np.int64)
    view = np.asfortranarray(Xi.astype(np.float64))
    want = s.RollingCorr(20)(Xi.astype(np.float64), y)
    np.testing.assert_array_equal(s.RollingCorr(20)(Xi, y), want)
    np.testing.assert_array_equal(s.RollingCorr(20)(view, y), want)
    np.testing.assert_array_equal(s.RollingCorr(20)(Xi.T.copy().T, y), want)


def test_empty_broadcast_shapes():
    assert s.Add()(np.zeros((0, 3)), np.zeros(0)).shape == (0, 3)
    assert s.Add()(np.zeros((T, 0)), np.zeros(T)).shape == (T, 0)
    assert s.BacktestPriceTarget()(np.zeros((0, 2)), np.zeros(0)).shape == (0, 2, 4)


def test_generator_into_one_input_functor_is_lazy():
    values = [1.0, np.nan, 3.0, np.inf]
    out = s.IsNan()(v for v in values)
    assert hasattr(out, "__next__") and not isinstance(out, list)
    np.testing.assert_array_equal(list(out), s.IsNan()(np.array(values)))


# Long strided columns run in tiles of 8 gathered columns (column_driver.h).
# Every tiling case must give each column's 1-D result bit for bit: a partial
# last tile, a conflict stride (K = 64), a broadcast (T,) input next to (T, K),
# a packed (T, K, 2) input, a ScreamerBase operator, and a multi-output one.
@pytest.mark.parametrize("shape", [(300, 13), (300, 64), (2000, 100), (600, 3, 5)])
@pytest.mark.parametrize("make", [s.IsFinite, s.Add, lambda: s.RollingCorr(20),
                                  lambda: s.RollingMean(20), lambda: s.BollingerBands(20)])
def test_tiled_columns_equal_each_column(shape, make):
    rng = np.random.default_rng(1)
    A = rng.normal(size=shape).cumsum(axis=0) + 100
    B = rng.normal(size=shape).cumsum(axis=0) + 50
    op = make()
    two = op.num_inputs == 2
    got = op(A, B) if two else op(A)
    flat_a = A.reshape(shape[0], -1)
    flat_b = B.reshape(shape[0], -1)
    flat_got = got.reshape(shape[0], flat_a.shape[1], -1)
    for k in range(flat_a.shape[1]):
        want = make()(flat_a[:, k], flat_b[:, k]) if two else make()(flat_a[:, k])
        np.testing.assert_array_equal(flat_got[:, k].reshape(want.shape), want)


@pytest.mark.parametrize("K", [13, 64, 100])
def test_tiled_broadcast_and_packed_inputs(K):
    rng = np.random.default_rng(2)
    A = rng.normal(size=(1000, K)).cumsum(axis=0) + 100
    y = rng.normal(size=1000).cumsum() + 50
    got = s.RollingCorr(20)(A, y)
    for k in range(K):
        np.testing.assert_array_equal(got[:, k], s.RollingCorr(20)(A[:, k], y))
    packed = np.stack([A, A[::-1] * 0.5], axis=-1)       # (T, K, 2)
    got = s.RollingCorr(20)(packed)
    for k in range(K):
        np.testing.assert_array_equal(got[:, k], s.RollingCorr(20)(packed[:, k, 0], packed[:, k, 1]))
