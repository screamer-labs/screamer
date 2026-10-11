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
