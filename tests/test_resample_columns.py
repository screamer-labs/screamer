"""Resample aggregates every column on its own (shape contract rule 4)."""
import numpy as np
import pytest

import screamer as s

rng = np.random.default_rng(2)
T, K = 60, 3
X = rng.normal(size=(T, K)).cumsum(axis=0)
idx = np.arange(T) * 10


def _per_column(agg, data=X):
    return [s.Resample(freq=50, agg=agg)((data[:, k], idx))[0] for k in range(data.shape[1])]


@pytest.mark.parametrize("agg", ["first", "last", "min", "max", "sum", "count", "mean"])
def test_single_value_aggs(agg):
    v, i = s.Resample(freq=50, agg=agg)((X, idx))
    assert v.shape == (len(i), K)
    for k, col in enumerate(_per_column(agg)):
        np.testing.assert_array_equal(v[:, k], col)


def test_ohlc_appends_a_field_axis():
    v, i = s.Resample(freq=50, agg="ohlc")((X, idx))
    assert v.shape == (len(i), K, 4)
    for k, col in enumerate(_per_column("ohlc")):
        np.testing.assert_array_equal(v[:, k], col)


def test_unit_field_consumed():
    v, _ = s.Resample(freq=50, agg="ohlc")((X[:, :, None], idx))
    assert v.shape[1:] == (K, 4)


def test_t_by_1_keeps_its_column():
    v, _ = s.Resample(count=5, agg="last")(X[:, :1])
    assert v.shape[1:] == (1,)


def test_count_mode_and_positional():
    v, _ = s.Resample(count=7, agg="max")(X)
    assert v.shape == (int(np.ceil(T / 7)), K)


def test_lazy_matches_batch():
    batch, bidx = s.Resample(freq=50, agg="ohlc")((X, idx))
    lazy = list(s.Resample(freq=50, agg="ohlc")(iter(zip(X, idx))))
    np.testing.assert_array_equal(np.array([v for v, _ in lazy]), batch)
    assert [i for _, i in lazy] == list(bidx)


def test_pipeline_wide_input():
    x = s.Input("x")
    p = s.Pipeline([x], [s.Resample(count=5, agg="last")(x)])
    v, _ = p(X)
    np.testing.assert_array_equal(v, X[4::5])


def test_functor_agg_takes_one_column():
    with pytest.raises(ValueError, match="one column"):
        s.Resample(count=5, agg=s.ExpandingSkew())(X)


def test_ohlc_bars_unchanged():
    bars = np.column_stack([X[:, 0]] * 4)
    v, _ = s.Resample(count=5, agg="ohlc_bars")(bars)
    assert v.shape[1] == 4
