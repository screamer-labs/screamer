import numpy as np
import pytest
from screamer.supervised import forecast_pairs


def test_forecast_pairs_count_pairs_features_with_future_target():
    # feature at row t pairs with the target `count` rows later
    X = np.arange(6.0)                       # 0,1,2,3,4,5
    y = np.arange(6.0) * 10                  # 0,10,20,30,40,50
    Xs, ys, as_of = forecast_pairs(X, y, count=2)
    # row t holds feature X[t-2] and target y[t]; first 2 rows warm up to NaN
    assert np.isnan(Xs[:2]).all()
    np.testing.assert_array_equal(Xs[2:], [0.0, 1.0, 2.0, 3.0])   # X[t-2]
    np.testing.assert_array_equal(ys, y)                          # y untouched
    np.testing.assert_array_equal(as_of, np.arange(6))


def test_forecast_pairs_count_dropna_returns_clean_pairs():
    X = np.arange(6.0)
    y = np.arange(6.0) * 10
    Xs, ys, as_of = forecast_pairs(X, y, count=2, dropna=True)
    assert not np.isnan(Xs).any()
    np.testing.assert_array_equal(Xs, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(ys, [20.0, 30.0, 40.0, 50.0])
    np.testing.assert_array_equal(as_of, [2, 3, 4, 5])


def test_forecast_pairs_count_matches_forward_return_reference():
    # forecast_pairs(X, RollingSum(h)(ret), count=h) reproduces the forward-return pairing
    from screamer import RollingSum
    rng = np.random.default_rng(0)
    n, h = 50, 5
    ret = rng.standard_normal(n) * 1e-3
    X = rng.standard_normal(n)
    y = np.asarray(RollingSum(h)(ret))
    Xs, ys, as_of = forecast_pairs(X, y, count=h, dropna=True)
    # reference: X[s] paired with sum(ret[s+1..s+h]) for valid s
    fwd = np.array([ret[s + 1:s + 1 + h].sum() for s in range(n - h)])
    np.testing.assert_allclose(Xs, X[:n - h])
    np.testing.assert_allclose(ys, fwd)


def test_forecast_pairs_requires_exactly_one_of_count_duration():
    X = np.arange(5.0); y = np.arange(5.0)
    with pytest.raises(ValueError):
        forecast_pairs(X, y)                       # neither
    with pytest.raises(ValueError):
        forecast_pairs(X, y, count=1, duration=1)  # both


def test_forecast_pairs_count_2d_features_per_column():
    X = np.column_stack([np.arange(6.0), np.arange(6.0) * 2])
    y = np.arange(6.0)
    Xs, ys, as_of = forecast_pairs(X, y, count=2, dropna=True)
    np.testing.assert_array_equal(Xs, np.column_stack([[0, 1, 2, 3], [0, 2, 4, 6]]))
