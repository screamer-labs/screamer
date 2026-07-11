import numpy as np
import pytest
from screamer.streams import resample, Resample, Stream


def test_resample_returns_stream_with_columns():
    x = np.arange(20.0); idx = np.arange(20, dtype=np.int64)
    bars = Resample(freq=5, agg="ohlc")(x, idx)
    assert isinstance(bars, Stream)
    assert tuple(bars.columns) == ("open", "high", "low", "close")
    np.testing.assert_allclose(bars["open"], bars.values[:, 0])


def test_resample_scalar_agg_columns_none():
    x = np.arange(20.0); idx = np.arange(20, dtype=np.int64)
    bars = Resample(freq=5, agg="sum")(x, idx)
    assert isinstance(bars, Stream)
    assert bars.columns is None


def test_stream_column_access():
    vals = np.array([[1.0, 2.0], [3.0, 4.0]])
    s = Stream(vals, columns=("a", "b"))
    np.testing.assert_array_equal(s["a"], [1.0, 3.0])
    np.testing.assert_array_equal(s.column("b"), [2.0, 4.0])


def test_stream_column_unlabelled_raises():
    s = Stream(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="no column labels"):
        _ = s["any"]


def test_stream_column_missing_raises():
    vals = np.array([[1.0, 2.0], [3.0, 4.0]])
    s = Stream(vals, columns=("a", "b"))
    with pytest.raises(KeyError):
        _ = s["z"]


def test_stream_unpackable_as_values_index():
    vals = np.array([1.0, 2.0, 3.0])
    idx = np.array([10, 20, 30], dtype=np.int64)
    s = Stream(vals, idx)
    v, k = s
    np.testing.assert_array_equal(v, vals)
    np.testing.assert_array_equal(k, idx)


def test_stream_int_getitem():
    vals = np.array([1.0, 2.0])
    idx = np.array([10, 20], dtype=np.int64)
    s = Stream(vals, idx)
    np.testing.assert_array_equal(s[0], vals)
    np.testing.assert_array_equal(s[1], idx)
