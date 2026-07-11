import numpy as np
import pytest
from screamer.streams import _to_streams, to_pandas, from_pandas, resample
from screamer import Input


def test_stream_not_in_all():
    import screamer.streams as s
    assert "Stream" not in s.__all__


def test_to_pandas_in_all():
    import screamer.streams as s
    assert "to_pandas" in s.__all__
    assert "from_pandas" in s.__all__


def test_to_streams_tuple_passthrough():
    v = np.array([1.0, 2.0])
    k = np.array([10, 20])
    result = _to_streams([(v, k), np.array([3.0, 4.0])], index=[None, np.array([30, 40])])
    # tuple input passes through
    assert result[0] is (v, k) or (np.array_equal(result[0][0], v) and np.array_equal(result[0][1], k))
    # raw array gets its index
    np.testing.assert_array_equal(result[1][1], [30, 40])


def test_to_streams_bare_array_gets_index():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    result = _to_streams([a, b], index=[np.array([10, 20]), np.array([30, 40])])
    np.testing.assert_array_equal(result[0][1], [10, 20])
    np.testing.assert_array_equal(result[1][1], [30, 40])


def test_to_pandas_1d():
    pd = pytest.importorskip("pandas")
    v = np.array([1.0, 2.0, 3.0])
    k = np.array([100, 200, 300])
    ser = to_pandas(v, k)
    np.testing.assert_array_equal(ser.to_numpy(), v)
    np.testing.assert_array_equal(np.asarray(ser.index), k)


def test_to_pandas_positional_default_index():
    pd = pytest.importorskip("pandas")
    v = np.array([1.0, 2.0])
    ser = to_pandas(v)
    np.testing.assert_array_equal(np.asarray(ser.index), [0, 1])


def test_to_pandas_2d_with_columns():
    pd = pytest.importorskip("pandas")
    v = np.array([[1.0, 2.0], [3.0, 4.0]])
    k = np.array([10, 20])
    df = to_pandas(v, k, columns=["a", "b"])
    assert list(df.columns) == ["a", "b"]
    np.testing.assert_array_equal(df["a"].to_numpy(), [1.0, 3.0])


def test_from_pandas_series():
    pd = pytest.importorskip("pandas")
    ser = pd.Series([1.0, 2.0, 3.0], index=[100, 200, 300])
    v, k = from_pandas(ser)
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(k, [100, 200, 300])


def test_from_pandas_roundtrip():
    pd = pytest.importorskip("pandas")
    ser = pd.Series([1.0, 2.0, 3.0], index=[100, 200, 300])
    v, k = from_pandas(ser)
    back = to_pandas(v, k)
    np.testing.assert_array_equal(back.to_numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(np.asarray(back.index), [100, 200, 300])


def test_resample_returns_tuple():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    result = resample(v, k, freq=2, agg="last")
    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert result[1] is not None  # bar labels always real


def test_positional_resample_index_is_not_none():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    result = resample(v, freq=2, agg="last")
    assert isinstance(result, tuple)
    assert result[1] is not None
