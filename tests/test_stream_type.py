import numpy as np
import pytest
from screamer.streams import Stream


def test_stream_positional_index_none():
    s = Stream(np.array([1.0, 2.0, 3.0]))
    assert s.index is None
    assert len(s) == 3
    np.testing.assert_array_equal(s.values, [1.0, 2.0, 3.0])


def test_stream_with_index():
    s = Stream(np.array([1.0, 2.0]), index=np.array([10, 20]))
    np.testing.assert_array_equal(s.index, [10, 20])


def test_stream_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        Stream(np.array([1.0, 2.0, 3.0]), index=np.array([10, 20]))


def test_stream_pandas_roundtrip():
    pd = pytest.importorskip("pandas")
    ser = pd.Series([1.0, 2.0, 3.0], index=[100, 200, 300])
    s = Stream.from_pandas(ser)
    np.testing.assert_array_equal(s.values, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(s.index, [100, 200, 300])
    back = s.to_pandas()
    np.testing.assert_array_equal(back.to_numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(np.asarray(back.index), [100, 200, 300])


def test_stream_to_pandas_positional_default_index():
    pd = pytest.importorskip("pandas")
    s = Stream(np.array([1.0, 2.0]))
    ser = s.to_pandas()
    np.testing.assert_array_equal(np.asarray(ser.index), [0, 1])
