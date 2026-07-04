import numpy as np
import pytest
from screamer.streams import Stream, _regime, _to_streams, _adapt
from screamer import Input


def test_stream_in_all():
    import screamer.streams as s
    assert "Stream" in s.__all__


def test_regime_classifies_inputs():
    assert _regime([np.array([1.0]), np.array([2.0])]) == "raw"
    assert _regime([Stream(np.array([1.0])), np.array([2.0])]) == "stream"
    assert _regime([Input("a"), np.array([2.0])]) == "graph"        # Node wins
    assert _regime([Input("a"), Stream(np.array([1.0]))]) == "graph"


def test_to_streams_passthrough_and_index_list():
    a = Stream(np.array([1.0, 2.0]), np.array([10, 20]))
    streams = _to_streams([a, np.array([3.0, 4.0])], index=[None, np.array([30, 40])])
    assert streams[0] is a                                          # Stream passes through
    np.testing.assert_array_equal(streams[1].index, [30, 40])       # raw gets its index


def test_to_streams_wrong_length_index_raises():
    with pytest.raises(ValueError, match="length"):
        _to_streams([np.array([1.0]), np.array([2.0])], index=[np.array([0])])


def test_adapt_shapes():
    v = np.array([1.0, 2.0]); idx = np.array([10, 20])
    out_stream = _adapt("stream", v, idx)
    assert isinstance(out_stream, Stream)
    np.testing.assert_array_equal(out_stream.index, idx)
    out_raw = _adapt("raw", v, None)
    assert isinstance(out_raw, tuple) and out_raw[1] is None


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
