import numpy as np
import pytest
from screamer.streams import combine_latest, combine_latest_iter, Stream


def test_positional_lockstep_equal_length():
    v, idx = combine_latest(np.array([10.0, 20.0, 40.0]), np.array([1.0, 3.0, 4.0]))
    assert idx is None
    assert v.shape == (3, 2)
    np.testing.assert_array_equal(v[:, 0], [10.0, 20.0, 40.0])


def test_positional_unequal_length_raises():
    with pytest.raises(ValueError, match="length"):
        combine_latest(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_indexed_coalesces_same_index_rows():
    a = np.array([10.0, 20.0, 40.0]); ta = np.array([1, 2, 4])
    b = np.array([1.0, 3.0, 4.0]);    tb = np.array([1, 3, 4])
    v, idx = combine_latest(a, b, index=[ta, tb])
    np.testing.assert_array_equal(idx, [1, 2, 3, 4])          # NOT [1,2,3,4,4]
    np.testing.assert_array_equal(v, [[10, 1], [20, 1], [20, 3], [40, 4]])


def test_stream_in_stream_out():
    a = Stream(np.array([10.0, 20.0]), np.array([1, 2]))
    b = Stream(np.array([1.0, 2.0]), np.array([1, 2]))
    out = combine_latest(a, b)
    assert isinstance(out, Stream)
    np.testing.assert_array_equal(out.index, [1, 2])


def test_mixed_positional_and_indexed_raises():
    with pytest.raises(ValueError, match="positional"):
        combine_latest(Stream(np.array([1.0, 2.0]), np.array([1, 2])),
                       Stream(np.array([3.0, 4.0])))


def test_node_in_node_out():
    from screamer import Input
    from screamer.dag import is_node
    assert is_node(combine_latest(Input("a"), Input("b")))


def test_iter_coalesces_and_yields_row_index():
    a = Stream(np.array([10.0, 20.0]), np.array([1, 2]))
    b = Stream(np.array([1.0, 2.0]), np.array([1, 2]))
    events = list(combine_latest_iter(a, b))
    idxs = [i for _, i in events]
    assert idxs == [1, 2]
    np.testing.assert_array_equal(np.asarray(events[0][0]), [10.0, 1.0])
