import numpy as np
import pytest

from screamer.streams import select, select_iter


def _wide():
    keys = np.array([1, 2, 3], dtype=np.int64)
    values = np.array([[10.0, 11.0, 12.0],
                       [20.0, 21.0, 22.0],
                       [30.0, 31.0, 32.0]])
    return keys, values


def test_select_single_int_returns_1d():
    keys, values = _wide()
    k, v = select(keys, values, 1)
    np.testing.assert_array_equal(k, keys)
    assert v.ndim == 1
    np.testing.assert_array_equal(v, [11.0, 21.0, 31.0])


def test_select_list_preserves_order():
    keys, values = _wide()
    k, v = select(keys, values, [2, 0])
    assert v.shape == (3, 2)
    np.testing.assert_array_equal(v, [[12.0, 10.0], [22.0, 20.0], [32.0, 30.0]])
    np.testing.assert_array_equal(k, keys)


def test_select_out_of_range_raises():
    keys, values = _wide()
    with pytest.raises((ValueError, IndexError)):
        select(keys, values, 5)


def test_select_negative_index_raises():
    keys, values = _wide()
    with pytest.raises(ValueError):
        select(keys, values, -1)


def test_select_iter_matches_batch():
    keys, values = _wide()
    events = list(zip(keys.tolist(), values.tolist()))
    got = list(select_iter(events, [0, 2]))
    assert [k for k, _ in got] == [1, 2, 3]
    assert [list(v) for _, v in got] == [[10.0, 12.0], [20.0, 22.0], [30.0, 32.0]]


def test_select_iter_scalar_int_yields_floats():
    # scalar-int path yields a bare float per event (not a 1-element list)
    keys, values = _wide()
    events = list(zip(keys.tolist(), values.tolist()))
    got = list(select_iter(events, 1))
    assert [k for k, _ in got] == [1, 2, 3]
    vs = [v for _, v in got]
    assert vs == [11.0, 21.0, 31.0]
    assert all(isinstance(v, float) for v in vs)


def test_select_1d_input_width1():
    # a flat (M,) stream is a width-1 stream; column 0 mirrors the input
    keys = np.array([1, 2, 3], dtype=np.int64)
    values = np.array([5.0, 6.0, 7.0])
    k, v = select(keys, values, 0)
    np.testing.assert_array_equal(k, keys)
    assert v.ndim == 1
    np.testing.assert_array_equal(v, values)
    # out-of-range on a width-1 stream still raises
    with pytest.raises(ValueError):
        select(keys, values, 1)
