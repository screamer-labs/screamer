import numpy as np
import pytest

from screamer.streams import select, select_iter, Stream


def _wide():
    """Return a (values, keys) pair with 3 rows and 3 columns."""
    values = np.array([[10.0, 11.0, 12.0],
                       [20.0, 21.0, 22.0],
                       [30.0, 31.0, 32.0]])
    keys = np.array([1, 2, 3], dtype=np.int64)
    return values, keys


# ---------------------------------------------------------------------------
# select – raw array, values-first
# ---------------------------------------------------------------------------

def test_select_single_int_returns_1d():
    values, keys = _wide()
    v, k = select(values, 1, index=keys)
    np.testing.assert_array_equal(k, keys)
    assert v.ndim == 1
    np.testing.assert_array_equal(v, [11.0, 21.0, 31.0])


def test_select_list_preserves_order():
    values, keys = _wide()
    v, k = select(values, [2, 0], index=keys)
    assert v.shape == (3, 2)
    np.testing.assert_array_equal(v, [[12.0, 10.0], [22.0, 20.0], [32.0, 30.0]])
    np.testing.assert_array_equal(k, keys)


def test_select_out_of_range_raises():
    values, keys = _wide()
    with pytest.raises((ValueError, IndexError)):
        select(values, 5, index=keys)


def test_select_negative_index_raises():
    values, keys = _wide()
    with pytest.raises(ValueError):
        select(values, -1, index=keys)


def test_select_1d_input_width1():
    """A flat (M,) stream is a width-1 stream; column 0 mirrors the input."""
    values = np.array([5.0, 6.0, 7.0])
    keys = np.array([1, 2, 3], dtype=np.int64)
    v, k = select(values, 0, index=keys)
    np.testing.assert_array_equal(k, keys)
    assert v.ndim == 1
    np.testing.assert_array_equal(v, values)
    # out-of-range on a width-1 stream still raises
    with pytest.raises(ValueError):
        select(values, 1, index=keys)


def test_select_positional_index_is_none():
    """index=None -> positional; returned index is None (no allocation)."""
    values, _ = _wide()
    v, k = select(values, 1)
    assert k is None
    assert v.ndim == 1
    np.testing.assert_array_equal(v, [11.0, 21.0, 31.0])


def test_select_missing_columns_raises_streams():
    """columns is required; omitting it raises TypeError (positional argument)."""
    values, _ = _wide()
    with pytest.raises(TypeError):
        select(values)


# ---------------------------------------------------------------------------
# select – Stream / Node mirroring
# ---------------------------------------------------------------------------

def test_select_stream_in_stream_out():
    values, keys = _wide()
    s = Stream(values, keys)
    out = select(s, 1)
    assert isinstance(out, Stream)
    # index is row-preserving (unchanged)
    np.testing.assert_array_equal(out.index, keys)
    np.testing.assert_array_equal(out.values, [11.0, 21.0, 31.0])


def test_select_stream_positional():
    values, _ = _wide()
    s = Stream(values)                # positional stream
    out = select(s, 0)
    assert isinstance(out, Stream)
    assert out.index is None
    np.testing.assert_array_equal(out.values, [10.0, 20.0, 30.0])


def test_select_node_in_node_out():
    from screamer import Input
    from screamer.dag import is_node
    x = Input("x")
    assert is_node(select(x, 0))


# ---------------------------------------------------------------------------
# select_iter – (value, index) event model
# ---------------------------------------------------------------------------

def test_select_iter_matches_batch():
    values, keys = _wide()
    # events are (value_row, index) tuples
    events = list(zip(values.tolist(), keys.tolist()))
    got = list(select_iter(events, [0, 2]))
    assert [k for _, k in got] == [1, 2, 3]
    assert [list(v) for v, _ in got] == [[10.0, 12.0], [20.0, 22.0], [30.0, 32.0]]


def test_select_iter_scalar_int_yields_floats():
    """scalar-int path yields a bare float per event (not a 1-element list)."""
    values, keys = _wide()
    events = list(zip(values.tolist(), keys.tolist()))
    got = list(select_iter(events, 1))
    assert [k for _, k in got] == [1, 2, 3]
    vs = [v for v, _ in got]
    assert vs == [11.0, 21.0, 31.0]
    assert all(isinstance(v, float) for v in vs)


def test_select_iter_positional():
    """Positional events use index=None; index passes through unchanged."""
    values, _ = _wide()
    events = [(row, None) for row in values.tolist()]
    got = list(select_iter(events, 1))
    assert [k for _, k in got] == [None, None, None]
    assert [v for v, _ in got] == [11.0, 21.0, 31.0]
