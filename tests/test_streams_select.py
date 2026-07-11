import numpy as np
import pytest

from screamer.streams import Select, Stream


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
    v, k = Select(1)(values, index=keys)
    np.testing.assert_array_equal(k, keys)
    assert v.ndim == 1
    np.testing.assert_array_equal(v, [11.0, 21.0, 31.0])


def test_select_list_preserves_order():
    values, keys = _wide()
    v, k = Select([2, 0])(values, index=keys)
    assert v.shape == (3, 2)
    np.testing.assert_array_equal(v, [[12.0, 10.0], [22.0, 20.0], [32.0, 30.0]])
    np.testing.assert_array_equal(k, keys)


def test_select_out_of_range_raises():
    values, keys = _wide()
    with pytest.raises((ValueError, IndexError)):
        Select(5)(values, index=keys)


def test_select_negative_index_raises():
    values, keys = _wide()
    with pytest.raises(ValueError):
        Select(-1)(values, index=keys)


def test_select_1d_input_width1():
    """A flat (M,) stream is a width-1 stream; column 0 mirrors the input."""
    values = np.array([5.0, 6.0, 7.0])
    keys = np.array([1, 2, 3], dtype=np.int64)
    v, k = Select(0)(values, index=keys)
    np.testing.assert_array_equal(k, keys)
    assert v.ndim == 1
    np.testing.assert_array_equal(v, values)
    # out-of-range on a width-1 stream still raises
    with pytest.raises(ValueError):
        Select(1)(values, index=keys)


def test_select_positional_index_is_none():
    """index=None -> positional; returned index is None (no allocation)."""
    values, _ = _wide()
    v, k = Select(1)(values)
    assert k is None
    assert v.ndim == 1
    np.testing.assert_array_equal(v, [11.0, 21.0, 31.0])


def test_select_missing_columns_raises_streams():
    """columns is required; omitting it raises TypeError (positional argument)."""
    values, _ = _wide()
    with pytest.raises(TypeError):
        Select()(values)


# ---------------------------------------------------------------------------
# select – Stream / Node mirroring
# ---------------------------------------------------------------------------

def test_select_stream_in_stream_out():
    values, keys = _wide()
    s = Stream(values, keys)
    out = Select(1)(s)
    assert isinstance(out, Stream)
    # index is row-preserving (unchanged)
    np.testing.assert_array_equal(out.index, keys)
    np.testing.assert_array_equal(out.values, [11.0, 21.0, 31.0])


def test_select_stream_positional():
    values, _ = _wide()
    s = Stream(values)                # positional stream
    out = Select(0)(s)
    assert isinstance(out, Stream)
    assert out.index is None
    np.testing.assert_array_equal(out.values, [10.0, 20.0, 30.0])


def test_select_node_in_node_out():
    from screamer import Input
    from screamer.dag import is_node
    x = Input("x")
    assert is_node(Select(0)(x))


# ---------------------------------------------------------------------------
# lazy operator dispatch - select (value, index) event model
# ---------------------------------------------------------------------------

def test_select_iter_matches_batch():
    values, keys = _wide()
    # events are (value_row, index) tuples
    gen = ((row, k) for row, k in zip(values.tolist(), keys.tolist()))
    got = list(Select([0, 2])(gen))
    assert [k for _, k in got] == [1, 2, 3]
    assert [list(v) for v, _ in got] == [[10.0, 12.0], [20.0, 22.0], [30.0, 32.0]]


def test_select_iter_scalar_int_yields_floats():
    """scalar-int path yields a bare float per event (not a 1-element list)."""
    values, keys = _wide()
    gen = ((row, k) for row, k in zip(values.tolist(), keys.tolist()))
    got = list(Select(1)(gen))
    assert [k for _, k in got] == [1, 2, 3]
    vs = [v for v, _ in got]
    assert vs == [11.0, 21.0, 31.0]
    assert all(isinstance(v, float) for v in vs)


def test_select_iter_positional():
    """Positional events use index=None; index passes through unchanged."""
    values, _ = _wide()
    gen = ((row, None) for row in values.tolist())
    got = list(Select(1)(gen))
    assert [k for _, k in got] == [None, None, None]
    assert [v for v, _ in got] == [11.0, 21.0, 31.0]


# ---------------------------------------------------------------------------
# lazy dispatch oracle test
# ---------------------------------------------------------------------------

def test_select_lazy_equals_batch():
    import numpy as np
    from screamer.streams import Select as _Select
    vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    idx = np.array([0, 1, 2])
    bv, bk = _Select(1)(vals, index=idx)                 # scalar column -> 1-D
    gen = ((row.tolist(), int(k)) for row, k in zip(vals, idx))
    rows = list(_Select(1)(gen))
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(bv))
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bk))
