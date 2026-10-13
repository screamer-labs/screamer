"""Dropna and Select on wide streams with equal indexes, Dropna of several
streams, and Select's fresh output (3.0.0, H2 findings 2 and 13).

The direct call, the iterator call and the same node inside a Pipeline give one
result: every row is kept or dropped on its own, also when rows share an index.
"""
import numpy as np
import pytest

import screamer as s

V = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [np.nan, 40.0], [5.0, 50.0]])
K = np.array([0, 1, 1, 1, 2])


def _pipeline(build, n=1):
    ins = [s.Input(f"x{i}") for i in range(n)]
    return s.Pipeline(ins, [build(*ins)])


def _lazy(values, index):
    return list(zip(values, index))


def _rows(events):
    return [tuple(np.atleast_1d(v)) for v, _ in events], [k for _, k in events]


@pytest.mark.parametrize("how, keep", [("any", [0, 1, 2, 4]), ("all", [0, 1, 2, 3, 4])])
def test_dropna_keeps_rows_with_equal_indexes(how, keep):
    want_v, want_k = V[keep], K[keep]
    v, k = s.Dropna(how=how)((V, K))
    np.testing.assert_array_equal(v, want_v)
    np.testing.assert_array_equal(k, want_k)
    pv, pk = _pipeline(lambda x: s.Dropna(how=how)(x))((V, K))
    np.testing.assert_array_equal(pv, want_v)
    np.testing.assert_array_equal(pk, want_k)
    lv, lk = _rows(list(s.Dropna(how=how)(iter(_lazy(V, K)))))
    np.testing.assert_array_equal(np.array(lv), want_v)
    assert lk == list(want_k)


@pytest.mark.parametrize("columns", [1, [1], [1, 0]])
def test_select_keeps_rows_with_equal_indexes(columns):
    want = V[:, columns]
    if want.ndim == 2 and want.shape[1] == 1:
        want = want[:, 0]                  # one column of a wide stream is 1-D
    v, k = s.Select(columns)((V, K))
    np.testing.assert_array_equal(v, want)
    np.testing.assert_array_equal(k, K)
    pv, pk = _pipeline(lambda x: s.Select(columns)(x))((V, K))
    np.testing.assert_array_equal(pv, want)
    np.testing.assert_array_equal(pk, K)
    events = list(s.Select(columns)(iter(_lazy(V, K))))
    np.testing.assert_array_equal(np.array([np.atleast_1d(e[0]) for e in events]).reshape(want.shape), want)
    assert [e[1] for e in events] == list(K)


def test_positional_wide_calls_keep_none_index():
    v, k = s.Dropna()(V)
    np.testing.assert_array_equal(v, V[[0, 1, 2, 4]])
    assert k is None
    v, k = s.Select([1, 0])(V)
    np.testing.assert_array_equal(v, V[:, [1, 0]])
    assert k is None
    assert [e[1] for e in s.Dropna()(iter(_lazy(V, [None] * 5)))] == [None] * 4


def test_dropna_of_several_streams_is_one_dropna_node():
    a = (np.array([1.0, np.nan, 3.0]), np.array([0, 2, 4]))
    b = (np.array([np.nan, 2.0, 4.0]), np.array([1, 2, 3]))
    want = _pipeline(lambda x, y: s.Dropna()(x, y), 2)(a, b)
    got = s.Dropna()(a, b)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    np.testing.assert_array_equal(got[0], [1.0, 2.0, 4.0, 3.0])   # in index order
    np.testing.assert_array_equal(got[1], [0, 2, 3, 4])
    lazy = list(s.Dropna()(iter(zip(*a)), iter(zip(*b))))
    assert [v for v, _ in lazy] == [1.0, 2.0, 4.0, 3.0]
    assert [k for _, k in lazy] == [0, 2, 3, 4]
    # The node form takes several streams too.
    x, y = s.Input("x"), s.Input("y")
    node = s.Dropna()(x, y)
    np.testing.assert_array_equal(s.Pipeline([x, y], [node])(a, b)[0], want[0])


def test_dropna_index_goes_in_a_tuple():
    vals = np.array([1.0, np.nan, 3.0])
    idx = np.array([5, 6, 7])
    np.testing.assert_array_equal(s.Dropna()((vals, idx))[1], [5, 7])
    np.testing.assert_array_equal(s.Dropna()(vals, index=idx)[1], [5, 7])
    with pytest.raises(TypeError, match="index= applies to one input"):
        s.Dropna()(vals, vals, index=idx)
    with pytest.raises(ValueError, match="cannot mix positional and indexed"):
        s.Dropna()(vals, (vals, idx))


def test_select_of_a_column_is_a_new_array():
    y = np.arange(4.0)
    for columns in (0, [0]):
        out, _ = s.Select(columns)(y)
        assert not np.shares_memory(out, y)
        out[...] = -1.0
    np.testing.assert_array_equal(y, np.arange(4.0))
