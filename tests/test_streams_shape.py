import numpy as np
import pytest
from screamer import streams


def test_dropna_any_on_aligned():
    keys = np.array([1, 2, 3, 4], dtype=np.int64)
    vals = np.array([[1.0, 10.0],
                     [np.nan, 20.0],
                     [3.0, np.nan],
                     [4.0, 40.0]])
    gk, gv = streams.dropna(keys, vals)                 # how="any" default
    np.testing.assert_array_equal(gk, np.array([1, 4], dtype=np.int64))
    np.testing.assert_array_equal(gv, np.array([[1.0, 10.0], [4.0, 40.0]]))


def test_dropna_all_on_aligned():
    keys = np.array([1, 2, 3], dtype=np.int64)
    vals = np.array([[np.nan, np.nan],
                     [np.nan, 2.0],
                     [3.0, 3.0]])
    gk, gv = streams.dropna(keys, vals, how="all")
    np.testing.assert_array_equal(gk, np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(gv, np.array([[np.nan, 2.0], [3.0, 3.0]]))


def test_dropna_1d():
    keys = np.array([1, 2, 3], dtype=np.int64)
    vals = np.array([1.0, np.nan, 3.0])
    gk, gv = streams.dropna(keys, vals)
    np.testing.assert_array_equal(gk, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(gv, np.array([1.0, 3.0]))


def test_filter_1d_predicate():
    keys = np.array([1, 2, 3, 4], dtype=np.int64)
    vals = np.array([-1.0, 2.0, -3.0, 4.0])
    gk, gv = streams.filter(keys, vals, lambda v: v > 0)
    np.testing.assert_array_equal(gk, np.array([2, 4], dtype=np.int64))
    np.testing.assert_array_equal(gv, np.array([2.0, 4.0]))


def test_dropna_iter_matches_batch():
    keys = np.array([1, 2, 3], dtype=np.int64)
    vals = np.array([1.0, np.nan, 3.0])
    events = [(int(k), float(v)) for k, v in zip(keys, vals)]
    got = list(streams.dropna_iter(events))
    assert got == [(1, 1.0), (3, 3.0)]


def test_split_inverts_merge():
    a_k = np.array([1, 3, 5], dtype=np.int64)
    a_v = np.array([10.0, 30.0, 50.0])
    b_k = np.array([2, 4], dtype=np.int64)
    b_v = np.array([20.0, 40.0])
    mk, mv, ms = streams.merge((a_k, a_v), (b_k, b_v))
    parts = streams.split(mk, mv, ms)
    assert len(parts) == 2
    np.testing.assert_array_equal(parts[0][0], a_k)
    np.testing.assert_array_equal(parts[0][1], a_v)
    np.testing.assert_array_equal(parts[1][0], b_k)
    np.testing.assert_array_equal(parts[1][1], b_v)


def test_split_explicit_n_includes_empty():
    keys = np.array([1, 2], dtype=np.int64)
    vals = np.array([1.0, 2.0])
    src = np.array([0, 0], dtype=np.uint32)
    parts = streams.split(keys, vals, src, n=3)   # sources 1 and 2 are empty
    assert len(parts) == 3
    assert parts[1][0].size == 0 and parts[2][0].size == 0


def test_split_rejects_too_small_n():
    keys = np.array([1, 2, 3], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    src = np.array([0, 1, 2], dtype=np.uint32)     # needs n >= 3
    with pytest.raises(ValueError):
        streams.split(keys, vals, src, n=2)         # would drop source 2 silently


def test_filter_iter_streaming():
    events = [(1, -1.0), (2, 2.0), (3, -3.0), (4, 4.0)]
    got = list(streams.filter_iter(events, lambda v: v > 0))
    assert got == [(2, 2.0), (4, 4.0)]


def test_dropna_iter_2d_rows():
    events = [(1, (1.0, 10.0)), (2, (np.nan, 20.0)), (3, (3.0, 30.0))]
    got_any = list(streams.dropna_iter(events, how="any"))
    assert [k for k, _ in got_any] == [1, 3]
    got_all = list(streams.dropna_iter(events, how="all"))
    assert [k for k, _ in got_all] == [1, 2, 3]      # no row is all-NaN


def test_filter_2d_row_predicate():
    keys = np.array([1, 2, 3], dtype=np.int64)
    vals = np.array([[1.0, 1.0], [5.0, 5.0], [2.0, 2.0]])
    gk, gv = streams.filter(keys, vals, lambda row: row.sum() > 5.0)
    np.testing.assert_array_equal(gk, np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(gv, np.array([[5.0, 5.0]]))
