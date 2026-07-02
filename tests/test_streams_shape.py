import numpy as np
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
