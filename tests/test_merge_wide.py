"""G27: Merge of wide streams merges rows in index order, as 1-D streams do.

Equal indexes keep the input order (a lower input first). Every input of a
merge has one row width; a different width raises ValueError naming both.
A wrong kind is a TypeError, never a C++ exception text.
"""
import numpy as np
import pytest

import screamer as s


def ref_merge(streams):
    """The merge by a stable sort on the index: rows, sources, index."""
    rows, srcs, idx = [], [], []
    for k, (v, i) in enumerate(streams):
        for r in range(len(v)):
            rows.append(np.asarray(v[r], dtype=float))
            srcs.append(k)
            idx.append(i[r])
    order = sorted(range(len(idx)), key=lambda j: (idx[j], srcs[j]))
    width = np.shape(streams[0][0])[1:]
    values = np.array([rows[j] for j in order]).reshape((len(order),) + width)
    return values, np.array([srcs[j] for j in order], dtype=np.uint32), np.array([idx[j] for j in order])


RNG = np.random.default_rng(27)


def wide(T, K):
    return RNG.standard_normal((T, K))


def index(T):
    return np.cumsum(RNG.integers(0, 3, T))


@pytest.mark.parametrize("K", [1, 2, 5])
@pytest.mark.parametrize("sizes", [(5, 3), (0, 4), (4, 0), (6, 6, 2), (0, 0)])
def test_indexed_batch_merges_rows_in_index_order(K, sizes):
    streams = [(wide(T, K), index(T)) for T in sizes]
    values, sources, idx = s.Merge()(*streams)
    rv, rs, ri = ref_merge(streams)
    assert values.shape == (sum(sizes), K)
    np.testing.assert_array_equal(values, rv)
    np.testing.assert_array_equal(sources, rs)
    np.testing.assert_array_equal(idx, ri)


@pytest.mark.parametrize("K", [1, 3])
def test_positional_batch_merges_by_row_number(K):
    a, b = wide(4, K), wide(2, K)
    values, sources, idx = s.Merge()(a, b)
    rv, rs, _ = ref_merge([(a, np.arange(4)), (b, np.arange(2))])
    assert idx is None
    np.testing.assert_array_equal(values, rv)
    np.testing.assert_array_equal(sources, rs)


def test_one_d_merge_is_unchanged():
    a, b = (np.arange(3.0), [0, 2, 4]), (np.arange(2.0) + 10, [1, 2])
    values, sources, idx = s.Merge()(a, b)
    np.testing.assert_array_equal(values, [0, 10, 1, 11, 2])
    np.testing.assert_array_equal(sources, [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(idx, [0, 1, 2, 2, 4])


def test_rank_three_rows_keep_their_shape():
    a, b = RNG.standard_normal((3, 2, 2)), RNG.standard_normal((2, 2, 2))
    values, sources, _ = s.Merge()((a, [0, 1, 5]), (b, [1, 2]))
    assert values.shape == (5, 2, 2)
    np.testing.assert_array_equal(values[1], a[1])          # a tie at 1: input 0 first
    np.testing.assert_array_equal(values[2], b[0])
    np.testing.assert_array_equal(sources, [0, 0, 1, 1, 0])


@pytest.mark.parametrize("pair, message", [
    (((np.ones((3, 2)), [0, 1, 2]), (np.ones((2, 3)), [0, 1])),
     "merge: input 1 has rows of width 3 but input 0 has rows of width 2; every input of a merge has the same width"),
    (((np.ones(3), [0, 1, 2]), (np.ones((2, 2)), [0, 1])),
     "merge: input 1 has rows of width 2 but input 0 has rows of width 1; every input of a merge has the same width"),
    (((np.ones((0,)), []), (np.ones((2, 2)), [0, 1])),
     "merge: input 1 has rows of width 2 but input 0 has rows of width 1; every input of a merge has the same width"),
    (((np.ones((2, 2, 2)), [0, 1]), (np.ones((2, 4)), [0, 1])),
     "merge: input 1 has rows of shape (4,) but input 0 has rows of shape (2, 2); every input of a merge has the same row shape"),
])
def test_mixed_widths_raise_naming_both(pair, message):
    with pytest.raises(ValueError) as e:
        s.Merge()(*pair)
    assert str(e.value) == message


@pytest.mark.parametrize("K", [1, 3])
def test_iterator_merge_equals_the_batch_merge(K):
    streams = [(wide(T, K), index(T)) for T in (5, 0, 3)]
    values, sources, idx = s.Merge()(*streams)
    events = list(s.Merge()(*[iter([(v[r] if K > 1 else v[r, 0], i[r]) for r in range(len(v))])
                              for v, i in streams]))
    assert [e[2] for e in events] == list(sources)
    assert [e[1] for e in events] == list(idx)
    got = np.array([np.atleast_1d(e[0]) for e in events]).reshape(values.shape)
    np.testing.assert_array_equal(got, values)
    if K > 1:
        assert all(isinstance(e[0], np.ndarray) and e[0].shape == (K,) for e in events)


def test_positional_iterator_merges_rows():
    a, b = wide(3, 2), wide(1, 2)
    events = list(s.Merge()(iter(list(a)), iter(list(b))))
    assert [e[2] for e in events] == [0, 1, 0, 0]
    assert all(e[1] is None for e in events)
    np.testing.assert_array_equal(events[1][0], b[0])
    np.testing.assert_array_equal(events[3][0], a[2])


def test_iterator_rows_may_be_lists():
    events = list(s.Merge()(iter([([1.0, 2.0], 0)]), iter([([3.0, 4.0], 0)])))
    np.testing.assert_array_equal(events[0][0], [1, 2])
    np.testing.assert_array_equal(events[1][0], [3, 4])


def test_iterator_width_change_raises_naming_the_event():
    with pytest.raises(ValueError, match=r"^merge: event 1 of input 0 has width 1 but input 0 has rows "
                                         r"of width 2; every input of a merge has the same width$"):
        list(s.Merge()(iter([(np.ones(2), 0), (1.0, 2)]), iter([(np.ones(2), 1)])))
    with pytest.raises(ValueError, match=r"^merge: event 0 of input 1 has width 3 but input 0 has rows of width 2"):
        list(s.Merge()(iter([(np.ones(2), 0)]), iter([(np.ones(3), 1)])))


def test_empty_iterators_merge_to_nothing():
    assert list(s.Merge()(iter([]), iter([]))) == []
    assert [e[2] for e in s.Merge()(iter([]), iter([(np.ones(2), 0)]))] == [1]


@pytest.mark.parametrize("call", [
    lambda: s.Merge()(np.array(["a", "b"]), np.ones(2)),
    lambda: s.Merge()((np.ones(2), np.array(["a", "b"])), (np.ones(2), [0, 1])),
    lambda: s.Merge()((np.ones(2), [0, 1]), (np.ones(2), np.array([1 + 1j, 2]))),
    lambda: list(s.Merge()(iter(["a"]), iter([1.0]))),
    lambda: list(s.Merge()(iter([(np.ones(2), "x")]), iter([(np.ones(2), 1)]))),
    lambda: list(s.Merge()(iter([(object(), 0)]), iter([(1.0, 1)]))),
])
def test_a_wrong_kind_is_a_type_error(call):
    with pytest.raises(TypeError) as e:
        call()
    assert "bad_cast" not in str(e.value) and "std::" not in str(e.value)


def test_a_pipeline_merges_wide_feeds_in_index_order():
    a, b = s.Input("a"), s.Input("b")
    p = s.Pipeline([a, b], [s.CombineLatest(emit="on_any")(a, b)])
    va, vb = wide(4, 2), wide(3, 3)
    ia, ib = [0, 2, 2, 5], [1, 2, 6]
    values, idx = p((va, ia), (vb, ib))
    assert values.shape[1] == 5
    live = p.live()
    for (v, i, name) in sorted([(va[r], ia[r], "a") for r in range(4)] +
                               [(vb[r], ib[r], "b") for r in range(3)], key=lambda e: (e[1], e[2])):
        live.push(name, i, v)
    live.flush()
    lv, li = live.result()
    np.testing.assert_array_equal(lv, values)
    np.testing.assert_array_equal(li, idx)
