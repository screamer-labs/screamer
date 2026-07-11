import numpy as np
import pytest
from screamer import streams


def _reference_merge(values_list, index_list):
    """Reference: stable sort by index, ties broken by source index."""
    all_idx = np.concatenate([np.asarray(k) for k in index_list])
    all_vals = np.concatenate([np.asarray(v) for v in values_list])
    all_src = np.concatenate([np.full(len(k), i, dtype=np.uint32)
                              for i, k in enumerate(index_list)])
    order = np.argsort(all_idx, kind="stable")
    return all_vals[order], all_src[order], all_idx[order]


# ---------------------------------------------------------------------------
# merge - indexed
# ---------------------------------------------------------------------------

def test_merge_node_raises_immediately():
    """merge does not support DAG graph nodes; it must raise ValueError immediately."""
    from screamer import Input
    x = Input("x")
    with pytest.raises(ValueError, match="not supported as a DAG graph node"):
        streams.Merge()(x)


def test_merge_indexed_two_int_sorted_series():
    a_v = np.array([10.0, 30.0, 50.0, 70.0])
    a_k = np.array([1, 3, 5, 7], dtype=np.int64)
    b_v = np.array([20.0, 40.0, 60.0])
    b_k = np.array([2, 4, 6], dtype=np.int64)
    got_v, got_s, got_i = streams.Merge()(a_v, b_v, index=[a_k, b_k])
    exp_v, exp_s, exp_i = _reference_merge([a_v, b_v], [a_k, b_k])
    np.testing.assert_array_equal(got_v, exp_v)
    np.testing.assert_array_equal(got_s, exp_s)
    np.testing.assert_array_equal(got_i, exp_i)


def test_merge_indexed_ties_break_by_source_order():
    a_v = np.array([1.0, 2.0, 2.5])
    a_k = np.array([1, 2, 2], dtype=np.int64)
    b_v = np.array([20.0, 30.0])
    b_k = np.array([2, 3], dtype=np.int64)
    got_v, got_s, got_i = streams.Merge()(a_v, b_v, index=[a_k, b_k])
    # at key==2: source 0's events (2.0, 2.5) come before source 1's (20.0)
    np.testing.assert_array_equal(got_i, np.array([1, 2, 2, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(got_v, np.array([1.0, 2.0, 2.5, 20.0, 30.0]))
    np.testing.assert_array_equal(got_s, np.array([0, 0, 0, 1, 1], dtype=np.uint32))


def test_merge_indexed_float_keys():
    a_v = np.array([5.0, 25.0])
    a_k = np.array([0.5, 2.5], dtype=np.float64)
    b_v = np.array([15.0])
    b_k = np.array([1.5], dtype=np.float64)
    got_v, got_s, got_i = streams.Merge()(a_v, b_v, index=[a_k, b_k])
    exp_v, exp_s, exp_i = _reference_merge([a_v, b_v], [a_k, b_k])
    np.testing.assert_array_equal(got_v, exp_v)
    np.testing.assert_array_equal(got_s, exp_s)
    np.testing.assert_array_equal(got_i, exp_i)


# ---------------------------------------------------------------------------
# merge - positional
# ---------------------------------------------------------------------------

def test_merge_positional_returns_none_index():
    a_v = np.array([10.0, 20.0, 30.0])
    b_v = np.array([5.0, 15.0, 25.0])
    got_v, got_s, got_i = streams.Merge()(a_v, b_v)
    assert got_i is None
    # positional: both streams share row-numbers [0,1,2]; ties go source-0-first
    np.testing.assert_array_equal(got_v, np.array([10.0, 5.0, 20.0, 15.0, 30.0, 25.0]))
    np.testing.assert_array_equal(got_s, np.array([0, 1, 0, 1, 0, 1], dtype=np.uint32))


def test_merge_positional_unequal_length_allowed():
    # Unlike combine_latest, positional merge does NOT require equal lengths
    a_v = np.array([1.0, 2.0, 3.0])
    b_v = np.array([10.0, 20.0])
    got_v, got_s, got_i = streams.Merge()(a_v, b_v)
    assert got_i is None
    assert len(got_v) == 5   # 3 + 2
    assert len(got_s) == 5


def test_merge_mixing_positional_and_indexed_raises():
    a_v = np.array([1.0, 2.0])
    a_k = np.array([0, 1], dtype=np.int64)
    b_v = np.array([10.0, 20.0])
    with pytest.raises(ValueError, match="mix"):
        streams.Merge()(a_v, b_v, index=[a_k, None])


# ---------------------------------------------------------------------------
# merge lazy - indexed and positional (migrated from merge_iter)
# ---------------------------------------------------------------------------

def test_merge_lazy_large_indexed_matches_batch():
    rng = np.random.default_rng(7)
    a_k = np.sort(rng.integers(0, 1000, size=200)).astype(np.int64)
    a_v = rng.standard_normal(200)
    b_k = np.sort(rng.integers(0, 1000, size=150)).astype(np.int64)
    b_v = rng.standard_normal(150)

    bv, bs, bi = streams.Merge()(a_v, b_v, index=[a_k, b_k])
    ga = ((float(v), int(k)) for v, k in zip(a_v, a_k))
    gb = ((float(v), int(k)) for v, k in zip(b_v, b_k))
    events = list(streams.Merge()(ga, gb))

    ev_v = np.array([e[0] for e in events])
    ev_i = np.array([e[1] for e in events], dtype=np.int64)
    ev_s = np.array([e[2] for e in events], dtype=np.uint32)
    np.testing.assert_array_equal(ev_v, bv)
    np.testing.assert_array_equal(ev_i, bi)
    np.testing.assert_array_equal(ev_s, bs)


def test_merge_lazy_positional_yields_none_index():
    a_v = np.array([1.0, 2.0])
    b_v = np.array([10.0, 20.0])
    events = list(streams.Merge()((x for x in a_v), (x for x in b_v)))
    for value, ev_index, source in events:
        assert ev_index is None


# ---------------------------------------------------------------------------
# split - round-trip
# ---------------------------------------------------------------------------

def test_split_indexed_round_trip():
    a_v = np.array([10.0, 20.0, 30.0])
    a_k = np.array([1, 3, 5], dtype=np.int64)
    b_v = np.array([5.0, 15.0, 25.0])
    b_k = np.array([2, 4, 6], dtype=np.int64)

    parts = streams.split(*streams.Merge()(a_v, b_v, index=[a_k, b_k]))

    np.testing.assert_array_equal(parts[0][0], a_v)
    np.testing.assert_array_equal(parts[0][1], a_k)
    np.testing.assert_array_equal(parts[1][0], b_v)
    np.testing.assert_array_equal(parts[1][1], b_k)


def test_split_positional_round_trip():
    a_v = np.array([1.0, 2.0, 3.0])
    b_v = np.array([10.0, 20.0, 30.0])

    parts = streams.split(*streams.Merge()(a_v, b_v))

    np.testing.assert_array_equal(parts[0][0], a_v)
    assert parts[0][1] is None
    np.testing.assert_array_equal(parts[1][0], b_v)
    assert parts[1][1] is None


def test_split_n_explicit():
    # n can be set to include sources that emitted nothing
    a_v = np.array([1.0, 2.0])
    a_k = np.array([0, 1], dtype=np.int64)
    merged_v, sources, merged_i = streams.Merge()(a_v, index=[a_k])
    parts = streams.split(merged_v, sources, merged_i, n=3)
    assert len(parts) == 3
    np.testing.assert_array_equal(parts[0][0], a_v)
    assert len(parts[1][0]) == 0
    assert len(parts[2][0]) == 0


# ---------------------------------------------------------------------------
# merge - lazy dispatch (oracle tests, written first)
# ---------------------------------------------------------------------------

def test_merge_lazy_is_lazy():
    from screamer.streams import Merge as _Merge
    pulled = {"a": [], "b": []}

    def spy(name, items):
        for i, v in enumerate(items):
            pulled[name].append(v)
            yield (float(v), i)

    it = _Merge()(spy("a", [1.0, 2.0]), spy("b", [10.0, 20.0]))
    assert pulled == {"a": [], "b": []}
    next(it)
    assert pulled["a"] == [1.0] and pulled["b"] == [10.0]   # one head per source


def test_merge_lazy_indexed_equals_batch():
    import numpy as np
    from screamer.streams import Merge as _Merge
    av, ak = np.array([1.0, 2.0, 3.0]), np.array([0, 2, 4])
    bv, bk = np.array([10.0, 20.0]),     np.array([1, 3])
    mvals, msrc, midx = _Merge()(av, bv, index=[ak, bk])          # batch oracle
    ga = ((float(v), int(k)) for v, k in zip(av, ak))
    gb = ((float(v), int(k)) for v, k in zip(bv, bk))
    out = _Merge()(ga, gb)
    assert hasattr(out, "__next__") and not isinstance(out, tuple)
    rows = list(out)                                           # [(value, index, source), ...]
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(mvals))
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(midx))
    np.testing.assert_array_equal([r[2] for r in rows], np.asarray(msrc))


def test_merge_lazy_positional_equals_batch_unequal_lengths():
    import numpy as np
    from screamer.streams import Merge as _Merge
    a = [1.0, 2.0, 3.0]
    b = [10.0, 20.0]
    mvals, msrc, midx = _Merge()(np.array(a), np.array(b))        # positional, index None
    rows = list(_Merge()((x for x in a), (x for x in b)))         # bare-value sources
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(mvals))
    assert all(r[1] is None for r in rows)                     # positional -> None index
    np.testing.assert_array_equal([r[2] for r in rows], np.asarray(msrc))
    assert midx is None
