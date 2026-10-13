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
    with pytest.raises(ValueError, match="not supported as a Pipeline graph node"):
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


@pytest.mark.parametrize("sources", [
    [0, 2, 1, 0], (0, 2, 1, 0), np.array([0, 2, 1, 0], dtype=np.uint8),
    np.array([0, 2, 1, 0], dtype=np.int32), np.array([0, 2, 1, 0], dtype=np.uint32),
    np.array([0, 2, 1, 0], dtype=np.uint64), [0.0, 2.0, 1.0, 0.0],
    np.array([0, 2, 1, 0], dtype=np.float32), np.array([0, 2, 1, 0], dtype=">i4"),
    list(np.array([0, 2, 1, 0])), list(np.array([0, 2, 1, 0], dtype=np.float32)),
    np.array([0, 2, 1, 0], dtype=object), np.array([0, 2, 1, 0])[::-1][::-1],
])
def test_split_sources_every_numeric_kind(sources):
    parts = streams.split(np.array([1.0, 2.0, 3.0, 4.0]), sources)
    assert [list(v) for v, _ in parts] == [[1.0, 4.0], [3.0], [2.0]]


@pytest.mark.parametrize("sources", [
    np.array([True, False]), [True, False], [True, 1], [1, np.True_],
    np.array([True, 1], dtype=object),
])
def test_split_bool_sources_raise(sources):
    with pytest.raises(TypeError, match="split: a source must be a whole number, got a bool$"):
        streams.split(np.array([1.0, 2.0]), sources)


@pytest.mark.parametrize("sources, text", [
    ([0.5, 0.0], "0.5"), ([np.nan, 0.0], "NaN"), (np.array([np.nan, 0.0]), "NaN"),
    ([None, 0], "None"), ([np.inf, 0.0], "inf"), (np.array([0.5, 0.0], dtype=np.float32), "0.5"),
    (["0", 1], "a str"),
])
def test_split_source_kind_rule(sources, text):
    with pytest.raises(TypeError, match=f"split: a source must be a whole number, got {text}$"):
        streams.split(np.array([1.0, 2.0]), sources)


@pytest.mark.parametrize("sources, text", [
    ([-1, 0], "-1"), (np.array([-1, 0]), "-1"), ([-1.0, 0.0], "-1"),
    ([2**31 - 1, 0], "2147483647"), ([2**32, 0], "4294967296"),
    (np.array([2**40, 0], dtype=np.int64), "1099511627776"),
    (np.array([2**40, 0], dtype=np.uint64), "1099511627776"),
    (np.array([2**31 - 1, 0], dtype=np.uint32), "2147483647"),
    (np.array([-1, 0], dtype=np.int8), "-1"), ([2**70, 0], str(2**70)),
])
def test_split_source_range_rule(sources, text):
    with pytest.raises(ValueError, match=f"split: a source must be from 0 to 2147483646, got {text}$"):
        streams.split(np.array([1.0, 2.0]), sources)


def test_split_largest_source():
    # the largest source needs 2**31 - 1 outputs; checked without building them
    with pytest.raises(ValueError, match="n=1 is too small for sources up to 2147483646"):
        streams.split(np.array([1.0]), [2**31 - 2], n=1)


def test_split_n_rules():
    v, src = np.array([1.0, 2.0, 3.0]), [0, 3, 1]
    with pytest.raises(ValueError, match="n=2 is too small for sources up to 3"):
        streams.split(v, src, n=2)
    with pytest.raises(ValueError, match="n must be non-negative"):
        streams.split(v[:0], [], n=-1)
    with pytest.raises(ValueError, match="at most 2\\*\\*31 - 1"):
        streams.split(v[:0], [], n=2**31)
    with pytest.raises(ValueError, match="at most 2\\*\\*31 - 1"):
        streams.split(v[:0], [], n=2**70)
    with pytest.raises(TypeError, match="n must be a number"):
        streams.split(v, src, n="4")
    for bad, text in [(2.0, "2.0"), (1.5, "1.5"), (np.float64(4.0), "np.float64\\(4.0\\)")]:
        with pytest.raises(TypeError, match=f"^split: n must be an integer, got {text}$"):
            streams.split(v, src, n=bad)
    assert len(streams.split(v, src, n=6)) == 6
    assert len(streams.split(v, src, n=np.int64(5))) == 5
    for flag in (True, np.True_):          # a bool is not a number setting (G25)
        with pytest.raises(TypeError, match="split: n must be a number, got bool"):
            streams.split(v[:1], [0], n=flag)


def test_split_shape_rules():
    with pytest.raises(ValueError, match="sources must be 1-D"):
        streams.split(np.zeros(2), np.zeros((2, 1), dtype=np.int64))
    with pytest.raises(ValueError, match="sources must be 1-D"):
        streams.split(np.zeros(2), [[0], [1]])
    with pytest.raises(ValueError, match="index must be 1-D"):
        streams.split(np.zeros(2), [0, 1], index=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="length differ"):
        streams.split(np.zeros(2), [0])
    with pytest.raises(ValueError, match="length differ"):
        streams.split(np.zeros((2, 3)), [0, 1, 0])
    with pytest.raises(ValueError, match="length differ"):
        streams.split(np.zeros(2), [0, 1], index=[1])
    with pytest.raises(TypeError):
        streams.split(np.zeros(2), [0, 1], index=["a", "b"])


@pytest.mark.parametrize("shape", [(6, 3), (6, 1), (6, 2, 2), (6, 0), (0, 3)])
def test_split_rows(shape):
    T = shape[0]
    values = np.arange(float(np.prod(shape))).reshape(shape)
    src = np.arange(T) % 3
    idx = np.arange(T) * 10
    parts = streams.split(values, src, index=idx, n=4)
    assert len(parts) == 4
    for s, (v, k) in enumerate(parts):
        assert v.shape == (int((src == s).sum()),) + shape[1:]
        np.testing.assert_array_equal(v, values[src == s])
        np.testing.assert_array_equal(k, idx[src == s])
    # each row matches a 1-D split of one column
    if shape[1:] and shape[1] and T:
        col = streams.split(values[:, 0], src)
        for (v, _), (c, _) in zip(parts, col):
            np.testing.assert_array_equal(v[:, 0], c)


def test_split_non_contiguous():
    values = np.arange(20.0)[::2]
    src = (np.arange(20) % 3)[::2]
    idx = (np.arange(20) * 1.5)[::2]
    parts = streams.split(values, src, index=idx)
    for s, (v, k) in enumerate(parts):
        np.testing.assert_array_equal(v, values[src == s])
        np.testing.assert_array_equal(k, idx[src == s])


def test_split_index_kinds_kept():
    v, src = np.array([1.0, 2.0, 3.0]), [0, 1, 0]
    (_, k0), _ = streams.split(v, src, index=np.array([1.5, 2.5, np.nan]))
    assert k0.dtype == np.float64 and k0[0] == 1.5 and np.isnan(k0[1])
    (_, k0), _ = streams.split(v, src, index=np.array([1, 2, 3], dtype=np.int32))
    assert k0.dtype == np.int64 and list(k0) == [1, 3]
    (_, k0), _ = streams.split(v, src, index=np.array([1, 2, 3], dtype="datetime64[s]"))
    assert k0.dtype == np.int64 and list(k0) == [1, 3]


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
