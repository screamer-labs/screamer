import numpy as np
import pytest
from screamer import streams
from screamer.streams import Stream


# ---------------------------------------------------------------------------
# dropna – raw array
# ---------------------------------------------------------------------------

def test_dropna_any_on_aligned():
    vals = np.array([[1.0, 10.0],
                     [np.nan, 20.0],
                     [3.0, np.nan],
                     [4.0, 40.0]])
    idx = np.array([1, 2, 3, 4], dtype=np.int64)
    gv, gi = streams.Dropna()(vals, index=idx)          # how="any" default
    np.testing.assert_array_equal(gi, np.array([1, 4], dtype=np.int64))
    np.testing.assert_array_equal(gv, np.array([[1.0, 10.0], [4.0, 40.0]]))


def test_dropna_all_on_aligned():
    vals = np.array([[np.nan, np.nan],
                     [np.nan, 2.0],
                     [3.0, 3.0]])
    idx = np.array([1, 2, 3], dtype=np.int64)
    gv, gi = streams.Dropna(how="all")(vals, index=idx)
    np.testing.assert_array_equal(gi, np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(gv, np.array([[np.nan, 2.0], [3.0, 3.0]]))


def test_dropna_1d():
    vals = np.array([1.0, np.nan, 3.0])
    idx = np.array([1, 2, 3], dtype=np.int64)
    gv, gi = streams.Dropna()(vals, index=idx)
    np.testing.assert_array_equal(gi, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(gv, np.array([1.0, 3.0]))


def test_dropna_positional_index_is_none():
    """index=None -> positional; returned index is None (no allocation)."""
    vals = np.array([1.0, np.nan, 3.0])
    gv, gi = streams.Dropna()(vals)
    assert gi is None
    np.testing.assert_array_equal(gv, np.array([1.0, 3.0]))


# ---------------------------------------------------------------------------
# dropna – Stream / Node mirroring
# ---------------------------------------------------------------------------

def test_dropna_stream_in_stream_out():
    s = Stream(np.array([1.0, np.nan, 3.0]), np.array([10, 20, 30], dtype=np.int64))
    out = streams.Dropna()(s)
    assert isinstance(out, Stream)
    np.testing.assert_array_equal(out.values, [1.0, 3.0])
    np.testing.assert_array_equal(out.index, [10, 30])


def test_dropna_stream_positional():
    s = Stream(np.array([1.0, np.nan, 3.0]))      # positional stream
    out = streams.Dropna()(s)
    assert isinstance(out, Stream)
    assert out.index is None
    np.testing.assert_array_equal(out.values, [1.0, 3.0])


def test_dropna_node_in_node_out():
    from screamer import Input
    from screamer.dag import is_node
    x = Input("x")
    assert is_node(streams.Dropna()(x))


# ---------------------------------------------------------------------------
# lazy operator dispatch - existing streaming tests (now via operator)
# ---------------------------------------------------------------------------

def test_dropna_iter_matches_batch():
    vals = np.array([1.0, np.nan, 3.0])
    idx = np.array([1, 2, 3], dtype=np.int64)
    gen = ((float(v), int(k)) for v, k in zip(vals, idx))
    got = list(streams.Dropna()(gen))
    assert got == [(1.0, 1), (3.0, 3)]


def test_dropna_iter_positional_yields_none_index():
    events = iter([(1.0, None), (float("nan"), None), (3.0, None)])
    got = list(streams.Dropna()(events))
    assert got == [(1.0, None), (3.0, None)]


def test_dropna_iter_2d_rows():
    events = iter([((1.0, 10.0), 1), ((np.nan, 20.0), 2), ((3.0, 30.0), 3)])
    got_any = list(streams.Dropna(how="any")(events))
    assert [idx for _, idx in got_any] == [1, 3]
    events2 = iter([((1.0, 10.0), 1), ((np.nan, 20.0), 2), ((3.0, 30.0), 3)])
    got_all = list(streams.Dropna(how="all")(events2))
    assert [idx for _, idx in got_all] == [1, 2, 3]    # no row is all-NaN


# ---------------------------------------------------------------------------
# split / merge (unchanged API – sanity checks kept)
# ---------------------------------------------------------------------------

def test_split_inverts_merge():
    a_k = np.array([1, 3, 5], dtype=np.int64)
    a_v = np.array([10.0, 30.0, 50.0])
    b_k = np.array([2, 4], dtype=np.int64)
    b_v = np.array([20.0, 40.0])
    mv, ms, mk = streams.Merge()(a_v, b_v, index=[a_k, b_k])   # values-first result
    parts = streams.split(mv, ms, index=mk)                    # split(values, sources, index=)
    assert len(parts) == 2
    np.testing.assert_array_equal(parts[0][0], a_v)   # parts[i] = (values, index)
    np.testing.assert_array_equal(parts[0][1], a_k)
    np.testing.assert_array_equal(parts[1][0], b_v)
    np.testing.assert_array_equal(parts[1][1], b_k)


def test_split_explicit_n_includes_empty():
    keys = np.array([1, 2], dtype=np.int64)
    vals = np.array([1.0, 2.0])
    src = np.array([0, 0], dtype=np.uint32)
    parts = streams.split(vals, src, index=keys, n=3)   # split(values, sources, index=, n=)
    assert len(parts) == 3
    assert parts[1][0].size == 0 and parts[2][0].size == 0   # parts[i][0]=values are empty


def test_split_rejects_too_small_n():
    keys = np.array([1, 2, 3], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    src = np.array([0, 1, 2], dtype=np.uint32)     # needs n >= 3
    with pytest.raises(ValueError):
        streams.split(vals, src, index=keys, n=2)   # would drop source 2 silently


# ---------------------------------------------------------------------------
# lazy dispatch - oracle tests (batch == streaming)
# ---------------------------------------------------------------------------

def test_dropna_lazy_equals_batch():
    import numpy as np
    from screamer.streams import Dropna as _Dropna
    vals = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    idx = np.array([10, 11, 12, 13, 14])
    bv, bk = _Dropna()(vals, idx)                          # raw -> (values, index)
    gen = ((float(v), int(k)) for v, k in zip(vals, idx))
    out = _Dropna()(gen)
    assert hasattr(out, "__next__") and not isinstance(out, tuple)
    rows = list(out)
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(bv))
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bk))


# ---------------------------------------------------------------------------
# lazy dispatch - laziness test (no pull until next())
# ---------------------------------------------------------------------------

def test_dropna_lazy_is_lazy():
    from screamer.streams import Dropna as _Dropna
    pulled = []

    def spy():
        for i, v in enumerate([1.0, 2.0, 3.0]):
            pulled.append(v)
            yield (v, i)

    it = _Dropna()(spy())
    assert pulled == []
    first = next(it)
    assert pulled == [1.0]
    assert first == (1.0, 0)
