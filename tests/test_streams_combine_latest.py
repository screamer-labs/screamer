import numpy as np
import pytest
from screamer.streams import combine_latest, Stream


def test_positional_lockstep_equal_length():
    v, idx = combine_latest(np.array([10.0, 20.0, 40.0]), np.array([1.0, 3.0, 4.0]))
    assert idx is None
    assert v.shape == (3, 2)
    np.testing.assert_array_equal(v[:, 0], [10.0, 20.0, 40.0])


def test_positional_unequal_length_raises():
    with pytest.raises(ValueError, match="length"):
        combine_latest(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_indexed_coalesces_same_index_rows():
    a = np.array([10.0, 20.0, 40.0]); ta = np.array([1, 2, 4])
    b = np.array([1.0, 3.0, 4.0]);    tb = np.array([1, 3, 4])
    v, idx = combine_latest(a, b, index=[ta, tb])
    np.testing.assert_array_equal(idx, [1, 2, 3, 4])          # NOT [1,2,3,4,4]
    np.testing.assert_array_equal(v, [[10, 1], [20, 1], [20, 3], [40, 4]])


def test_stream_in_stream_out():
    a = Stream(np.array([10.0, 20.0]), np.array([1, 2]))
    b = Stream(np.array([1.0, 2.0]), np.array([1, 2]))
    out = combine_latest(a, b)
    assert isinstance(out, Stream)
    np.testing.assert_array_equal(out.index, [1, 2])


def test_mixed_positional_and_indexed_raises():
    with pytest.raises(ValueError, match="positional"):
        combine_latest(Stream(np.array([1.0, 2.0]), np.array([1, 2])),
                       Stream(np.array([3.0, 4.0])))


def test_node_in_node_out():
    from screamer import Input
    from screamer.dag import is_node
    assert is_node(combine_latest(Input("a"), Input("b")))


def test_indexed_lazy_coalesces_and_yields_row_index():
    """Previously tested combine_latest_iter; now uses lazy generator API."""
    ga = ((float(v), int(k)) for v, k in zip([10.0, 20.0], [1, 2]))
    gb = ((float(v), int(k)) for v, k in zip([1.0, 2.0], [1, 2]))
    events = list(combine_latest(ga, gb))
    idxs = [i for _, i in events]
    assert idxs == [1, 2]
    np.testing.assert_array_equal(np.asarray(events[0][0]), [10.0, 1.0])


def test_combine_latest_lazy_indexed_equals_batch():
    from screamer.streams import combine_latest
    av, ak = np.array([10.0, 20.0, 30.0]), np.array([1, 2, 4])
    bv, bk = np.array([1.0, 2.0, 3.0]),   np.array([1, 3, 4])
    brows, bidx = combine_latest(av, bv, index=[ak, bk])       # batch oracle
    ga = ((float(v), int(k)) for v, k in zip(av, ak))
    gb = ((float(v), int(k)) for v, k in zip(bv, bk))
    out = combine_latest(ga, gb)
    assert hasattr(out, "__next__") and not isinstance(out, tuple)
    rows = list(out)                                            # [((a,b), index), ...]
    np.testing.assert_allclose([list(r[0]) for r in rows], np.asarray(brows))
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bidx))


def test_combine_latest_lazy_positional_equals_batch():
    from screamer.streams import combine_latest
    a = [10.0, 20.0, 30.0]
    b = [1.0, 2.0, 3.0]
    brows, bidx = combine_latest(np.array(a), np.array(b))      # aligned clocks, index None
    out = list(combine_latest((x for x in a), (x for x in b)))  # bare-value sources
    np.testing.assert_allclose([list(r[0]) for r in out], np.asarray(brows))
    assert all(r[1] is None for r in out)                       # positional -> None index
    assert bidx is None


def test_combine_latest_lazy_positional_unequal_raises():
    from screamer.streams import combine_latest
    out = combine_latest((x for x in [1.0, 2.0, 3.0]), (x for x in [10.0, 20.0]))
    with pytest.raises(ValueError):
        list(out)                                              # error surfaces at exhaustion


def test_combine_latest_lazy_mixed_sources_raise():
    from screamer.streams import combine_latest
    # mix positional (bare) and indexed ((v,k)) lazy sources -> ValueError
    with pytest.raises(ValueError):
        list(combine_latest((x for x in [1.0, 2.0]),
                            ((float(v), int(k)) for v, k in [(1.0, 0), (2.0, 1)])))
    # mix lazy and concrete -> TypeError
    with pytest.raises(TypeError):
        combine_latest((x for x in [1.0, 2.0]), np.array([1.0, 2.0]))


def test_combine_latest_lazy_is_lazy():
    """Construction pulls nothing; the first next() pulls one head per source."""
    from screamer.streams import combine_latest
    pulled = {"a": [], "b": []}

    def spy(name, items):
        for v in items:
            pulled[name].append(v)
            yield float(v)   # bare scalar -> positional source

    it = combine_latest(spy("a", [1.0, 2.0]), spy("b", [10.0, 20.0]))
    assert pulled == {"a": [], "b": []}      # nothing before first next()
    next(it)
    assert pulled["a"] == [1.0] and pulled["b"] == [10.0]   # one head per source
