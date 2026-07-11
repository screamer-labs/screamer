"""Oracle tests for merge lazy path: byte-identical output before and after
routing through the C++ MergeLazyPuller.

Oracle output captured from the original Python _merge_lazy implementation.
These tests must pass both before and after the C++ migration.
"""
import numpy as np
import pytest
from screamer import streams


def _collect(gen):
    """Drain a lazy merge generator into a list of (value, index_or_None, source)."""
    return list(gen)


# ---------------------------------------------------------------------------
# Case 1: indexed sources with interleaved indices
# ---------------------------------------------------------------------------

def test_oracle_indexed_interleaved():
    a_v, a_k = [1.0, 3.0], [0, 2]
    b_v, b_k = [2.0, 4.0], [1, 3]
    ga = ((v, k) for v, k in zip(a_v, a_k))
    gb = ((v, k) for v, k in zip(b_v, b_k))
    events = _collect(streams.Merge()(ga, gb))
    assert events == [(1.0, 0, 0), (2.0, 1, 1), (3.0, 2, 0), (4.0, 3, 1)]


# ---------------------------------------------------------------------------
# Case 2: ties - equal index across sources -> lower source index wins
# All three events at index 0; source 0 has two events (both come first).
# ---------------------------------------------------------------------------

def test_oracle_ties_lower_source_wins():
    ga = ((v, k) for v, k in zip([1.0, 2.0], [0, 0]))
    gb = ((v, k) for v, k in zip([10.0], [0]))
    events = _collect(streams.Merge()(ga, gb))
    assert events == [(1.0, 0, 0), (2.0, 0, 0), (10.0, 0, 1)]


# ---------------------------------------------------------------------------
# Case 3: one source exhausts before the other
# ---------------------------------------------------------------------------

def test_oracle_early_exhaust():
    ga = ((v, k) for v, k in zip([1.0], [0]))
    gb = ((v, k) for v, k in zip([2.0, 3.0], [1, 2]))
    events = _collect(streams.Merge()(ga, gb))
    assert events == [(1.0, 0, 0), (2.0, 1, 1), (3.0, 2, 1)]


# ---------------------------------------------------------------------------
# Case 4: positional bare-value generators, UNEQUAL lengths; index must be None
# ---------------------------------------------------------------------------

def test_oracle_positional_unequal_none_index():
    ga = (x for x in [1.0, 2.0, 3.0])
    gb = (x for x in [10.0, 20.0])
    events = _collect(streams.Merge()(ga, gb))
    # values interleaved by row-number; all indices None
    assert events == [(1.0, None, 0), (10.0, None, 1),
                      (2.0, None, 0), (20.0, None, 1),
                      (3.0, None, 0)]
    for _, idx, _ in events:
        assert idx is None


# ---------------------------------------------------------------------------
# Case 5: single source
# ---------------------------------------------------------------------------

def test_oracle_single_source():
    ga = ((v, k) for v, k in zip([1.0, 2.0], [5, 10]))
    events = _collect(streams.Merge()(ga))
    assert events == [(1.0, 5, 0), (2.0, 10, 0)]


# ---------------------------------------------------------------------------
# Case 6: empty source (yields nothing)
# ---------------------------------------------------------------------------

def test_oracle_empty_source():
    ga = ((v, k) for v, k in zip([], []))
    events = _collect(streams.Merge()(ga))
    assert events == []


def test_oracle_one_empty_one_nonempty():
    ga = ((v, k) for v, k in zip([], []))
    gb = ((v, k) for v, k in zip([5.0, 6.0], [1, 2]))
    events = _collect(streams.Merge()(ga, gb))
    assert events == [(5.0, 1, 1), (6.0, 2, 1)]


# ---------------------------------------------------------------------------
# Case 7: float indices preserved in output
# ---------------------------------------------------------------------------

def test_oracle_float_index_preserved():
    ga = ((v, k) for v, k in zip([1.0, 2.0], [0.5, 1.5]))
    gb = ((v, k) for v, k in zip([3.0], [1.0]))
    events = _collect(streams.Merge()(ga, gb))
    assert events == [(1.0, 0.5, 0), (3.0, 1.0, 1), (2.0, 1.5, 0)]
    for _, idx, _ in events:
        assert isinstance(idx, float)


# ---------------------------------------------------------------------------
# Laziness: before first next(), NO source is consumed.
# After first next(), exactly one head from each source is consumed.
# ---------------------------------------------------------------------------

def test_laziness_no_consumption_before_next():
    consumed = {"a": 0, "b": 0}

    def spy(name, items):
        for i, v in enumerate(items):
            consumed[name] += 1
            yield (float(v), i)

    it = streams.Merge()(spy("a", [1.0, 2.0]), spy("b", [10.0, 20.0]))
    assert consumed == {"a": 0, "b": 0}, "Nothing consumed before first next()"
    next(it)
    assert consumed["a"] == 1 and consumed["b"] == 1, "One head per source after first next()"


# ---------------------------------------------------------------------------
# Matches batch merge exactly (large random test)
# ---------------------------------------------------------------------------

def test_lazy_indexed_matches_batch_int():
    rng = np.random.default_rng(42)
    a_k = np.sort(rng.integers(0, 500, size=100)).astype(np.int64)
    a_v = rng.standard_normal(100)
    b_k = np.sort(rng.integers(0, 500, size=80)).astype(np.int64)
    b_v = rng.standard_normal(80)

    bv, bs, bi = streams.Merge()(a_v, b_v, index=[a_k, b_k])
    ga = ((float(v), int(k)) for v, k in zip(a_v, a_k))
    gb = ((float(v), int(k)) for v, k in zip(b_v, b_k))
    events = _collect(streams.Merge()(ga, gb))

    np.testing.assert_array_equal([e[0] for e in events], bv)
    np.testing.assert_array_equal([e[1] for e in events], bi)
    np.testing.assert_array_equal([e[2] for e in events], bs)


def test_lazy_positional_matches_batch():
    rng = np.random.default_rng(99)
    a_v = rng.standard_normal(50)
    b_v = rng.standard_normal(37)   # unequal length

    bv, bs, bi = streams.Merge()(a_v, b_v)
    assert bi is None
    events = _collect(streams.Merge()((x for x in a_v), (x for x in b_v)))

    np.testing.assert_allclose([e[0] for e in events], bv)
    assert all(e[1] is None for e in events)
    np.testing.assert_array_equal([e[2] for e in events], bs)


# ---------------------------------------------------------------------------
# Three-source merge
# ---------------------------------------------------------------------------

def test_three_sources_indexed():
    rng = np.random.default_rng(7)
    series = [(np.sort(rng.integers(0, 200, size=40)).astype(np.int64),
               rng.standard_normal(40)) for _ in range(3)]
    bv, bs, bi = streams.Merge()(*[v for _, v in series],
                               index=[k for k, _ in series])
    gens = [((float(v), int(k)) for v, k in zip(vals, idxs))
            for idxs, vals in series]
    events = _collect(streams.Merge()(*gens))
    np.testing.assert_array_equal([e[0] for e in events], bv)
    np.testing.assert_array_equal([e[1] for e in events], bi)
    np.testing.assert_array_equal([e[2] for e in events], bs)
