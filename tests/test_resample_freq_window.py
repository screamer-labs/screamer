"""freq= is always a WINDOW (span over the index), in every regime (Option B).

Regression test for the graph/lazy gap: freq= must resolve to a span against the
RUNTIME index, not collapse to count when no index is present at definition time
(a Node or lazy stream has index=None at build time). On a sparse index, a span
and an arrival count differ, so this pins the correct (span) behavior.
"""
import numpy as np
from screamer import Resample, Input, Dag, ExpandingSum


# Sparse index where a span of 10 (3 buckets) differs from count=10 (1 bucket).
_IDX = np.array([0, 3, 10, 25], dtype=np.int64)
_X = np.array([1.0, 1.0, 1.0, 1.0])
# Eager span=10 oracle: buckets [0,10) has 2 ticks, [10,20) has 1, [20,30) has 1.
_EXP_V = [2.0, 1.0, 1.0]
_EXP_K = [0, 10, 20]


def _vi(r):
    return (np.asarray(r[0]), np.asarray(r[1]))


def test_eager_freq_is_span():
    v, k = _vi(Resample(freq=10, agg="sum")(_X, _IDX))
    np.testing.assert_array_equal(v, _EXP_V)
    np.testing.assert_array_equal(k, _EXP_K)


def test_node_freq_is_span_not_count():
    s = Input("x")
    dag = Dag([s], [Resample(freq=10, agg=ExpandingSum())(s)])
    v, k = _vi(dag((_X, _IDX)))
    np.testing.assert_array_equal(v, _EXP_V)
    np.testing.assert_array_equal(k, _EXP_K)


def test_lazy_freq_is_span_not_count():
    lz = list(Resample(freq=10, agg=ExpandingSum())(iter(list(zip(_X, _IDX)))))
    np.testing.assert_array_equal([v for v, _ in lz], _EXP_V)
    np.testing.assert_array_equal([k for _, k in lz], _EXP_K)


def test_count_is_arrival_distinct_from_freq_on_sparse_index():
    # count=10 is arrival (every 10 events): only 4 events -> a single partial bar.
    v, k = _vi(Resample(count=10, agg="sum")(_X, _IDX))
    assert len(v) == 1 and v[0] == 4.0        # all 4 ticks in one arrival bar
    # freq=10 (span) gives 3 buckets - the two modes are genuinely different.
    assert len(_EXP_V) == 3


def test_positional_freq_right_label_is_grid_edge():
    """freq= is a window in EVERY regime, so a positional stream with
    label="right" grid-edge-labels (like the indexed regime), NOT actual-tick.
    This is the intended consequence of the freq=window thesis; pin it so the
    behavior is documented and cannot silently drift back."""
    x = np.arange(10.0)
    v, k = _vi(Resample(freq=3, agg="sum", label="right")(x))
    np.testing.assert_array_equal(v, [3.0, 12.0, 21.0, 9.0])   # values unchanged
    np.testing.assert_array_equal(k, [3, 6, 9, 12])            # grid edges (origin+n*W)
