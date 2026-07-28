"""Tests for Resample clock= target-clock resample (Task 5a).

A bucket closes at each event of the ``clock`` stream. The value stream is
aggregated over (previous clock tick, this clock tick] (inclusive of the tick)
and emitted labelled at the clock tick.

All tests follow TDD: written first to fail, then green after implementation.
"""
import numpy as np
import pytest
from screamer import Input, Pipeline
from screamer.streams import Resample
from screamer import RollingMean  # a real EvalOp for functor-agg tests (C2)
from tests._dag_oracle import lazy_batch as _lazy_batch


# ---------------------------------------------------------------------------
# Core semantics: as-of / last-value carry
# ---------------------------------------------------------------------------

def test_clock_asof_basic():
    """As-of: for each clock tick, emit the last value event with index <= tick."""
    # value ticks at  0,  2,  4,  6
    # clock ticks at  1,  3,  5
    # bucket (prev, tick]:
    #   tick 1: value events with idx in (None, 1] -> {0} -> last = 10
    #   tick 3: value events with idx in (1, 3]    -> {2} -> last = 20
    #   tick 5: value events with idx in (3, 5]    -> {4} -> last = 30
    value_v = np.array([10., 20., 30., 40.])
    value_i = np.array([0, 2, 4, 6], dtype=np.int64)
    clock_v = np.zeros(3)          # clock values are ignored
    clock_i = np.array([1, 3, 5], dtype=np.int64)

    out, idx = Resample((value_v, value_i), (clock_v, clock_i), clock=True,
                        agg='last', fill='carry')
    np.testing.assert_array_equal(idx, [1, 3, 5])
    np.testing.assert_allclose(out, [10., 20., 30.])


def test_clock_fill_carry_when_no_value_in_bucket():
    """fill='carry': if no value arrived in a bucket, repeat the last emitted value."""
    # value ticks at 0,        6
    # clock ticks at    2,  4,    8
    # bucket (prev, tick]:
    #   tick 2: value events {0} -> last = 1.0
    #   tick 4: no value events -> carry -> 1.0
    #   tick 8: value events {6} -> last = 2.0
    value_v = np.array([1., 2.])
    value_i = np.array([0, 6], dtype=np.int64)
    clock_v = np.zeros(3)
    clock_i = np.array([2, 4, 8], dtype=np.int64)

    out, idx = Resample((value_v, value_i), (clock_v, clock_i), clock=True,
                        agg='last', fill='carry')
    np.testing.assert_array_equal(idx, [2, 4, 8])
    np.testing.assert_allclose(out, [1., 1., 2.])   # tick 4 carries 1.0


def test_clock_fill_skip_when_no_value_in_bucket():
    """fill='skip' (default): if no value arrived, no row emitted for that clock tick."""
    value_v = np.array([1., 2.])
    value_i = np.array([0, 6], dtype=np.int64)
    clock_v = np.zeros(3)
    clock_i = np.array([2, 4, 8], dtype=np.int64)

    out, idx = Resample((value_v, value_i), (clock_v, clock_i), clock=True,
                        agg='last', fill='skip')
    # tick 4 has no value -> skipped; tick 2 and tick 8 have values
    np.testing.assert_array_equal(idx, [2, 8])
    np.testing.assert_allclose(out, [1., 2.])


def test_clock_fill_nan_when_no_value_in_bucket():
    """fill='nan': emit NaN at clock ticks with no value events."""
    value_v = np.array([1., 2.])
    value_i = np.array([0, 6], dtype=np.int64)
    clock_v = np.zeros(3)
    clock_i = np.array([2, 4, 8], dtype=np.int64)

    out, idx = Resample((value_v, value_i), (clock_v, clock_i), clock=True,
                        agg='last', fill='nan')
    np.testing.assert_array_equal(idx, [2, 4, 8])
    np.testing.assert_allclose(out[0], 1.)
    assert np.isnan(out[1]), "tick 4 has no value; fill='nan' should emit NaN"
    np.testing.assert_allclose(out[2], 2.)


def test_clock_tie_value_index_equals_clock_tick_is_included():
    """A value event at exactly the clock tick index is INCLUDED in that bucket."""
    # value at index 5, clock tick at index 5 -> value IS in this bucket
    value_v = np.array([99.])
    value_i = np.array([5], dtype=np.int64)
    clock_v = np.zeros(2)
    clock_i = np.array([3, 5], dtype=np.int64)

    out, idx = Resample((value_v, value_i), (clock_v, clock_i), clock=True,
                        agg='last', fill='carry')
    # tick 3: no value (value is at 5, not <= 3 after prev tick 0)
    # tick 5: value at 5 is <= 5 and > 3 -> included -> last = 99
    np.testing.assert_array_equal(idx, [5])   # tick 3 has no value -> carry but no prev -> skip
    np.testing.assert_allclose(out, [99.])


def test_clock_values_before_first_tick_go_in_first_bucket():
    """Value events before the first clock tick are included in the first bucket."""
    # value at 0, 1, 2 (all before clock tick at 5)
    value_v = np.array([10., 20., 30.])
    value_i = np.array([0, 1, 2], dtype=np.int64)
    clock_v = np.zeros(1)
    clock_i = np.array([5], dtype=np.int64)

    out, idx = Resample((value_v, value_i), (clock_v, clock_i), clock=True,
                        agg='last', fill='carry')
    np.testing.assert_array_equal(idx, [5])
    np.testing.assert_allclose(out, [30.])   # last value before/at tick 5


def test_clock_sum_agg_between_ticks():
    """agg='sum' sums all values between clock ticks."""
    # value ticks: idx 1->5, 2->7, 4->3
    # clock ticks: idx 3, 6
    # bucket (None, 3]: {1, 2} -> 5+7=12
    # bucket (3,   6]: {4}    -> 3
    value_v = np.array([5., 7., 3.])
    value_i = np.array([1, 2, 4], dtype=np.int64)
    clock_v = np.zeros(2)
    clock_i = np.array([3, 6], dtype=np.int64)

    out, idx = Resample((value_v, value_i), (clock_v, clock_i), clock=True,
                        agg='sum', fill='skip')
    np.testing.assert_array_equal(idx, [3, 6])
    np.testing.assert_allclose(out, [12., 3.])


def test_clock_multicol_value():
    """Multi-column value input: last across all columns, each reduced independently."""
    # 2-col value: col0 = price, col1 = vol
    value_v = np.array([[10., 100.],
                        [20., 200.],
                        [30., 300.]])
    value_i = np.array([0, 2, 4], dtype=np.int64)
    clock_v = np.zeros(2)
    clock_i = np.array([2, 5], dtype=np.int64)

    # Use ohlcv here? No, let's test with explicit multi-column + agg='last'
    # Actually we need a plan-based multi-column reduce. Let's use agg='last'
    # on a combined 2-col wide input.
    # For now test with sum to keep it simple:
    out, idx = Resample((value_v, value_i), (clock_v, clock_i), clock=True,
                        agg='last', fill='carry')
    np.testing.assert_array_equal(idx, [2, 5])
    np.testing.assert_allclose(out[0], [20., 200.])  # last up to tick 2
    np.testing.assert_allclose(out[1], [30., 300.])  # last up to tick 5


def test_clock_empty_clock_returns_empty():
    """Empty clock stream: no output."""
    value_v = np.array([1., 2., 3.])
    value_i = np.array([0, 1, 2], dtype=np.int64)
    clock_v = np.zeros(0)
    clock_i = np.zeros(0, dtype=np.int64)

    out, idx = Resample((value_v, value_i), (clock_v, clock_i), clock=True,
                        agg='last', fill='carry')
    assert len(out) == 0
    assert len(idx) == 0


# ---------------------------------------------------------------------------
# Edge rejections
# ---------------------------------------------------------------------------

def test_clock_conflicts_with_freq():
    """clock= with freq= must raise ValueError."""
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), (np.ones(3), np.arange(3)),
                 clock=True, freq=5, agg='last')


def test_clock_conflicts_with_every():
    """clock= with every= must raise ValueError."""
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), (np.ones(3), np.arange(3)),
                 clock=True, every=5, agg='last')


def test_clock_conflicts_with_count():
    """clock= with count= must raise ValueError."""
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), (np.ones(3), np.arange(3)),
                 clock=True, count=5, agg='last')


def test_clock_conflicts_with_threshold():
    """clock= with threshold= must raise ValueError."""
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), (np.ones(3), np.arange(3)),
                 clock=True, threshold=5, agg='last')


def test_clock_true_without_clock_input_raises():
    """clock=True without a clock stream raises ValueError."""
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), clock=True, agg='last')


def test_clock_input_without_clock_mode_raises():
    """Providing a driver (2nd positional arg) without any mode raises ValueError."""
    # A second positional stream without threshold= or clock=True should raise
    # (the driver alone without a mode selector is ambiguous).
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), (np.ones(3), np.arange(3)), agg='last')


# ---------------------------------------------------------------------------
# All-regime test: eager == graph == lazy
# ---------------------------------------------------------------------------

def test_clock_mode_runs_in_all_regimes():
    """Crown-jewel: eager, graph, and lazy produce identical results for ByClock."""
    value_v = np.array([1., 3., 7., 5., 9., 2.])
    value_i = np.array([0, 1, 3, 5, 7, 8], dtype=np.int64)
    clock_v = np.zeros(4)
    clock_i = np.array([2, 4, 6, 10], dtype=np.int64)

    # eager
    eager_v, eager_k = Resample((value_v, value_i), (clock_v, clock_i),
                                 clock=True, agg='last', fill='carry')

    # graph regime
    x = Input('x')
    c = Input('c')
    node = Resample(x, c, clock=True, agg='last', fill='carry')
    dag = Pipeline([x, c], [node])
    graph_v, graph_k = dag((value_v, value_i), (clock_v, clock_i))

    # lazy regime
    lazy_v, lazy_k = _lazy_batch(dag, (value_v, value_i), (clock_v, clock_i))

    np.testing.assert_allclose(graph_v, eager_v, equal_nan=True,
                               err_msg="graph != eager for clock mode")
    np.testing.assert_array_equal(graph_k, eager_k,
                                  err_msg="graph index != eager index for clock mode")
    np.testing.assert_allclose(lazy_v, eager_v, equal_nan=True,
                               err_msg="lazy != eager for clock mode")
    np.testing.assert_array_equal(lazy_k, eager_k,
                                  err_msg="lazy index != eager index for clock mode")


# ---------------------------------------------------------------------------
# I2(a): C1 — clock-listed-first Pipeline gives IDENTICAL result to value-first
# ---------------------------------------------------------------------------

def test_clock_tie_pipeline_order_independent():
    """C1: Pipeline([c_in, x_in], ...) must give the same result as Pipeline([x_in, c_in], ...).

    The (prev_tick, tick] inclusive semantic requires a value event at the SAME
    index as a clock tick to land in that tick's bucket, regardless of which
    input is listed first in Pipeline([...]).
    """
    # value at idx 5 = 99; clock ticks at idx 3 and idx 5
    value_v = np.array([99.])
    value_i = np.array([5], dtype=np.int64)
    clock_v = np.zeros(2)
    clock_i = np.array([3, 5], dtype=np.int64)

    x = Input('x')
    c = Input('c')
    node_xc = Resample(x, c, clock=True, agg='last', fill='carry')
    node_cx = Resample(x, c, clock=True, agg='last', fill='carry')

    # value-first pipeline (previously correct)
    dag_xc = Pipeline([x, c], [node_xc])
    out_xc, idx_xc = dag_xc((value_v, value_i), (clock_v, clock_i))

    # clock-first pipeline (was silently wrong — clock fires before value buffered)
    dag_cx = Pipeline([c, x], [node_cx])
    out_cx, idx_cx = dag_cx((clock_v, clock_i), (value_v, value_i))

    # Both should see value 99 in the tick-5 bucket.
    np.testing.assert_array_equal(idx_xc, idx_cx,
                                   err_msg="indices differ by Pipeline input order")
    np.testing.assert_allclose(out_xc, out_cx,
                               err_msg="values differ by Pipeline input order — tie-order bug")


# ---------------------------------------------------------------------------
# I2(b): C2 — functor agg + clock= raises ValueError (no silent garbage)
# ---------------------------------------------------------------------------

def test_clock_functor_agg_raises():
    """C2: clock= with a non-string agg (EvalOp functor) must raise ValueError.

    Previously the ByClock/functor-agg combination silently routed to
    GenericResampleNode which ignores the clock= mode.
    """
    x = Input('x')
    c = Input('c')
    with pytest.raises(ValueError):
        Resample(x, c, clock=True, agg=RollingMean(5))


# ---------------------------------------------------------------------------
# I2(c): I1 — NaN-only value bucket behavior (ByClock + fill='carry')
# ---------------------------------------------------------------------------

def test_clock_nan_value_bucket_carries_with_fill_carry():
    """I1: with fill='carry', a bucket whose only value is NaN should carry
    the last real value (as-of semantics), not emit NaN.

    This tests the corrected ByClock behavior: a NaN value event should not
    count as 'has real data'. The accumulator's count (non-NaN entries) is
    used for carry decisions.
    """
    # value stream: real=10 at 0, NaN at 4, real=30 at 8
    # clock ticks at 2, 6, 10
    # bucket (none, 2]: real 10 -> emit 10
    # bucket (2,   6]: only NaN at 4 -> carry -> emit 10
    # bucket (6,  10]: real 30 -> emit 30
    value_v = np.array([10., float('nan'), 30.])
    value_i = np.array([0, 4, 8], dtype=np.int64)
    clock_v = np.zeros(3)
    clock_i = np.array([2, 6, 10], dtype=np.int64)

    out, idx = Resample((value_v, value_i), (clock_v, clock_i), clock=True,
                        agg='last', fill='carry')
    np.testing.assert_array_equal(idx, [2, 6, 10])
    np.testing.assert_allclose(out[0], 10.)
    # bucket (2,6] has only a NaN value: with fixed behavior it should carry 10
    assert not np.isnan(out[1]), (
        "NaN-only bucket with fill='carry' should carry last real value, got NaN. "
        "This is the I1 bug: has=True was set before the NaN check."
    )
    np.testing.assert_allclose(out[1], 10.)
    np.testing.assert_allclose(out[2], 30.)
