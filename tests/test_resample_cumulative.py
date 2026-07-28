"""Tests for Resample cumulative-driver mode (information bars).

Task 4: ByCumulative mode — a bar closes when the cumulative driver since the
last close reaches threshold, the crossing observation included, then the
accumulator resets to 0.

All tests follow TDD: written first to fail, then green after implementation.
"""
import numpy as np
import pytest
from screamer import Input, Pipeline
from screamer.streams import Resample
from tests._dag_oracle import lazy_batch as _lazy_batch


# ---------------------------------------------------------------------------
# Core semantics: volume bars close on threshold
# ---------------------------------------------------------------------------

def test_volume_bars_close_on_threshold():
    """A bar closes when cumulative driver >= threshold (crossing obs included)."""
    price  = np.array([10., 11, 12, 13, 14, 15])
    volume = np.array([ 4.,  3,  5,  2,  6,  1])
    # cumsum: 4, 7(>=5 -> close incl obs1), reset -> 5(>=5 -> close incl obs2), ...
    out, idx = Resample((price, np.arange(6)), volume, threshold=5, agg='ohlc')
    # first bar: obs 0..1 (cum 4->7 crosses 5), ohlc of [10, 11]
    np.testing.assert_allclose(out[0], [10, 11, 10, 11])   # O H L C


def test_volume_bars_second_bar():
    """Second bar closes at the single observation whose cum alone >= threshold."""
    price  = np.array([10., 11, 12, 13, 14, 15])
    volume = np.array([ 4.,  3,  5,  2,  6,  1])
    out, idx = Resample((price, np.arange(6)), volume, threshold=5, agg='ohlc')
    # second bar: obs 2 alone (cum resets after obs1, then 5>=5)
    np.testing.assert_allclose(out[1], [12, 12, 12, 12])


def test_volume_bars_produces_correct_bar_count():
    """Three complete bars from 6 obs (plus trailing partial)."""
    price  = np.array([10., 11, 12, 13, 14, 15])
    volume = np.array([ 4.,  3,  5,  2,  6,  1])
    out, idx = Resample((price, np.arange(6)), volume, threshold=5, agg='last')
    # bar 0: obs 0..1 (cum 7>=5) -> last=11
    # bar 1: obs 2 (cum 5>=5) -> last=12
    # bar 2: obs 3..4 (cum 8>=5) -> last=14
    # bar 3 (trailing partial): obs 5 (cum 1<5) -> last=15
    assert len(out) == 4


def test_volume_bars_index_labels():
    """Bar labels are the left (first) index of each bar by default."""
    price  = np.array([10., 11, 12, 13, 14, 15])
    volume = np.array([ 4.,  3,  5,  2,  6,  1])
    out, idx = Resample((price, np.arange(6)), volume, threshold=5, agg='last')
    # bar 0: first obs = 0; bar 1: first obs = 2; bar 2: first obs = 3;
    # bar 3 (trailing): first obs = 5
    np.testing.assert_array_equal(idx, [0, 2, 3, 5])


def test_volume_bars_sum_agg():
    """sum agg over volume bars."""
    price  = np.array([10., 11, 12, 13, 14, 15])
    volume = np.array([ 4.,  3,  5,  2,  6,  1])
    out, idx = Resample((price, np.arange(6)), volume, threshold=5, agg='sum')
    # bar 0: price obs 0..1, sum = 10+11=21
    np.testing.assert_allclose(out[0], 21.0)


# ---------------------------------------------------------------------------
# Validation: threshold must be positive
# ---------------------------------------------------------------------------

def test_threshold_non_positive_rejected():
    """threshold <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), np.ones(3), threshold=0, agg='sum')


def test_threshold_negative_rejected():
    """Negative threshold raises ValueError."""
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), np.ones(3), threshold=-1, agg='sum')


# ---------------------------------------------------------------------------
# Validation: threshold conflicts with other mode selectors
# ---------------------------------------------------------------------------

def test_threshold_conflicts_with_count():
    """threshold= with count= raises ValueError."""
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), np.ones(3),
                 threshold=5, count=2, agg='sum')


def test_threshold_conflicts_with_freq():
    """threshold= with freq= raises ValueError."""
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), np.ones(3),
                 threshold=5, freq=2, agg='sum')


# ---------------------------------------------------------------------------
# NaN driver: NaN does not advance the clock
# ---------------------------------------------------------------------------

def test_nan_driver_does_not_advance_clock():
    """NaN driver values are ignored (do NOT accumulate toward threshold)."""
    price  = np.array([10., 11, 12, 13])
    volume = np.array([ 4., np.nan, 3, 10])
    # clock: cum=4 (obs0), NaN skipped (obs1), cum=7>=6 (obs2, closes bar 0..2),
    # reset; cum=10>=6 (obs3, closes bar obs3)
    out, idx = Resample((price, np.arange(4)), volume, threshold=6, agg='last')
    assert len(out) == 2
    # bar 0 spans obs 0,1,2 (NaN obs included but driver ignored): last price = 12
    np.testing.assert_allclose(out[0], 12.0)


def test_nan_driver_bar_includes_nan_obs_in_value():
    """NaN driver obs still contributes its VALUE to the bar's reducer."""
    price  = np.array([10., 11, 12, 13])
    volume = np.array([ 4., np.nan, 3, 10])
    # bar 0 includes obs 0..2 (3 price obs); ohlc open=10, close=12
    out, idx = Resample((price, np.arange(4)), volume, threshold=6, agg='ohlc')
    np.testing.assert_allclose(out[0], [10, 12, 10, 12])   # O H L C


# ---------------------------------------------------------------------------
# All-regime test: eager == graph == lazy
# ---------------------------------------------------------------------------

def test_cumulative_mode_runs_in_all_regimes():
    """Crown-jewel: eager, graph, and lazy produce identical results for ByCumulative."""
    price  = np.array([10., 11, 12, 13, 14, 15, 16, 17])
    volume = np.array([ 3.,  4,  2,  5,  1,  6,  2,  3])
    idx    = np.arange(8)
    threshold = 5

    # eager
    eager_v, eager_k = Resample((price, idx), volume, threshold=threshold, agg='ohlc')

    # graph regime
    x = Input('x')
    d = Input('d')
    node = Resample(x, d, threshold=threshold, agg='ohlc')
    dag = Pipeline([x, d], [node])
    graph_v, graph_k = dag((price, idx), (volume, idx))

    # lazy regime (wide input: value+driver interleaved)
    lazy_v, lazy_k = _lazy_batch(dag, (price, idx), (volume, idx))

    np.testing.assert_allclose(
        graph_v, eager_v, equal_nan=True,
        err_msg="graph != eager for cumulative mode")
    np.testing.assert_array_equal(graph_k, eager_k,
        err_msg="graph index != eager index for cumulative mode")
    np.testing.assert_allclose(
        lazy_v, eager_v, equal_nan=True,
        err_msg="lazy != eager for cumulative mode")
    np.testing.assert_array_equal(lazy_k, eager_k,
        err_msg="lazy index != eager index for cumulative mode")


# ---------------------------------------------------------------------------
# CRITICAL 1: threshold= with multi-column-value aggs must be rejected
# ---------------------------------------------------------------------------

_MULTICOL_VALUE_AGGS = ("ohlcv", "ohlcv2", "ohlc_bars", "ohlcv_bars")


@pytest.mark.parametrize("agg", _MULTICOL_VALUE_AGGS)
def test_threshold_rejects_multicol_value_agg(agg):
    """threshold= mode requires a 1-D value stream; multi-column-value aggs
    (ohlcv, ohlcv2, ohlc_bars, ohlcv_bars) silently fed only width-1 value
    to the node -- they must be rejected with a clear ValueError."""
    price  = np.array([10., 11, 12, 13, 14, 15])
    volume = np.array([ 4.,  3,  5,  2,  6,  1])
    idx    = np.arange(6)
    with pytest.raises(ValueError, match="threshold="):
        Resample((price, idx), volume, threshold=5, agg=agg)


def test_threshold_ohlc_single_col_still_works():
    """threshold= with agg='ohlc' must still work (ohlc builds 4 cols from 1 value)."""
    price  = np.array([10., 11, 12, 13, 14, 15])
    volume = np.array([ 4.,  3,  5,  2,  6,  1])
    idx    = np.arange(6)
    out, bar_idx = Resample((price, idx), volume, threshold=5, agg='ohlc')
    assert out.shape[1] == 4
    assert len(bar_idx) > 0


# ---------------------------------------------------------------------------
# label="right" test for ByCumulative
# ---------------------------------------------------------------------------

def test_threshold_label_right():
    """label='right' uses the last event's index as the bar label."""
    price  = np.array([10., 11, 12, 13, 14, 15])
    volume = np.array([ 4.,  3,  5,  2,  6,  1])
    idx    = np.arange(6)
    # bar 0 spans obs 0..1 (cum 7>=5): last index = 1
    # bar 1 spans obs 2 alone: last index = 2
    # bar 2 spans obs 3..4: last index = 4
    # bar 3 (trailing partial): last index = 5
    out, bar_idx = Resample((price, idx), volume, threshold=5, agg='last',
                            label='right')
    np.testing.assert_array_equal(bar_idx, [1, 2, 4, 5])
