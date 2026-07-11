"""Focused tests for FilterNode graph wiring (Task 1).

Builds graphs at the _GraphBuilder/_CompiledGraph level (same pattern as
test_dag_compile.py) to verify that the mask gate works correctly:
  - mask == 0.0  -> row is dropped
  - mask is NaN  -> row is dropped
  - any other mask value -> row is kept
  - data NaN with nonzero mask -> row is KEPT (only the mask gates)
"""
import math
import numpy as np
import pytest
from screamer import screamer_bindings as _b


def _feed(values, index=None):
    """Return (index_array, values_array) ready for run_batch feeds."""
    v = np.asarray(values, dtype=np.float64)
    k = np.arange(v.size, dtype=np.int64) if index is None else np.asarray(index, dtype=np.int64)
    return k, v


def _build_filter_graph():
    """Return a GraphBuilder with two inputs wired through a Filter output."""
    g = _b._GraphBuilder()
    data_in = g.add_input()
    mask_in = g.add_input()
    filt = g.add_filter([data_in, mask_in])
    g.set_outputs([filt])
    return g


# ---------------------------------------------------------------------------
# Basic gating
# ---------------------------------------------------------------------------

def test_filter_basic_gate():
    """data=[1,2,3,4] mask=[1,0,1,0] at same aligned index -> keep rows 0 and 2."""
    g = _build_filter_graph()
    data = [1.0, 2.0, 3.0, 4.0]
    mask = [1.0, 0.0, 1.0, 0.0]
    (out_k, out_v), = g.run_batch([_feed(data), _feed(mask)])
    np.testing.assert_array_equal(out_k, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(out_v.reshape(-1), np.array([1.0, 3.0]))


def test_filter_all_pass():
    """All nonzero mask -> all rows pass through."""
    g = _build_filter_graph()
    data = [10.0, 20.0, 30.0]
    mask = [1.0, 2.0, -1.0]   # negative is also a keeper
    (out_k, out_v), = g.run_batch([_feed(data), _feed(mask)])
    np.testing.assert_array_equal(out_k, np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(out_v.reshape(-1), np.array([10.0, 20.0, 30.0]))


def test_filter_all_zero_mask_yields_empty():
    """All-zero mask -> empty output."""
    g = _build_filter_graph()
    data = [1.0, 2.0, 3.0]
    mask = [0.0, 0.0, 0.0]
    (out_k, out_v), = g.run_batch([_feed(data), _feed(mask)])
    assert len(out_k) == 0
    assert len(out_v.reshape(-1)) == 0


# ---------------------------------------------------------------------------
# NaN mask drops
# ---------------------------------------------------------------------------

def test_filter_nan_mask_drops_row():
    """A NaN mask value drops that row."""
    g = _build_filter_graph()
    nan = math.nan
    data = [1.0, 2.0, 3.0, 4.0]
    mask = [1.0, nan, 1.0, 0.0]
    (out_k, out_v), = g.run_batch([_feed(data), _feed(mask)])
    # row 0 (mask=1.0): keep; row 1 (mask=NaN): drop; row 2 (mask=1.0): keep; row 3 (mask=0): drop
    np.testing.assert_array_equal(out_k, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(out_v.reshape(-1), np.array([1.0, 3.0]))


def test_filter_all_nan_mask_yields_empty():
    """All-NaN mask -> empty output."""
    g = _build_filter_graph()
    nan = math.nan
    data = [5.0, 6.0]
    mask = [nan, nan]
    (out_k, out_v), = g.run_batch([_feed(data), _feed(mask)])
    assert len(out_k) == 0


# ---------------------------------------------------------------------------
# NaN data is KEPT when mask permits
# ---------------------------------------------------------------------------

def test_filter_nan_data_passes_through_when_mask_nonzero():
    """NaN data value with nonzero mask is kept unchanged (mask gates, not data)."""
    g = _build_filter_graph()
    nan = math.nan
    data = [nan, 2.0, nan]
    mask = [1.0, 0.0, -3.5]   # rows 0 and 2 have nonzero mask -> kept
    (out_k, out_v), = g.run_batch([_feed(data), _feed(mask)])
    np.testing.assert_array_equal(out_k, np.array([0, 2], dtype=np.int64))
    assert math.isnan(out_v.reshape(-1)[0])    # NaN data preserved
    assert math.isnan(out_v.reshape(-1)[1])    # NaN data preserved


# ---------------------------------------------------------------------------
# Reset: running the same compiled graph twice gives identical output
# ---------------------------------------------------------------------------

def test_filter_reset_between_runs():
    """A compiled FilterNode graph must produce identical output on repeated runs."""
    g = _build_filter_graph()
    cg = g.compile()
    data = [1.0, 2.0, 3.0, 4.0]
    mask = [1.0, 0.0, 1.0, 0.0]
    feeds = [_feed(data), _feed(mask)]
    (k1, v1), = cg.run_batch(feeds)
    (k2, v2), = cg.run_batch(feeds)
    np.testing.assert_array_equal(k1, k2)
    np.testing.assert_array_equal(v1.reshape(-1), v2.reshape(-1))


# ---------------------------------------------------------------------------
# Coalescing: same-index updates settle before emit
# ---------------------------------------------------------------------------

def test_filter_coalescing_same_index_settles():
    """When data and mask arrive at the same index the final mask state governs.

    data arrives at [0,1,2], mask arrives at [0,1,2].
    All at same integer indices, but mask coalesces: the settled CombineLatest
    row at each index determines the gate. mask=[1,0,1] -> keep 0 and 2."""
    g = _build_filter_graph()
    data = [10.0, 20.0, 30.0]
    mask = [1.0,  0.0,  1.0]
    (out_k, out_v), = g.run_batch([_feed(data), _feed(mask)])
    np.testing.assert_array_equal(out_k, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(out_v.reshape(-1), np.array([10.0, 30.0]))


# ---------------------------------------------------------------------------
# Negative mask value is kept
# ---------------------------------------------------------------------------

def test_filter_negative_mask_keeps_row():
    """Negative mask values are nonzero, so they keep the row."""
    g = _build_filter_graph()
    data = [7.0, 8.0, 9.0]
    mask = [-1.0, 0.0, -0.001]
    (out_k, out_v), = g.run_batch([_feed(data), _feed(mask)])
    np.testing.assert_array_equal(out_k, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(out_v.reshape(-1), np.array([7.0, 9.0]))


# ---------------------------------------------------------------------------
# Negative-zero mask drops (IEEE 754: -0.0 == 0.0)
# ---------------------------------------------------------------------------

def test_filter_negative_zero_mask_drops_row():
    """-0.0 compares equal to 0.0, so it drops the row like +0.0."""
    g = _build_filter_graph()
    data = [1.0, 2.0, 3.0]
    mask = [-0.0, 1.0, 0.0]
    (out_k, out_v), = g.run_batch([_feed(data), _feed(mask)])
    np.testing.assert_array_equal(out_k, np.array([1], dtype=np.int64))
    np.testing.assert_array_equal(out_v.reshape(-1), np.array([2.0]))


# ---------------------------------------------------------------------------
# True same-index overwrite: two events at one index before the index advances.
# The last value at an index wins; the gate is applied to the settled row.
# ---------------------------------------------------------------------------

def test_filter_same_index_overwrite_settles_last():
    """Two data events at index 0 (10 then 20) with the mask flipping to keep
    only settle the LAST value; the gate uses the settled mask at that index."""
    g = _build_filter_graph()
    # data: index 0 twice (10, 20), then index 1 (30)
    # mask: index 0 (0 -> would drop), then index 0 again (1 -> keeps), index 1 (1)
    data_k = np.array([0, 0, 1], dtype=np.int64)
    data_v = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    mask_k = np.array([0, 0, 1], dtype=np.int64)
    mask_v = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    (out_k, out_v), = g.run_batch([(data_k, data_v), (mask_k, mask_v)])
    # index 0 settles to data=20, mask=1 -> kept; index 1 -> kept
    np.testing.assert_array_equal(out_k, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(out_v.reshape(-1), np.array([20.0, 30.0]))


def test_filter_wrong_arity_raises():
    """add_filter requires exactly 2 inputs (data, mask)."""
    g = _b._GraphBuilder()
    a = g.add_input()
    with pytest.raises(Exception):
        g.add_filter([a])
