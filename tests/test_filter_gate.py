"""Oracle tests for Filter (2-input mask gate).

Asserts batch == lazy == graph byte-identical for all interesting mask cases.
Cases:
  - all-keep mask
  - all-drop (zero) mask
  - alternating mask
  - NaN-in-mask drops
  - negative mask keeps
  - NaN data value passed through under a nonzero mask
  - mask built via GreaterThan()(x, 0.0) - end-to-end with comparison family
  - indexed (values, index) pair input
  - old predicate-based filter function is absent
"""
import math
import numpy as np
import pytest

from screamer import Filter, GreaterThan, Input, Dag
import screamer
import screamer.streams as _streams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _oracle(data, mask):
    """Reference survivors: data[i] where mask[i] != 0 and not isnan(mask[i])."""
    data = np.asarray(data, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    keep = (mask != 0.0) & ~np.isnan(mask)
    idx = np.where(keep)[0].astype(np.int64)
    return data[keep], idx


def _run_batch(data, mask):
    """Positional batch call: Filter()(array, array) -> (values, index)."""
    return Filter()(data, mask)


def _run_indexed(data, mask, idx):
    """Indexed batch call: Filter()((values, index), (mask, index)) -> (v, k)."""
    return Filter()((data, idx), (mask, idx))


def _run_lazy(data, mask):
    """Lazy call: Filter()(iter(...), iter(...)) -> list of (value, index)."""
    data = np.asarray(data, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    n = len(data)
    data_events = iter([(float(data[i]), i) for i in range(n)])
    mask_events = iter([(float(mask[i]), i) for i in range(n)])
    return list(Filter()(data_events, mask_events))


def _run_graph(data, mask):
    """Graph call: Dag with two Inputs -> Filter node -> run batch."""
    d, m = Input("data"), Input("mask")
    dag = Dag(inputs=[d, m], outputs=[Filter()(d, m)])
    return dag(data, mask)


def _assert_all_regimes(data, mask):
    """Assert batch == lazy == graph == oracle, byte-identical."""
    data = np.asarray(data, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    n = len(data)
    idx = np.arange(n, dtype=np.int64)

    exp_vals, exp_idx = _oracle(data, mask)

    # batch (positional)
    bv, bi = _run_batch(data, mask)
    np.testing.assert_array_equal(bi, exp_idx, err_msg="batch index vs oracle")
    np.testing.assert_array_equal(bv, exp_vals, err_msg="batch values vs oracle")

    # indexed batch
    iv, ii = _run_indexed(data, mask, idx)
    np.testing.assert_array_equal(ii, exp_idx, err_msg="indexed batch index vs oracle")
    np.testing.assert_array_equal(iv, exp_vals, err_msg="indexed batch values vs oracle")

    # lazy
    lazy_rows = _run_lazy(data, mask)
    if exp_vals.size == 0:
        assert lazy_rows == [], f"lazy expected empty, got {lazy_rows}"
    else:
        lv = np.array([r[0] for r in lazy_rows], dtype=np.float64)
        lk = np.array([r[1] for r in lazy_rows], dtype=np.int64)
        np.testing.assert_array_equal(lk, exp_idx, err_msg="lazy index vs oracle")
        # NaN-aware comparison: NaN == NaN for byte-identical check
        nan_pos = np.isnan(exp_vals)
        np.testing.assert_array_equal(np.isnan(lv), nan_pos, err_msg="lazy NaN positions")
        np.testing.assert_array_equal(lv[~nan_pos], exp_vals[~nan_pos],
                                      err_msg="lazy non-NaN values vs oracle")

    # graph
    gv, gi = _run_graph(data, mask)
    np.testing.assert_array_equal(gi, exp_idx, err_msg="graph index vs oracle")
    nan_pos = np.isnan(exp_vals)
    np.testing.assert_array_equal(np.isnan(gv), nan_pos, err_msg="graph NaN positions")
    if (~nan_pos).any():
        np.testing.assert_array_equal(gv[~nan_pos], exp_vals[~nan_pos],
                                      err_msg="graph non-NaN values vs oracle")

    # byte-identical across regimes for non-NaN survivors
    # batch == graph
    np.testing.assert_array_equal(gi, bi, err_msg="graph idx == batch idx")
    if nan_pos.any():
        np.testing.assert_array_equal(np.isnan(gv), np.isnan(bv))
        np.testing.assert_array_equal(gv[~nan_pos], bv[~nan_pos])
    else:
        np.testing.assert_array_equal(gv, bv, err_msg="graph values == batch values")


# ---------------------------------------------------------------------------
# Gate semantics cases
# ---------------------------------------------------------------------------

def test_filter_all_keep():
    """All nonzero mask - every data value passes."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([1.0, 2.0, -1.0, 0.5])
    _assert_all_regimes(data, mask)


def test_filter_all_drop_zero():
    """All-zero mask - no data values pass."""
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([0.0, 0.0, 0.0])
    _assert_all_regimes(data, mask)


def test_filter_alternating():
    """Alternating mask - odd indices kept."""
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    mask = np.array([1.0,  0.0,  1.0,  0.0,  1.0,  0.0])
    _assert_all_regimes(data, mask)


def test_filter_nan_in_mask_drops():
    """NaN mask values drop, zero drops, nonzero keeps."""
    nan = math.nan
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([1.0, nan, 0.0, nan, -2.0])
    # expected: indices 0 and 4
    _assert_all_regimes(data, mask)
    bv, bi = _run_batch(data, mask)
    np.testing.assert_array_equal(bi, [0, 4])
    np.testing.assert_array_equal(bv, [1.0, 5.0])


def test_filter_negative_mask_keeps():
    """Negative mask values are nonzero, so they keep the row."""
    data = np.array([7.0, 8.0, 9.0])
    mask = np.array([-1.0, 0.0, -0.001])
    _assert_all_regimes(data, mask)
    bv, bi = _run_batch(data, mask)
    np.testing.assert_array_equal(bi, [0, 2])
    np.testing.assert_array_equal(bv, [7.0, 9.0])


def test_filter_nan_data_kept_under_nonzero_mask():
    """NaN data passes through when its aligned mask is nonzero (mask gates, not data)."""
    nan = math.nan
    data = np.array([nan, 2.0, nan, 4.0])
    mask = np.array([1.0, 0.0, -3.5, 0.0])
    # expected: indices 0 (nan) and 2 (nan)
    _assert_all_regimes(data, mask)
    bv, bi = _run_batch(data, mask)
    np.testing.assert_array_equal(bi, [0, 2])
    assert math.isnan(bv[0])
    assert math.isnan(bv[1])


def test_filter_all_nan_mask():
    """All-NaN mask - empty output."""
    nan = math.nan
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([nan, nan, nan])
    _assert_all_regimes(data, mask)
    bv, bi = _run_batch(data, mask)
    assert len(bv) == 0 and len(bi) == 0


def test_filter_single_element_keep():
    """Single element, kept."""
    _assert_all_regimes([42.0], [1.0])


def test_filter_single_element_drop():
    """Single element, dropped."""
    _assert_all_regimes([42.0], [0.0])


def test_filter_empty_arrays():
    """Empty input -> empty output."""
    data = np.array([], dtype=np.float64)
    mask = np.array([], dtype=np.float64)
    bv, bi = _run_batch(data, mask)
    assert len(bv) == 0 and len(bi) == 0


# ---------------------------------------------------------------------------
# End-to-end with comparison family
# ---------------------------------------------------------------------------

def test_filter_via_greater_than_mask():
    """Filter()(x, GreaterThan()(x, zeros)) keeps positive values."""
    x = np.array([5.0, -2.0, 8.0, -1.0, 3.0, 0.0])
    thresh = np.zeros_like(x)
    mask = GreaterThan()(x, thresh)
    survivors, idx = Filter()(x, mask)
    expected = np.array([5.0, 8.0, 3.0])
    np.testing.assert_array_equal(survivors, expected)
    np.testing.assert_array_equal(idx, [0, 2, 4])


def test_filter_graph_two_input_dag():
    """Build a 2-input Dag with Filter and verify it gives the same result as batch."""
    d_in = Input("data")
    m_in = Input("mask")
    out_node = Filter()(d_in, m_in)
    dag = Dag(inputs=[d_in, m_in], outputs=[out_node])
    x = np.array([5.0, -2.0, 8.0, -1.0, 3.0])
    # mask: keep positives (1) and drop non-positives (0)
    mask = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    survivors, idx = dag(x, mask)
    np.testing.assert_array_equal(survivors, [5.0, 8.0, 3.0])
    np.testing.assert_array_equal(idx, [0, 2, 4])


# ---------------------------------------------------------------------------
# Node / graph path
# ---------------------------------------------------------------------------

def test_filter_returns_node_when_inputs_are_nodes():
    """Filter()(node, node) returns a Node (for lazy graph building)."""
    from screamer.dag import is_node
    d = Input("d")
    m = Input("m")
    result = Filter()(d, m)
    assert is_node(result)


def test_filter_returns_node_when_data_is_node():
    d = Input("d")
    from screamer.dag import is_node
    result = Filter()(d, np.array([1.0]))
    # data is a node, mask is not; make_operator_node should still be called
    assert is_node(result)


def test_filter_returns_node_when_mask_is_node():
    from screamer.dag import is_node
    m = Input("m")
    result = Filter()(np.array([1.0]), m)
    assert is_node(result)


# ---------------------------------------------------------------------------
# Lazy regime: iterator returns iterator
# ---------------------------------------------------------------------------

def test_filter_lazy_returns_iterator():
    data_events = iter([(1.0, 0), (2.0, 1)])
    mask_events = iter([(1.0, 0), (0.0, 1)])
    result = Filter()(data_events, mask_events)
    assert hasattr(result, "__next__"), "lazy Filter must return an iterator"
    rows = list(result)
    assert len(rows) == 1
    assert rows[0][0] == 1.0


def test_filter_lazy_nan_mask_drops():
    nan = math.nan
    data_events = iter([(1.0, 0), (2.0, 1), (3.0, 2)])
    mask_events = iter([(1.0, 0), (nan, 1), (1.0, 2)])
    rows = list(Filter()(data_events, mask_events))
    assert [r[0] for r in rows] == [1.0, 3.0]
    assert [r[1] for r in rows] == [0, 2]


# ---------------------------------------------------------------------------
# Old predicate-based filter is gone
# ---------------------------------------------------------------------------

def test_old_filter_not_in_streams():
    """The predicate-based filter function must not exist in screamer.streams."""
    assert not hasattr(_streams, "filter"), \
        "screamer.streams.filter (predicate-based) must have been removed"


def test_screamer_filter_is_Filter_class():
    """screamer.Filter is the new mask-gate class, not the old function."""
    assert screamer.Filter is Filter
    assert isinstance(screamer.Filter(), Filter)


def test_filter_in_all():
    """Filter must be present in screamer.__all__."""
    assert "Filter" in screamer.__all__
    assert "filter" not in screamer.__all__


def test_filter_composes_as_graph_node():
    """Filter composes with upstream/downstream nodes inside one Dag: data from an
    upstream functor, mask from a 1-input classifier, and the gated output feeding
    a downstream functor - all in C++, no eager mask."""
    import numpy as np
    from screamer import Filter, IsFinite, RollingMean, Input, Dag
    x = np.array([1., 2, np.nan, 4, 5, 6], dtype=float)
    idx = np.arange(len(x), dtype=np.int64)

    # upstream data node + 1-in mask node
    d = Input("x")
    up = Dag(inputs=[d], outputs=[Filter()(RollingMean(2)(d), IsFinite()(d))])
    uv, uk = up((x, idx))

    # Filter feeding a downstream node
    d2 = Input("x")
    down = Dag(inputs=[d2], outputs=[RollingMean(2)(Filter()(d2, IsFinite()(d2)))])
    dv, dk = down((x, idx))

    # index 2 (x=NaN) is dropped in both; kept indices are [0,1,3,4,5]
    np.testing.assert_array_equal(uk, np.array([0, 1, 3, 4, 5], dtype=np.int64))
    np.testing.assert_array_equal(dk, np.array([0, 1, 3, 4, 5], dtype=np.int64))
    # downstream RollingMean(2) over the survivors [1,2,4,5,6] -> [nan,1.5,3,4.5,5.5]
    np.testing.assert_allclose(dv, [np.nan, 1.5, 3.0, 4.5, 5.5], equal_nan=True)
