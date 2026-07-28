"""Tests for multi-column bar aggs: ohlc_bars, ohlcv_bars, ohlcv (all-regime),
ohlcv2 (all-regime).

Task 3: C++ multi-column resample reducer.
"""
import numpy as np
import pytest
from screamer import Input, Pipeline
from screamer.streams import Resample, resample
from tests._dag_oracle import lazy_batch as _lazy_batch


# ---------------------------------------------------------------------------
# ohlc_bars: reaggregate already-built OHLC bars
# ---------------------------------------------------------------------------

def test_ohlc_bars_reaggregates_built_bars():
    # 6 one-minute OHLC bars -> 2 three-bar bars
    O = np.array([1., 2, 3, 4, 5, 6])
    H = O + 0.5
    L = O - 0.5
    C = O + 0.1
    bars = np.column_stack([O, H, L, C])
    out, idx = Resample(count=3, agg='ohlc_bars')((bars, np.arange(6)))
    # bar0 = [first O, max H, min L, last C] over rows 0..2
    np.testing.assert_allclose(out[0], [1.0, 3.5, 0.5, 3.1])
    np.testing.assert_allclose(out[1], [4.0, 6.5, 3.5, 6.1])


def test_ohlc_bars_freq_based():
    # By freq (index-based)
    O = np.array([1., 2, 3, 4, 5, 6])
    H = O + 0.5
    L = O - 0.5
    C = O + 0.1
    bars = np.column_stack([O, H, L, C])
    idx = np.arange(6)
    out, bar_idx = Resample(freq=3, agg='ohlc_bars')((bars, idx))
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out[0], [1.0, 3.5, 0.5, 3.1])
    np.testing.assert_allclose(out[1], [4.0, 6.5, 3.5, 6.1])


def test_ohlc_bars_rejects_wrong_width():
    """ohlc_bars requires exactly 4 input columns."""
    with pytest.raises(ValueError):
        Resample(count=2, agg='ohlc_bars')((np.ones((4, 3)), np.arange(4)))


def test_ohlc_bars_rejects_1d_input():
    """ohlc_bars requires a 2D input."""
    with pytest.raises(ValueError):
        Resample(count=2, agg='ohlc_bars')((np.ones(4), np.arange(4)))


def test_ohlc_bars_rejects_wrong_width_5col():
    """ohlc_bars requires exactly 4 columns, not 5."""
    with pytest.raises(ValueError):
        Resample(count=2, agg='ohlc_bars')((np.ones((4, 5)), np.arange(4)))


# ---------------------------------------------------------------------------
# ohlcv_bars: reaggregate OHLCV bars (sum on every trailing column)
# ---------------------------------------------------------------------------

def test_ohlcv_bars_sums_trailing_columns():
    O = np.array([1., 2, 3, 4])
    H = O + 1
    L = O - 1
    C = O
    V = np.array([10., 20, 30, 40])
    bars = np.column_stack([O, H, L, C, V])
    out, _ = Resample(count=2, agg='ohlcv_bars')((bars, np.arange(4)))
    np.testing.assert_allclose(out[0], [1, 3, 0, 2, 30])   # V summed
    np.testing.assert_allclose(out[1], [3, 5, 2, 4, 70])


def test_ohlcv_bars_multiple_trailing_columns():
    """ohlcv_bars sums ALL trailing columns (5+)."""
    O = np.array([1., 2, 3, 4])
    H = O + 1
    L = O - 1
    C = O
    V1 = np.array([10., 20, 30, 40])
    V2 = np.array([1., 1, 1, 1])
    bars = np.column_stack([O, H, L, C, V1, V2])
    out, _ = Resample(count=2, agg='ohlcv_bars')((bars, np.arange(4)))
    assert out.shape == (2, 6)
    np.testing.assert_allclose(out[0, 4], 30.0)   # V1 sum
    np.testing.assert_allclose(out[0, 5], 2.0)    # V2 sum
    np.testing.assert_allclose(out[1, 4], 70.0)
    np.testing.assert_allclose(out[1, 5], 2.0)


def test_ohlcv_bars_rejects_fewer_than_5_columns():
    """ohlcv_bars requires at least 5 columns (4 OHLC + at least 1 volume)."""
    with pytest.raises(ValueError):
        Resample(count=2, agg='ohlcv_bars')((np.ones((4, 4)), np.arange(4)))


# ---------------------------------------------------------------------------
# ohlcv all-regime (was eager-only; must now NOT raise in graph/lazy)
# ---------------------------------------------------------------------------

def test_ohlcv_graph_does_not_raise():
    """ohlcv must work in the graph (Node) regime without raising."""
    price = np.arange(6, dtype=float)
    vol = np.ones(6)
    bars = np.column_stack([price, vol])
    idx = np.arange(6)
    x = Input('x')
    node = Resample(count=2, agg='ohlcv')(x)   # must not raise
    dag = Pipeline([x], [node])
    out_v, out_k = dag((bars, idx))
    assert out_v.shape == (3, 5)


def test_ohlcv_lazy_does_not_raise():
    """ohlcv must work in the lazy iterator regime without raising."""
    price = np.arange(6, dtype=float)
    vol = np.ones(6)
    bars = np.column_stack([price, vol])
    idx = np.arange(6)
    x = Input('x')
    dag = Pipeline([x], [Resample(count=2, agg='ohlcv')(x)])
    sv, sk = _lazy_batch(dag, (bars, idx))
    assert sv.shape == (3, 5)


def test_ohlcv_graph_matches_eager():
    """Graph regime of ohlcv produces the same result as eager."""
    price = np.arange(10, dtype=float)
    vol = np.arange(10, dtype=float) * 0.5
    bars = np.column_stack([price, vol])
    idx = np.arange(10)
    # eager
    eager_v, eager_k = Resample(count=5, agg='ohlcv')((bars, idx))
    # graph
    x = Input('x')
    dag = Pipeline([x], [Resample(count=5, agg='ohlcv')(x)])
    graph_v, graph_k = dag((bars, idx))
    np.testing.assert_allclose(graph_v, eager_v, equal_nan=True)
    np.testing.assert_array_equal(graph_k, eager_k)


# ---------------------------------------------------------------------------
# ohlcv2 all-regime (was eager-only; must now NOT raise in graph/lazy)
# ---------------------------------------------------------------------------

def test_ohlcv2_graph_does_not_raise():
    """ohlcv2 must work in the graph (Node) regime without raising."""
    price = np.arange(6, dtype=float)
    svol = np.array([1., -1, 1, -1, 1, -1])
    bars = np.column_stack([price, svol])
    idx = np.arange(6)
    x = Input('x')
    node = Resample(count=2, agg='ohlcv2')(x)
    dag = Pipeline([x], [node])
    out_v, out_k = dag((bars, idx))
    assert out_v.shape == (3, 6)


def test_ohlcv2_lazy_does_not_raise():
    """ohlcv2 must work in the lazy regime without raising."""
    price = np.arange(6, dtype=float)
    svol = np.array([1., -1, 1, -1, 1, -1])
    bars = np.column_stack([price, svol])
    idx = np.arange(6)
    x = Input('x')
    dag = Pipeline([x], [Resample(count=2, agg='ohlcv2')(x)])
    sv, sk = _lazy_batch(dag, (bars, idx))
    assert sv.shape == (3, 6)


def test_ohlcv2_graph_matches_eager():
    """Graph regime of ohlcv2 produces the same result as eager."""
    price = np.arange(10, dtype=float)
    svol = np.where(np.arange(10) % 2, 1.0, -1.0)
    bars = np.column_stack([price, svol])
    idx = np.arange(10)
    # eager
    eager_v, eager_k = Resample(count=5, agg='ohlcv2')((bars, idx))
    # graph
    x = Input('x')
    dag = Pipeline([x], [Resample(count=5, agg='ohlcv2')(x)])
    graph_v, graph_k = dag((bars, idx))
    np.testing.assert_allclose(graph_v, eager_v, equal_nan=True)
    np.testing.assert_array_equal(graph_k, eager_k)


# ---------------------------------------------------------------------------
# All-regime compliance test: eager == graph == lazy for all 4 bar aggs
# ---------------------------------------------------------------------------

def test_multicol_aggs_run_in_all_regimes():
    """Crown-jewel: eager, graph, and lazy produce identical results for all
    four multi-column bar aggs. ohlcv/ohlcv2 previously raised in graph/lazy -
    this verifies they no longer do."""
    bars_5col = np.column_stack([
        np.arange(6, dtype=float),       # col0: price / open
        np.arange(6, dtype=float) + 1,   # col1: high
        np.arange(6, dtype=float) - 1,   # col2: low
        np.arange(6, dtype=float),       # col3: close
        np.ones(6) * 10,                 # col4: volume
    ])
    idx = np.arange(6)

    configs = {
        'ohlc_bars':  bars_5col[:, :4],
        'ohlcv_bars': bars_5col,
        'ohlcv':      bars_5col[:, :2],
        'ohlcv2':     bars_5col[:, :2],
    }

    for agg, b in configs.items():
        # eager
        eager_v, eager_k = Resample(count=2, agg=agg)((b, idx))
        # graph
        x = Input('x')
        dag = Pipeline([x], [Resample(count=2, agg=agg)(x)])
        graph_v, graph_k = dag((b, idx))
        # lazy
        lazy_v, lazy_k = _lazy_batch(dag, (b, idx))
        # all three must match
        np.testing.assert_allclose(
            graph_v, eager_v, equal_nan=True,
            err_msg=f"graph != eager for agg={agg!r}")
        np.testing.assert_array_equal(graph_k, eager_k,
            err_msg=f"graph index != eager index for agg={agg!r}")
        np.testing.assert_allclose(
            lazy_v, eager_v, equal_nan=True,
            err_msg=f"lazy != eager for agg={agg!r}")
        np.testing.assert_array_equal(lazy_k, eager_k,
            err_msg=f"lazy index != eager index for agg={agg!r}")


# ---------------------------------------------------------------------------
# label="right" test for a multi-column bar agg (Task 3 new test)
# ---------------------------------------------------------------------------

def test_ohlcv_label_right():
    """label='right' returns the right grid edge, not the left, for ohlcv."""
    price = np.array([1., 2, 3, 4, 5, 6])
    vol   = np.ones(6)
    bars  = np.column_stack([price, vol])
    idx   = np.arange(6)
    out, bar_idx = Resample(freq=3, agg='ohlcv', label='right')((bars, idx))
    # freq=3, label='right': bar 0 -> right edge = 3; bar 1 -> right edge = 6
    np.testing.assert_array_equal(bar_idx, [3, 6])


def test_ohlcv_bars_label_right():
    """label='right' for ohlc_bars returns the right grid edge."""
    O = np.array([1., 4.])
    H = O + 0.5
    L = O - 0.5
    C = O + 0.1
    bars = np.column_stack([O, H, L, C])
    idx  = np.arange(2)
    out, bar_idx = Resample(count=1, agg='ohlc_bars', label='right')((bars, idx))
    # count=1, label='right': each bar's last-tick index
    np.testing.assert_array_equal(bar_idx, [0, 1])


# ---------------------------------------------------------------------------
# fill= with multi-column bar agg: internal empty bucket
# ---------------------------------------------------------------------------

def test_fill_nan_multicol_bar_agg():
    """fill='nan' for ohlcv emits an all-NaN row for internal empty buckets."""
    price = np.array([1., 4.])
    vol   = np.array([10., 40.])
    bars  = np.column_stack([price, vol])
    # index [0, 3]: events at bucket 0 and bucket 3, bucket 1 and 2 are empty.
    idx   = np.array([0, 3], dtype=np.int64)
    out, bar_idx = Resample(freq=1, agg='ohlcv', fill='nan')((bars, idx))
    # Should have 4 rows: buckets 0, 1, 2, 3
    assert out.shape[0] == 4, f"expected 4 rows, got {out.shape[0]}"
    assert out.shape[1] == 5, f"expected 5 cols, got {out.shape[1]}"
    # Row 0 has real values
    assert not np.isnan(out[0]).any()
    # Rows 1 and 2 are all NaN (internal gaps)
    assert np.isnan(out[1]).all(), "fill=nan gap row 1 should be all NaN"
    assert np.isnan(out[2]).all(), "fill=nan gap row 2 should be all NaN"
    # Row 3 has real values
    assert not np.isnan(out[3]).any()


def test_fill_carry_multicol_bar_agg():
    """fill='carry' for ohlcv repeats the previous row for internal empty buckets."""
    price = np.array([1., 4.])
    vol   = np.array([10., 40.])
    bars  = np.column_stack([price, vol])
    idx   = np.array([0, 3], dtype=np.int64)
    out, bar_idx = Resample(freq=1, agg='ohlcv', fill='carry')((bars, idx))
    # Buckets 1 and 2 carry bucket 0's values
    assert out.shape[0] == 4
    np.testing.assert_allclose(out[1], out[0], err_msg="carry row 1 != row 0")
    np.testing.assert_allclose(out[2], out[0], err_msg="carry row 2 != row 0")


# ---------------------------------------------------------------------------
# Golden-value regression: ohlcv and ohlcv2 (locked against a concrete oracle)
# ---------------------------------------------------------------------------

def test_ohlcv_golden_values():
    """Golden-value regression: ohlcv output is locked to a hardcoded oracle.

    Input (6 ticks, freq=3):
      price = [1, 2, 3, 4, 5, 6]
      vol   = [10, 20, 30, 40, 50, 60]
    Bar 0 (ticks 0..2): O=1, H=3, L=1, C=3, V=60
    Bar 1 (ticks 3..5): O=4, H=6, L=4, C=6, V=150
    """
    price = np.array([1., 2., 3., 4., 5., 6.])
    vol   = np.array([10., 20., 30., 40., 50., 60.])
    bars  = np.column_stack([price, vol])
    idx   = np.arange(6, dtype=np.int64)
    out, bar_idx = Resample(freq=3, agg='ohlcv')((bars, idx))

    expected = np.array([
        [1., 3., 1., 3., 60.],
        [4., 6., 4., 6., 150.],
    ])
    np.testing.assert_allclose(out, expected, err_msg="ohlcv golden values mismatch")
    np.testing.assert_array_equal(bar_idx, [0, 3])


def test_ohlcv2_golden_values():
    """Golden-value regression: ohlcv2 output is locked to a hardcoded oracle.

    Input (6 ticks, freq=3):
      price    = [1, 2, 3, 4, 5, 6]
      svol     = [1, -1, 1, -1, 1, -1]   (alternating)
    Bar 0 (ticks 0..2): O=1, H=3, L=1, C=3
      PosPart(svol) = [1, 0, 1] -> buy_vol  = 2
      NegPart(svol) = [0, 1, 0] -> sell_vol = 1  (magnitudes: |(-1)|=1)
    Bar 1 (ticks 3..5): O=4, H=6, L=4, C=6
      PosPart(svol) = [0, 1, 0] -> buy_vol  = 1
      NegPart(svol) = [1, 0, 1] -> sell_vol = 2
    """
    price = np.array([1., 2., 3., 4., 5., 6.])
    svol  = np.array([1., -1., 1., -1., 1., -1.])
    bars  = np.column_stack([price, svol])
    idx   = np.arange(6, dtype=np.int64)
    out, bar_idx = Resample(freq=3, agg='ohlcv2')((bars, idx))

    expected = np.array([
        [1., 3., 1., 3., 2., 1.],
        [4., 6., 4., 6., 1., 2.],
    ])
    np.testing.assert_allclose(out, expected, err_msg="ohlcv2 golden values mismatch")
    np.testing.assert_array_equal(bar_idx, [0, 3])
