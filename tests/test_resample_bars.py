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
