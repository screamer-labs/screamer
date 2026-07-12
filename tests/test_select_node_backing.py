"""Oracle tests for the select -> SelectNode routing.

Step 1 oracle: expected outputs captured from the numpy implementation before
migration.  Each assertion uses equal_nan-aware comparison because select passes
NaN values through (it only projects columns, it does not filter rows).

These tests exercise every regime: 1-D passthrough (scalar and list), 2-D single
column (1-D result), 2-D multi-column (2-D result, including reordered columns),
positional and indexed, NaN passthrough, out-of-range ValueError, and lazy (1-D
and 2-D indexed and positional events).

After the migration the same expected values must hold against the C++ path.
"""
import numpy as np
import pytest
from screamer.streams import Select


# ---------------------------------------------------------------------------
# shared test data
# ---------------------------------------------------------------------------

# 3-row, 3-column input - row 1 has a NaN in column 1
_V3  = np.array([
    [10.0, 11.0, 12.0],
    [20.0, np.nan, 22.0],
    [30.0, 31.0, 32.0],
])
_K3  = np.array([1, 2, 3], dtype=np.int64)

# Expected single-column (scalar) output for column 1 -> 1-D, NaN passes through
_EV_COL1_1D = np.array([11.0, np.nan, 31.0])

# Expected multi-column output for [2, 0] -> 2-D reordered
_EV_MULTI = np.array([
    [12.0, 10.0],
    [22.0, 20.0],
    [32.0, 30.0],
])

# 2-column input with NaN - NaN must pass through in all regimes
_VNAN = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])
_KNAN = np.array([1, 2, 3], dtype=np.int64)
_ENAN_BOTH = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])   # all rows kept

# 1-D input
_V1D = np.array([5.0, 6.0, 7.0])
_K1D = np.array([1, 2, 3], dtype=np.int64)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cmp_nan(got_v, got_k, exp_v, exp_k):
    """Assert values (NaN-aware) and index match."""
    if not np.array_equal(np.asarray(got_v), np.asarray(exp_v), equal_nan=True):
        raise AssertionError(
            f"values mismatch (equal_nan=True):\n  got  {got_v}\n  want {exp_v}"
        )
    np.testing.assert_array_equal(
        got_k, exp_k,
        err_msg=f"index mismatch:\n  got  {got_k}\n  want {exp_k}"
    )


# ---------------------------------------------------------------------------
# batch - 1-D input (passthrough)
# ---------------------------------------------------------------------------

def test_select_1d_scalar_col0():
    """1-D input, scalar int 0 -> mirrors input (1-D result)."""
    v, k = Select(0)(_V1D, index=_K1D)
    assert v.ndim == 1
    np.testing.assert_array_equal(v, _V1D)
    np.testing.assert_array_equal(k, _K1D)


def test_select_1d_list_col0():
    """1-D input, list [0] -> 2-D (1 column) reshape."""
    v, k = Select([0])(_V1D, index=_K1D)
    assert v.shape == (3, 1)
    np.testing.assert_array_equal(v.reshape(-1), _V1D)
    np.testing.assert_array_equal(k, _K1D)


def test_select_1d_out_of_range_raises():
    """1-D input, column >= 1 is out of range."""
    with pytest.raises(ValueError, match="select: column 1 out of range for width 1"):
        Select(1)(_V1D, index=_K1D)


# ---------------------------------------------------------------------------
# batch - 2-D single column (scalar int -> 1-D result)
# ---------------------------------------------------------------------------

def test_select_2d_scalar_col1_indexed():
    """Scalar int column 1 -> 1-D result, NaN passes through."""
    v, k = Select(1)(_V3, index=_K3)
    assert v.ndim == 1
    _cmp_nan(v, k, _EV_COL1_1D, _K3)


def test_select_2d_scalar_col1_positional():
    """Positional (index=None) -> returned index is None."""
    v, k = Select(1)(_V3)
    assert k is None
    assert v.ndim == 1
    if not np.array_equal(v, _EV_COL1_1D, equal_nan=True):
        raise AssertionError(f"values mismatch: got {v}, want {_EV_COL1_1D}")


# ---------------------------------------------------------------------------
# batch - 2-D multi-column (list -> 2-D result)
# ---------------------------------------------------------------------------

def test_select_2d_multi_reordered():
    """List [2, 0] -> 2-D result with columns in that order."""
    v, k = Select([2, 0])(_V3, index=_K3)
    assert v.shape == (3, 2)
    np.testing.assert_array_equal(v, _EV_MULTI)
    np.testing.assert_array_equal(k, _K3)


def test_select_2d_nan_passthrough():
    """NaN values are NOT dropped by select; they pass through unchanged."""
    v, k = Select([0, 1])(_VNAN, index=_KNAN)
    assert v.shape == (3, 2)
    _cmp_nan(v, k, _ENAN_BOTH, _KNAN)


# ---------------------------------------------------------------------------
# batch - out-of-range column validation
# ---------------------------------------------------------------------------

def test_select_out_of_range_exact_message():
    """Out-of-range column raises ValueError with the exact message."""
    with pytest.raises(ValueError, match=r"select: column 5 out of range for width 3"):
        Select(5)(_V3, index=_K3)


def test_select_negative_raises():
    """Negative column index raises ValueError."""
    with pytest.raises(ValueError):
        Select(-1)(_V3, index=_K3)


def test_select_lazy_out_of_range_exact_message():
    """Lazy regime validates the column too, with the exact same message."""
    feed = ((row, int(k)) for row, k in zip(_V3, _K3))
    with pytest.raises(ValueError, match=r"select: column 5 out of range for width 3"):
        list(Select(5)(feed))


# ---------------------------------------------------------------------------
# Tuple regime
# ---------------------------------------------------------------------------

def test_select_tuple_scalar():
    """Tuple in -> tuple out, single column."""
    s = Select(1)((_V3, _K3))
    assert isinstance(s, tuple) and isinstance(s[0], np.ndarray)
    assert s[1] is not None
    np.testing.assert_array_equal(s[1], _K3)
    _cmp_nan(s[0], s[1], _EV_COL1_1D, _K3)


def test_select_tuple_multi():
    """Tuple in -> tuple out, multiple columns."""
    s = Select([2, 0])((_V3, _K3))
    assert isinstance(s, tuple) and isinstance(s[0], np.ndarray)
    np.testing.assert_array_equal(s[0], _EV_MULTI)
    np.testing.assert_array_equal(s[1], _K3)


def test_select_positional_array():
    """Positional bare array -> tuple with index=None; values correct."""
    s = Select(1)(_V3)
    assert isinstance(s, tuple) and isinstance(s[0], np.ndarray)
    assert s[1] is None
    if not np.array_equal(np.asarray(s[0]), _EV_COL1_1D, equal_nan=True):
        raise AssertionError(f"values mismatch: got {s[0]} want {_EV_COL1_1D}")


# ---------------------------------------------------------------------------
# lazy - 1-D indexed
# ---------------------------------------------------------------------------

def test_select_lazy_1d_indexed_scalar():
    """Lazy 1-D scalar events, column 0 -> same value passes through."""
    events = ((float(v), int(k)) for v, k in zip(_V1D.tolist(), _K1D.tolist()))
    rows = list(Select(0)(events))
    assert len(rows) == 3
    assert all(isinstance(v, float) for v, _ in rows)
    np.testing.assert_array_equal([v for v, _ in rows], _V1D)
    np.testing.assert_array_equal([k for _, k in rows], _K1D)


# ---------------------------------------------------------------------------
# lazy - 2-D indexed
# ---------------------------------------------------------------------------

def test_select_lazy_2d_scalar_col1():
    """Lazy 2-D events, scalar column 1 -> float per event, NaN passes through."""
    events = ((row, int(k)) for row, k in zip(_V3.tolist(), _K3.tolist()))
    rows = list(Select(1)(events))
    got_v = np.array([v for v, _ in rows], dtype=np.float64)
    got_k = np.array([k for _, k in rows], dtype=np.int64)
    assert all(isinstance(r[0], float) for r in rows), "scalar path must yield floats"
    _cmp_nan(got_v, got_k, _EV_COL1_1D, _K3)


def test_select_lazy_2d_multi_reordered():
    """Lazy 2-D events, multi-column [2, 0] -> list/tuple per event."""
    events = ((row, int(k)) for row, k in zip(_V3.tolist(), _K3.tolist()))
    rows = list(Select([2, 0])(events))
    got_k = np.array([k for _, k in rows], dtype=np.int64)
    got_v = np.array([list(v) for v, _ in rows], dtype=np.float64)
    np.testing.assert_array_equal(got_v, _EV_MULTI)
    np.testing.assert_array_equal(got_k, _K3)


def test_select_lazy_2d_positional_none_index():
    """Positional events (index=None) pass through with None index unchanged."""
    events = ((row, None) for row in _V3.tolist())
    rows = list(Select(1)(events))
    assert len(rows) == 3
    assert all(k is None for _, k in rows), "positional index must remain None"
    got_v = np.array([v for v, _ in rows], dtype=np.float64)
    if not np.array_equal(got_v, _EV_COL1_1D, equal_nan=True):
        raise AssertionError(f"values mismatch: got {got_v}, want {_EV_COL1_1D}")


# ---------------------------------------------------------------------------
# batch == lazy == graph consistency
# ---------------------------------------------------------------------------

def test_batch_lazy_graph_2d_scalar():
    """2-D scalar: batch, lazy, and graph all produce identical 1-D values+index."""
    from screamer import Input, Pipeline
    from tests._dag_oracle import lazy_batch as _lb

    a, b, c = Input("a"), Input("b"), Input("c")
    dag = Pipeline([a, b, c], [Select(1)(combine_latest_node(a, b, c))])

    bv, bk = Select(1)(_V3, index=_K3)

    events = ((row, int(k)) for row, k in zip(_V3.tolist(), _K3.tolist()))
    lazy_rows = list(Select(1)(events))
    lv = np.array([r[0] for r in lazy_rows], dtype=np.float64)
    lk = np.array([r[1] for r in lazy_rows], dtype=np.int64)

    col0, col1, col2 = _V3[:, 0], _V3[:, 1], _V3[:, 2]
    gv, gk = _lb(dag, (col0, _K3), (col1, _K3), (col2, _K3))

    # batch oracle
    _cmp_nan(bv, bk, _EV_COL1_1D, _K3)
    # lazy == batch
    np.testing.assert_array_equal(lk, bk)
    if not np.array_equal(lv, bv, equal_nan=True):
        raise AssertionError(f"lazy != batch: {lv} vs {bv}")
    # graph == batch
    np.testing.assert_array_equal(gk, bk)
    if not np.array_equal(gv.reshape(-1), bv, equal_nan=True):
        raise AssertionError(f"graph != batch: {gv} vs {bv}")


def test_batch_lazy_graph_2d_multi():
    """2-D multi-column [2,0]: batch, lazy, and graph all produce identical 2-D results."""
    from screamer import Input, Pipeline
    from tests._dag_oracle import lazy_batch as _lb

    a, b, c = Input("a"), Input("b"), Input("c")
    dag = Pipeline([a, b, c], [Select([2, 0])(combine_latest_node(a, b, c))])

    bv, bk = Select([2, 0])(_V3, index=_K3)

    events = ((row, int(k)) for row, k in zip(_V3.tolist(), _K3.tolist()))
    lazy_rows = list(Select([2, 0])(events))
    lv = np.array([list(v) for v, _ in lazy_rows], dtype=np.float64)
    lk = np.array([k for _, k in lazy_rows], dtype=np.int64)

    col0, col1, col2 = _V3[:, 0], _V3[:, 1], _V3[:, 2]
    gv, gk = _lb(dag, (col0, _K3), (col1, _K3), (col2, _K3))

    # batch oracle
    np.testing.assert_array_equal(bv, _EV_MULTI)
    np.testing.assert_array_equal(bk, _K3)
    # lazy == batch
    np.testing.assert_array_equal(lk, bk)
    assert np.array_equal(lv, bv, equal_nan=True)
    # graph == batch
    np.testing.assert_array_equal(gk, bk)
    assert np.array_equal(gv, bv, equal_nan=True)


# ---------------------------------------------------------------------------
# helper import (combine_latest for graph building)
# ---------------------------------------------------------------------------

def combine_latest_node(*nodes):
    """CombineLatest applied to Node inputs - returns a Node."""
    from screamer.streams import CombineLatest
    return CombineLatest()(*nodes)
