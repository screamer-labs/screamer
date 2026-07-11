"""Oracle tests for the dropna -> DropNaNode routing.

Step 1 oracle: expected outputs captured from the numpy implementation before
migration.  Each assertion uses equal_nan-aware comparison because the
how="all" path retains rows that contain NaN (only all-NaN rows are dropped).

These tests exercise every regime: 1-D positional, 1-D indexed, 2-D how="any",
2-D how="all" (with all-NaN, partial-NaN, leading-NaN, trailing-NaN rows), and
lazy 1-D and 2-D event feeds (both indexed and positional-None index).

After the migration the same expected values must hold against the C++ path.
"""
import numpy as np
import pytest
from screamer.streams import dropna


# ---------------------------------------------------------------------------
# shared test data
# ---------------------------------------------------------------------------

# 1-D: leading NaN, middle NaN, trailing NaN
_V1 = np.array([np.nan, 1.0, 2.0, np.nan, 3.0, 4.0, np.nan])
_K1 = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)

# 2-D: leading-partial, no-NaN, partial (col 0), partial (col 1),
#       all-NaN, no-NaN, trailing-partial
_V2 = np.array([
    [np.nan, 1.0],
    [1.0, 2.0],
    [np.nan, 2.0],
    [1.0, np.nan],
    [np.nan, np.nan],
    [3.0, 4.0],
    [1.0, np.nan],
])
_K2 = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int64)

# Expected batch survivors (precomputed with the numpy implementation)
_EV1  = np.array([1.0, 2.0, 3.0, 4.0])
_EK1  = np.array([2, 3, 5, 6], dtype=np.int64)

_EV2_ANY = np.array([[1.0, 2.0], [3.0, 4.0]])
_EK2_ANY = np.array([20, 60], dtype=np.int64)

_EV2_ALL = np.array([
    [np.nan, 1.0],
    [1.0,    2.0],
    [np.nan, 2.0],
    [1.0,    np.nan],
    [3.0,    4.0],
    [1.0,    np.nan],
])
_EK2_ALL = np.array([10, 20, 30, 40, 60, 70], dtype=np.int64)


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _cmp(got_v, got_k, exp_v, exp_k):
    """Assert values and index match; values use equal_nan=True."""
    np.testing.assert_array_equal(
        got_v, exp_v,
        err_msg=f"values mismatch:\n  got  {got_v}\n  want {exp_v}"
    )
    np.testing.assert_array_equal(
        got_k, exp_k,
        err_msg=f"index mismatch:\n  got  {got_k}\n  want {exp_k}"
    )


def _cmp_nan(got_v, got_k, exp_v, exp_k):
    """Assert values and index match; values use equal_nan=True (for how='all')."""
    if not np.array_equal(np.asarray(got_v), np.asarray(exp_v), equal_nan=True):
        raise AssertionError(
            f"values mismatch (equal_nan=True):\n  got  {got_v}\n  want {exp_v}"
        )
    np.testing.assert_array_equal(
        got_k, exp_k,
        err_msg=f"index mismatch:\n  got  {got_k}\n  want {exp_k}"
    )


# ---------------------------------------------------------------------------
# batch - 1-D
# ---------------------------------------------------------------------------

def test_dropna_1d_positional():
    gv, gk = dropna(_V1)
    assert gk is None, "positional input must return None index"
    _cmp(gv, np.arange(len(_EV1), dtype=np.int64), _EV1, np.arange(len(_EV1), dtype=np.int64))
    np.testing.assert_array_equal(gv, _EV1)


def test_dropna_1d_indexed():
    gv, gk = dropna(_V1, index=_K1)
    _cmp(gv, gk, _EV1, _EK1)


# ---------------------------------------------------------------------------
# batch - 2-D how="any"
# ---------------------------------------------------------------------------

def test_dropna_2d_any_indexed():
    gv, gk = dropna(_V2, index=_K2, how="any")
    _cmp(gv, gk, _EV2_ANY, _EK2_ANY)


def test_dropna_2d_any_positional():
    gv, gk = dropna(_V2, how="any")
    assert gk is None, "positional input must return None index"
    _cmp(gv, np.arange(len(_EV2_ANY), dtype=np.int64),
         _EV2_ANY, np.arange(len(_EV2_ANY), dtype=np.int64))
    np.testing.assert_array_equal(gv, _EV2_ANY)


# ---------------------------------------------------------------------------
# batch - 2-D how="all" (survivors may contain NaN - use equal_nan comparison)
# ---------------------------------------------------------------------------

def test_dropna_2d_all_indexed():
    """how='all' only drops the fully-NaN row; partial-NaN rows survive with NaN intact."""
    gv, gk = dropna(_V2, index=_K2, how="all")
    _cmp_nan(gv, gk, _EV2_ALL, _EK2_ALL)


def test_dropna_2d_all_positional():
    gv, gk = dropna(_V2, how="all")
    assert gk is None, "positional input must return None index"
    if not np.array_equal(np.asarray(gv), np.asarray(_EV2_ALL), equal_nan=True):
        raise AssertionError(
            f"values mismatch:\n  got  {gv}\n  want {_EV2_ALL}"
        )


# ---------------------------------------------------------------------------
# Stream regime (dropna returns Stream, not tuple)
# ---------------------------------------------------------------------------

def test_dropna_stream_1d_indexed():
    from screamer.streams import Stream
    s = dropna(Stream(_V1, _K1))
    from screamer.streams import Stream as Sm
    assert isinstance(s, Sm)
    _cmp(s.values, s.index, _EV1, _EK1)


def test_dropna_stream_1d_positional():
    from screamer.streams import Stream
    s = dropna(Stream(_V1))
    assert isinstance(s, Stream)
    assert s.index is None
    np.testing.assert_array_equal(s.values, _EV1)


def test_dropna_stream_2d_any():
    from screamer.streams import Stream
    s = dropna(Stream(_V2, _K2), how="any")
    assert isinstance(s, Stream)
    _cmp(s.values, s.index, _EV2_ANY, _EK2_ANY)


def test_dropna_stream_2d_all():
    from screamer.streams import Stream
    s = dropna(Stream(_V2, _K2), how="all")
    assert isinstance(s, Stream)
    _cmp_nan(s.values, s.index, _EV2_ALL, _EK2_ALL)


# ---------------------------------------------------------------------------
# lazy - 1-D indexed
# ---------------------------------------------------------------------------

def test_dropna_lazy_1d_indexed_values():
    events = ((float(v), int(k)) for v, k in zip(_V1, _K1))
    rows = list(dropna(events))
    got_v = np.array([r[0] for r in rows], dtype=np.float64)
    got_k = np.array([r[1] for r in rows], dtype=np.int64)
    _cmp(got_v, got_k, _EV1, _EK1)


def test_dropna_lazy_1d_indexed_is_lazy():
    """No events consumed before the first next() call."""
    pulled = []

    def _spy():
        for v, k in zip(_V1, _K1):
            pulled.append(v)
            yield float(v), int(k)

    it = dropna(_spy())
    assert pulled == [], "dropna must not consume events at construction time"
    next(it)
    assert len(pulled) >= 1


def test_dropna_lazy_1d_positional_none_index():
    """Positional events (index=None) survive and keep None as index."""
    events = iter([(1.0, None), (float("nan"), None), (3.0, None)])
    rows = list(dropna(events))
    assert rows == [(1.0, None), (3.0, None)]


# ---------------------------------------------------------------------------
# lazy - 2-D indexed
# ---------------------------------------------------------------------------

def test_dropna_lazy_2d_any():
    events = ((tuple(float(x) for x in row), int(k)) for row, k in zip(_V2, _K2))
    rows = list(dropna(events, how="any"))
    got_k = np.array([r[1] for r in rows], dtype=np.int64)
    got_v_arr = np.array([list(r[0]) for r in rows], dtype=np.float64)
    _cmp(got_v_arr, got_k, _EV2_ANY, _EK2_ANY)


def test_dropna_lazy_2d_all():
    """how='all' lazy: only the all-NaN row is dropped; partial-NaN rows survive."""
    events = ((tuple(float(x) for x in row), int(k)) for row, k in zip(_V2, _K2))
    rows = list(dropna(events, how="all"))
    got_k = np.array([r[1] for r in rows], dtype=np.int64)
    got_v_arr = np.array([list(r[0]) for r in rows], dtype=np.float64)
    np.testing.assert_array_equal(got_k, _EK2_ALL)
    if not np.array_equal(got_v_arr, _EV2_ALL, equal_nan=True):
        raise AssertionError(
            f"values mismatch (equal_nan=True):\n  got  {got_v_arr}\n  want {_EV2_ALL}"
        )


# ---------------------------------------------------------------------------
# batch == lazy == graph consistency
# ---------------------------------------------------------------------------

def test_batch_lazy_graph_1d_any():
    """1-D any: batch, lazy, and graph all produce the same values and index."""
    from screamer import Input, Dag
    from tests._dag_oracle import lazy_batch as _lb

    x = Input("x")
    dag = Dag([x], [dropna(x, how="any")])

    bv, bk = dropna(_V1, index=_K1)

    events = ((float(v), int(k)) for v, k in zip(_V1, _K1))
    lazy_rows = list(dropna(events))
    lv = np.array([r[0] for r in lazy_rows], dtype=np.float64)
    lk = np.array([r[1] for r in lazy_rows], dtype=np.int64)

    gv, gk = _lb(dag, (_V1, _K1))

    _cmp(bv, bk, _EV1, _EK1)
    np.testing.assert_array_equal(lv, bv)
    np.testing.assert_array_equal(lk, bk)
    np.testing.assert_array_equal(gv.reshape(-1), bv)
    np.testing.assert_array_equal(gk, bk)


def test_batch_lazy_graph_2d_any():
    """2-D any: batch, lazy, and graph match the oracle."""
    from screamer import Input, Dag
    from screamer.streams import combine_latest
    from tests._dag_oracle import lazy_batch as _lb

    a, b = Input("a"), Input("b")
    dag = Dag([a, b], [dropna(combine_latest(a, b), how="any")])

    bv, bk = dropna(_V2, index=_K2, how="any")

    events = ((tuple(float(x) for x in row), int(k)) for row, k in zip(_V2, _K2))
    lazy_rows = list(dropna(events, how="any"))
    lk = np.array([r[1] for r in lazy_rows], dtype=np.int64)
    lv = np.array([list(r[0]) for r in lazy_rows], dtype=np.float64)

    col0 = _V2[:, 0].astype(np.float64)
    col1 = _V2[:, 1].astype(np.float64)
    gv, gk = _lb(dag, (col0, _K2), (col1, _K2))

    _cmp(bv, bk, _EV2_ANY, _EK2_ANY)
    np.testing.assert_array_equal(lk, bk)
    np.testing.assert_array_equal(lv, bv)
    np.testing.assert_array_equal(gk, bk)
    np.testing.assert_array_equal(gv.reshape(-1, 2), bv)


def test_batch_lazy_graph_2d_all():
    """2-D all: batch, lazy, and graph match; NaN-retaining rows compared with equal_nan."""
    from screamer import Input, Dag
    from screamer.streams import combine_latest
    from tests._dag_oracle import lazy_batch as _lb

    a, b = Input("a"), Input("b")
    dag = Dag([a, b], [dropna(combine_latest(a, b), how="all")])

    bv, bk = dropna(_V2, index=_K2, how="all")

    events = ((tuple(float(x) for x in row), int(k)) for row, k in zip(_V2, _K2))
    lazy_rows = list(dropna(events, how="all"))
    lk = np.array([r[1] for r in lazy_rows], dtype=np.int64)
    lv = np.array([list(r[0]) for r in lazy_rows], dtype=np.float64)

    col0 = _V2[:, 0].astype(np.float64)
    col1 = _V2[:, 1].astype(np.float64)
    gv, gk = _lb(dag, (col0, _K2), (col1, _K2))

    np.testing.assert_array_equal(bk, _EK2_ALL)
    if not np.array_equal(np.asarray(bv), _EV2_ALL, equal_nan=True):
        raise AssertionError(f"batch values mismatch: {bv}")

    np.testing.assert_array_equal(lk, bk)
    if not np.array_equal(lv, bv, equal_nan=True):
        raise AssertionError(f"lazy values mismatch: {lv}")

    np.testing.assert_array_equal(gk, bk)
    gv_2d = gv.reshape(-1, 2)
    if not np.array_equal(gv_2d, bv, equal_nan=True):
        raise AssertionError(f"graph values mismatch: {gv_2d}")
