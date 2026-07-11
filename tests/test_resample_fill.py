"""Empty-window `fill` policy for resample (internal gaps).

Events land only in buckets 0 and 3 (every=1, index [0, 3]), so buckets 1 and 2
are internally empty. The `fill` policy controls what happens to those gaps:

* ``fill="skip"`` (default): no rows for empty buckets (today's behavior).
* ``fill="nan"``: an all-NaN row at each empty bucket's label.
* ``fill="carry"``: carry the previous emitted row forward.

Trailing empties (after the last event) are out of scope for this task; only
internal gaps (between two events) are filled here.
"""
import numpy as np
import pytest

from screamer.streams import Resample
from screamer import ExpandingSum


IDX = np.array([0, 3], dtype=np.int64)
VALS = np.array([10.0, 40.0])


# ---------------------------------------------------------------------------
# skip (default, unchanged)
# ---------------------------------------------------------------------------

def test_fill_skip_is_default_and_unchanged():
    v_default, k_default = Resample(freq=1, agg="last")(VALS, IDX)
    v_skip, k_skip = Resample(freq=1, agg="last", fill="skip")(VALS, IDX)
    np.testing.assert_array_equal(v_default, v_skip)
    np.testing.assert_array_equal(k_default, k_skip)
    # 2 rows only: buckets 0 and 3
    np.testing.assert_array_equal(k_skip, [0, 3])
    np.testing.assert_array_equal(v_skip, [10.0, 40.0])


# ---------------------------------------------------------------------------
# nan
# ---------------------------------------------------------------------------

def test_fill_nan_emits_nan_rows_for_internal_gaps():
    v, k = Resample(freq=1, agg="last", fill="nan")(VALS, IDX)
    np.testing.assert_array_equal(k, [0, 1, 2, 3])
    assert v[0] == 10.0
    assert np.isnan(v[1])
    assert np.isnan(v[2])
    assert v[3] == 40.0


# ---------------------------------------------------------------------------
# carry
# ---------------------------------------------------------------------------

def test_fill_carry_carries_previous_row_for_internal_gaps():
    v, k = Resample(freq=1, agg="last", fill="carry")(VALS, IDX)
    np.testing.assert_array_equal(k, [0, 1, 2, 3])
    np.testing.assert_array_equal(v, [10.0, 10.0, 10.0, 40.0])


# ---------------------------------------------------------------------------
# multi-column (ohlc, width 4)
# ---------------------------------------------------------------------------

def test_fill_nan_ohlc_multicolumn():
    s = Resample(freq=1, agg="ohlc", fill="nan")(VALS, IDX)
    v, k = s[0], s[1]
    np.testing.assert_array_equal(k, [0, 1, 2, 3])
    assert v.shape == (4, 4)
    np.testing.assert_array_equal(v[0], [10.0, 10.0, 10.0, 10.0])
    assert np.all(np.isnan(v[1]))
    assert np.all(np.isnan(v[2]))
    np.testing.assert_array_equal(v[3], [40.0, 40.0, 40.0, 40.0])


def test_fill_carry_ohlc_multicolumn():
    s = Resample(freq=1, agg="ohlc", fill="carry")(VALS, IDX)
    v, k = s[0], s[1]
    np.testing.assert_array_equal(k, [0, 1, 2, 3])
    assert v.shape == (4, 4)
    np.testing.assert_array_equal(v[0], [10.0, 10.0, 10.0, 10.0])
    np.testing.assert_array_equal(v[1], [10.0, 10.0, 10.0, 10.0])
    np.testing.assert_array_equal(v[2], [10.0, 10.0, 10.0, 10.0])
    np.testing.assert_array_equal(v[3], [40.0, 40.0, 40.0, 40.0])


# ---------------------------------------------------------------------------
# functor reducer path (GenericResampleNode)
# ---------------------------------------------------------------------------

def test_fill_carry_functor_reducer():
    v, k = Resample(freq=1, agg=ExpandingSum(), fill="carry")(VALS, IDX)
    np.testing.assert_array_equal(k, [0, 1, 2, 3])
    # bucket 0 sum=10, gaps carry 10, bucket 3 sum=40
    np.testing.assert_array_equal(np.asarray(v).reshape(-1), [10.0, 10.0, 10.0, 40.0])


def test_fill_nan_functor_reducer():
    v, k = Resample(freq=1, agg=ExpandingSum(), fill="nan")(VALS, IDX)
    v = np.asarray(v).reshape(-1)
    np.testing.assert_array_equal(k, [0, 1, 2, 3])
    assert v[0] == 10.0
    assert np.isnan(v[1])
    assert np.isnan(v[2])
    assert v[3] == 40.0


# ---------------------------------------------------------------------------
# graph (Node) path
# ---------------------------------------------------------------------------

def test_fill_carry_graph_path():
    from screamer.dag import Input, Dag
    src = Input("x")
    # node-mode span window via freq= (resolved against the runtime index)
    node = Resample(freq=1, agg="last", fill="carry")(src)
    dag = Dag([src], [node])
    v, k = dag((VALS, IDX))
    np.testing.assert_array_equal(k, [0, 1, 2, 3])
    np.testing.assert_array_equal(v, [10.0, 10.0, 10.0, 40.0])


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_fill_invalid_raises():
    with pytest.raises(ValueError):
        Resample(freq=1, agg="last", fill="bogus")(VALS, IDX)


# ---------------------------------------------------------------------------
# no trailing synthesis (out of scope): last event's bucket emits once, no
# trailing empties beyond it.
# ---------------------------------------------------------------------------

def test_fill_no_trailing_empties():
    v, k = Resample(freq=1, agg="last", fill="nan")(VALS, IDX)
    # last label is 3 (the last event's bucket); nothing after it.
    assert k[-1] == 3
    assert len(k) == 4
