"""Empty-window `fill` policy for resample (internal gaps).

Events land only in buckets 0 and 3 (every=1, index [0, 3]), so buckets 1 and 2
are internally empty. The `fill` policy controls what happens to those gaps:

* ``fill="skip"`` (default): no rows for empty buckets (today's behavior).
* ``fill="nan"``: an all-NaN row at each empty bucket's label.
* ``fill="carry"``: each column takes its last non-NaN value: an empty bucket
  repeats those values, and a NaN cell of a bar takes its column's value (G23).

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
    from screamer.dag import Input, Pipeline
    src = Input("x")
    # node-mode span window via freq= (resolved against the runtime index)
    node = Resample(freq=1, agg="last", fill="carry")(src)
    dag = Pipeline([src], [node])
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


# ---------------------------------------------------------------------------
# G23: carry is per column, in every mode and regime
# ---------------------------------------------------------------------------

def _carry_reference(bars):
    """Per-column forward fill of the bars (NaN until a column has a value)."""
    out = np.array(bars, dtype=float)
    last = np.full(out.shape[1:], np.nan)
    for t in range(len(out)):
        nan = np.isnan(out[t])
        out[t][nan] = last[nan]
        last = out[t].copy()
    return out


WIDE_K = np.array([0, 10, 25, 40, 41, 70], dtype=np.int64)
WIDE = np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, np.nan],
                 [np.nan, np.nan], [np.nan, 50.0], [6.0, np.nan]])


@pytest.mark.parametrize("agg", ["last", "mean", "ohlc"])
def test_fill_carry_is_per_column_on_wide_rows(agg):
    v, k = Resample(freq=10, agg=agg, fill="carry")((WIDE, WIDE_K))
    bars, _ = Resample(freq=10, agg=agg, fill="nan")((WIDE, WIDE_K))
    np.testing.assert_array_equal(k, [0, 10, 20, 30, 40, 50, 60, 70])
    np.testing.assert_array_equal(v, _carry_reference(bars))
    # Each column alone gives the same values: the 1-D rule, per column.
    for c in range(2):
        vc, _ = Resample(freq=10, agg=agg, fill="carry")((WIDE[:, c], WIDE_K))
        np.testing.assert_array_equal(v[:, c], vc)


def test_fill_carry_a_nan_bar_takes_the_last_value_in_one_column():
    v, k = Resample(freq=10, fill="carry")((np.array([1.0, np.nan, 3.0]), np.array([0, 10, 40])))
    np.testing.assert_array_equal(k, [0, 10, 20, 30, 40])
    np.testing.assert_array_equal(v, [1.0, 1.0, 1.0, 1.0, 3.0])
    # Before a column has a value, its cells stay NaN.
    v, _ = Resample(freq=10, fill="carry")((np.array([np.nan, 2.0]), np.array([0, 10])))
    assert np.isnan(v[0]) and v[1] == 2.0


def test_fill_carry_clock_is_per_column():
    clock = (np.zeros(5), np.array([5, 15, 30, 45, 75], dtype=np.int64))
    v, k = Resample((WIDE, WIDE_K), clock, clock=True, agg="last", fill="carry")
    np.testing.assert_array_equal(k, [5, 15, 30, 45, 75])
    np.testing.assert_array_equal(v, [[1, 10], [1, 20], [3, 20], [3, 50], [6, 50]])
    for c in range(2):
        vc, _ = Resample((WIDE[:, c], WIDE_K), clock, clock=True, agg="last", fill="carry")
        np.testing.assert_array_equal(v[:, c], vc)


def test_fill_carry_functor_reducer_is_per_column():
    v, k = Resample(freq=10, agg=ExpandingSum(), fill="carry")(
        (np.array([1.0, np.nan, 3.0]), np.array([0, 10, 40])))
    np.testing.assert_array_equal(np.asarray(v).reshape(-1), [1.0, 1.0, 1.0, 1.0, 3.0])


def test_fill_carry_every_regime_agrees():
    from screamer.dag import Input, Pipeline
    batch = Resample(freq=10, agg="last", fill="carry")((WIDE, WIDE_K))
    lazy = list(Resample(freq=10, agg="last", fill="carry")(zip(WIDE, WIDE_K.tolist())))
    np.testing.assert_array_equal(np.stack([r[0] for r in lazy]), batch[0])
    x = Input("x")
    pipe = Pipeline([x], [Resample(freq=10, agg="last", fill="carry")(x)])
    np.testing.assert_array_equal(pipe((WIDE, WIDE_K))[0], batch[0])
    live = pipe.live()
    for row, t in zip(WIDE, WIDE_K):
        live.push("x", int(t), row.tolist())
    live.flush()
    np.testing.assert_array_equal(live.result()[0], batch[0])


@pytest.mark.parametrize("fill", ["nan", "carry"])
def test_fill_is_refused_under_count_and_threshold(fill):
    # fill= applies to freq= and clock= bars; a count= or threshold= bar is
    # never empty, so fill= other than "skip" raises (2.8.0 ignored it).
    from screamer.dag import Input, Pipeline
    with pytest.raises(ValueError, match="a count= bar is never empty"):
        Resample(count=1, fill=fill)(WIDE)
    v = np.arange(4.0)
    with pytest.raises(ValueError, match="a threshold= bar is never empty"):
        Resample(v, np.ones(4), threshold=1.0, fill=fill)
    x = Input("x")
    with pytest.raises(ValueError, match="a count= bar is never empty"):
        Pipeline([x], [Resample(count=2, fill=fill)(x)])
    np.testing.assert_array_equal(Resample(count=1, fill="skip")(WIDE)[0], WIDE)
