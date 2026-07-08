"""Time-driven `advance(now)` and on-demand `flush()` for the resample path.

`dag.live()` opens an incremental live streaming session: push events, drive a
clock with `advance(now)` (closing windows whose boundary has passed, even when
empty), force the current partial with `flush()`, and collect aligned outputs
with `result()`.

Labels use the default LEFT window edge convention (bucket label = its start
index origin + nb*width), matching tests/test_resample_fill.py.
"""
import numpy as np
import pytest

from screamer.streams import resample
from screamer.dag import Input, Dag
from screamer import ExpandingSum


def _live_last(fill):
    """A width-100, origin-0, agg='last' single-column live Dag session."""
    src = Input("x")
    node = resample(src, every=100, origin=0, agg="last", fill=fill)
    dag = Dag([src], [node])
    return dag.live()


# ---------------------------------------------------------------------------
# 1. Time-driven empty close, fill="nan"
# ---------------------------------------------------------------------------

def test_advance_closes_empty_windows_nan():
    live = _live_last("nan")
    live.push("x", 0, 10.0)
    live.advance(350)   # target bucket 3 (labels 0,100,200 close; 300 stays open)
    live.flush()        # open bucket 3 is empty -> emits nothing
    v, k = live.result()
    np.testing.assert_array_equal(k, [0, 100, 200])
    assert v[0] == 10.0
    assert np.isnan(v[1])
    assert np.isnan(v[2])


# ---------------------------------------------------------------------------
# 2. Time-driven empty close, fill="carry"
# ---------------------------------------------------------------------------

def test_advance_closes_empty_windows_carry():
    live = _live_last("carry")
    live.push("x", 0, 10.0)
    live.advance(350)
    live.flush()
    v, k = live.result()
    np.testing.assert_array_equal(k, [0, 100, 200])
    np.testing.assert_array_equal(v, [10.0, 10.0, 10.0])


# ---------------------------------------------------------------------------
# 3. fill="skip" (default): advance closes current window, no trailing empties
# ---------------------------------------------------------------------------

def test_advance_skip_emits_no_trailing_empties():
    live = _live_last("skip")
    live.push("x", 0, 10.0)
    live.advance(350)
    live.flush()
    v, k = live.result()
    np.testing.assert_array_equal(k, [0])
    np.testing.assert_array_equal(v, [10.0])


# ---------------------------------------------------------------------------
# 4. advance no-op inside the current bucket
# ---------------------------------------------------------------------------

def test_advance_noop_inside_current_bucket():
    # With advance(50) (same bucket as index 0, width 100): no premature emit.
    live_a = _live_last("skip")
    live_a.push("x", 0, 10.0)
    live_a.advance(50)          # still inside bucket 0 -> nothing yet
    live_a.push("x", 120, 20.0) # crosses into bucket 1 -> closes bucket 0
    live_a.flush()
    va, ka = live_a.result()

    # Same sequence WITHOUT the advance call.
    live_b = _live_last("skip")
    live_b.push("x", 0, 10.0)
    live_b.push("x", 120, 20.0)
    live_b.flush()
    vb, kb = live_b.result()

    np.testing.assert_array_equal(ka, kb)
    np.testing.assert_array_equal(va, vb)
    np.testing.assert_array_equal(ka, [0, 100])
    np.testing.assert_array_equal(va, [10.0, 20.0])


# ---------------------------------------------------------------------------
# 5. On-demand flush mid-stream
# ---------------------------------------------------------------------------

def test_flush_mid_stream_emits_each_partial_once():
    live = _live_last("skip")
    live.push("x", 0, 10.0)
    live.push("x", 50, 20.0)    # both in bucket 0
    live.flush()                # emit partial bucket 0 now (last=20 at label 0)
    live.push("x", 120, 30.0)   # new bucket 1
    live.flush()                # emit partial bucket 1 (label 100, value 30)
    v, k = live.result()
    np.testing.assert_array_equal(k, [0, 100])
    np.testing.assert_array_equal(v, [20.0, 30.0])


# ---------------------------------------------------------------------------
# 6. Batch parity unchanged (guards shared emit/emit_fill code)
# ---------------------------------------------------------------------------

def test_batch_parity_internal_gap_nan_unchanged():
    IDX = np.array([0, 3], dtype=np.int64)
    VALS = np.array([10.0, 40.0])
    src = Input("x")
    node = resample(src, every=1, agg="last", fill="nan")
    dag = Dag([src], [node])
    v, k = dag((VALS, IDX))
    np.testing.assert_array_equal(k, [0, 1, 2, 3])
    assert v[0] == 10.0
    assert np.isnan(v[1])
    assert np.isnan(v[2])
    assert v[3] == 40.0


# ---------------------------------------------------------------------------
# GenericResampleNode (functor reducer) time-driven empty close
# ---------------------------------------------------------------------------

def test_advance_closes_empty_windows_generic_reducer():
    src = Input("x")
    node = resample(src, every=100, origin=0, agg=ExpandingSum(), fill="nan")
    dag = Dag([src], [node])
    live = dag.live()
    live.push("x", 0, 10.0)
    live.advance(350)
    live.flush()
    v, k = live.result()
    v = np.asarray(v).reshape(-1)
    np.testing.assert_array_equal(k, [0, 100, 200])
    assert v[0] == 10.0
    assert np.isnan(v[1])
    assert np.isnan(v[2])
