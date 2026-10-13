"""Time-driven `advance(now)` and on-demand `flush()` for the resample path.

`dag.live()` opens an incremental live streaming session: push events, drive a
clock with `advance(now)` (emitting the open window once its boundary has
passed), force the current partial with `flush()`, and collect aligned outputs
with `result()`. An empty window is filled only between two windows with data,
when the later one's event arrives, so `advance` changes when a bar is emitted,
never which bars.

Labels use the default LEFT window edge convention (bucket label = its start
index origin + nb*width), matching tests/test_resample_fill.py.
"""
import numpy as np
import pytest

from screamer.streams import Resample
from screamer.dag import Input, Pipeline
from screamer import ExpandingSum


def _live_last(fill):
    """A width-100, origin-0, agg='last' single-column live Pipeline session."""
    src = Input("x")
    # node-mode span window via freq= (resolved against the runtime index)
    node = Resample(freq=100, origin=0, agg="last", fill=fill)(src)
    dag = Pipeline([src], [node])
    return dag.live()


# ---------------------------------------------------------------------------
# 1-2. advance closes the open window; trailing empty windows are not filled
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fill", ["nan", "carry"])
def test_advance_closes_the_window_and_fills_no_trailing_ones(fill):
    live = _live_last(fill)
    live.push("x", 0, 10.0)
    live.advance(350)   # bucket 0 closes; 100 and 200 are empty and trailing
    v, k = live.result()
    np.testing.assert_array_equal(k, [0])
    np.testing.assert_array_equal(v, [10.0])
    live.flush()        # nothing open
    v, k = live.result()
    assert len(k) == 0


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
    # node-mode span: use every= for correct span semantics
    node = Resample(freq=1, agg="last", fill="nan")(src)
    dag = Pipeline([src], [node])
    v, k = dag((VALS, IDX))
    np.testing.assert_array_equal(k, [0, 1, 2, 3])
    assert v[0] == 10.0
    assert np.isnan(v[1])
    assert np.isnan(v[2])
    assert v[3] == 40.0


# ---------------------------------------------------------------------------
# GenericResampleNode (functor reducer) time-driven empty close
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fill", ["nan", "carry"])
def test_advance_fills_no_trailing_windows_generic_reducer(fill):
    # The functor-reducer node follows the same rule.
    src = Input("x")
    node = Resample(freq=100, origin=0, agg=ExpandingSum(), fill=fill)(src)
    dag = Pipeline([src], [node])
    live = dag.live()
    live.push("x", 0, 10.0)
    live.advance(350)
    live.flush()
    v, k = live.result()
    np.testing.assert_array_equal(k, [0])
    np.testing.assert_array_equal(np.asarray(v).reshape(-1), [10.0])


# ---------------------------------------------------------------------------
# Idempotence / monotonicity: repeated advance must not double-emit, and a real
# event landing in the still-open target bucket must EXTEND it (not re-open it).
# ---------------------------------------------------------------------------

def test_advance_idempotent_and_extends_open_target_bucket():
    live = _live_last("nan")
    live.push("x", 0, 10.0)
    live.advance(350)   # closes 0(10); 100 and 200 wait for a later event
    live.advance(350)   # repeat same now -> no double-emit
    live.advance(360)   # no-op
    live.push("x", 320, 30.0)   # bucket 3 has data: 100 and 200 are filled
    live.advance(450)   # closes bucket 3 with the real value (label 300, value 30)
    live.flush()        # bucket 4 (label 400) empty and open -> nothing
    v, k = live.result()
    np.testing.assert_array_equal(k, [0, 100, 200, 300])
    assert v[0] == 10.0
    assert np.isnan(v[1])
    assert np.isnan(v[2])
    assert v[3] == 30.0


# ---------------------------------------------------------------------------
# advance is a no-op before the first event (no anchor) and in count mode
# (event-counted windows have no time semantics).
# ---------------------------------------------------------------------------

def test_advance_noop_not_started():
    live = _live_last("nan")
    live.advance(1000)   # no event yet -> nothing to anchor, emits nothing
    live.flush()
    v, k = live.result()
    assert len(np.asarray(k)) == 0


def test_advance_then_later_event_fills_parked_bucket_builtin():
    # Single-column builtin reducer: a later event in a further bucket fills the
    # empty buckets that advance() passed, per fill policy.
    live = _live_last("nan")
    live.push("x", 0, 10.0)
    live.advance(150)          # emit bucket 0 (=10)
    live.push("x", 320, 40.0)  # bucket 3 -> crosses past parked bucket 1 and bucket 2
    live.flush()
    v, k = live.result()
    v = np.asarray(v).reshape(-1)
    np.testing.assert_array_equal(k, [0, 100, 200, 300])
    assert v[0] == 10.0
    assert np.isnan(v[1])      # parked-then-crossed empty bucket -> NaN fill
    assert np.isnan(v[2])
    assert v[3] == 40.0


def test_advance_then_later_event_fills_parked_bucket_generic():
    # Same, single-column functor reducer (GenericResampleNode path).
    src = Input("x")
    node = Resample(freq=100, origin=0, agg=ExpandingSum(), fill="nan")(src)
    dag = Pipeline([src], [node])
    live = dag.live()
    live.push("x", 0, 10.0)
    live.advance(150)
    live.push("x", 320, 40.0)
    live.flush()
    v, k = live.result()
    v = np.asarray(v).reshape(-1)
    np.testing.assert_array_equal(k, [0, 100, 200, 300])
    assert v[0] == 10.0
    assert np.isnan(v[1])
    assert np.isnan(v[2])
    assert v[3] == 40.0


def test_advance_noop_count_mode():
    src = Input("x")
    node = Resample(count=2, agg="last", fill="nan")(src)
    dag = Pipeline([src], [node])
    live = dag.live()
    live.push("x", 0, 10.0)
    live.advance(10_000)   # count mode: advance has no time meaning -> no-op
    live.flush()           # flush emits the 1-event partial bucket at label 0
    v, k = live.result()
    np.testing.assert_array_equal(k, [0])
    np.testing.assert_array_equal(v, [10.0])
