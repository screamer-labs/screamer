"""Count-mode (distinct ticks) and optional clock port for MultiResampleNode.

Two capabilities layered on the Task-5 MultiResampleNode:

1. Count mode over DISTINCT TICKS: ``count=N`` closes a bar every N distinct
   indices (ticks), not every N port-pushes. Two ports pushing at the same index
   is ONE tick. A full bar closes on the NEXT new tick (deferred close) so batch
   == stream and multi-port correctness hold.
2. An optional trailing clock port: an extra input that only advances the bucket
   (feeds no reducer, adds no column). A clock tick crossing a bucket boundary
   closes the current bar even with zero trades, finalizing empty time-bars
   straight from the data (ByIndex/``every=`` only).
"""
import numpy as np
import pytest

from screamer.dag import Input, Dag
from screamer.streams import multi_resample, resample
from screamer import First, Last, ExpandingMax, ExpandingMin, ExpandingSum, PosPart


# ---------------------------------------------------------------------------
# 1. Count mode == stacked single-column count resamples (labels AND values)
# ---------------------------------------------------------------------------

def test_count_mode_matches_stacked_single_resamples():
    idx = np.arange(20, dtype=np.int64)
    vals = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3,
                     5, 8, 9, 7, 9, 3, 2, 3, 8, 4], dtype=np.float64)

    price = Input("p")
    bars = multi_resample(
        [price, price, price],
        [First(), ExpandingMax(), Last()],
        count=3)
    v, k = Dag([price], [bars])((vals, idx))

    assert v.shape[1] == 3
    for col, agg in enumerate(["first", "max", "last"]):
        rv, rk = resample(vals, idx, count=3, agg=agg)
        np.testing.assert_array_equal(k, rk)
        np.testing.assert_array_equal(v[:, col], rv)


# ---------------------------------------------------------------------------
# 2. Distinct-tick counting: two ports at the SAME indices => bar every 2 TICKS
# ---------------------------------------------------------------------------

def test_count_mode_counts_distinct_ticks_not_pushes():
    idx = np.arange(8, dtype=np.int64)
    price = np.array([10, 11, 9, 12, 13, 8, 7, 14], dtype=np.float64)
    vol = np.array([2, -3, 4, -1, 5, -6, 7, -2], dtype=np.float64)

    p = Input("p")
    q = Input("v")
    bars = multi_resample([p, PosPart()(q)], [Last(), ExpandingSum()], count=2)
    v, k = Dag([p, q], [bars])((price, idx), (vol, idx))

    # A bar every 2 TICKS (2 shared indices): 8 indices -> 4 bars.
    # A per-PUSH bug (2 pushes per tick) would close every 1 tick -> 8 bars.
    assert v.shape == (4, 2)
    np.testing.assert_array_equal(k, [0, 2, 4, 6])   # left = first tick of each pair

    pos = np.where(vol > 0, vol, 0.0)
    # column 0: Last of price per 2-tick bar; column 1: Sum of PosPart(vol)
    rv0, rk0 = resample(price, idx, count=2, agg="last")
    rv1, _ = resample(pos, idx, count=2, agg="sum")
    np.testing.assert_array_equal(k, rk0)
    np.testing.assert_array_equal(v[:, 0], rv0)
    np.testing.assert_array_equal(v[:, 1], rv1)

    # hand-computed guard: last price of ticks (0,1)=11, (2,3)=12, (4,5)=8, (6,7)=14
    np.testing.assert_array_equal(v[:, 0], [11, 12, 8, 14])
    # sum of PosPart(vol): (2,0)=2, (4,0)=4, (5,0)=5, (7,0)=7
    np.testing.assert_array_equal(v[:, 1], [2, 4, 5, 7])


# ---------------------------------------------------------------------------
# 3. Count mode batch == stream (two-input graph)
# ---------------------------------------------------------------------------

def test_count_mode_batch_equals_stream():
    idx = np.arange(11, dtype=np.int64)
    price = np.array([10, 11, 9, 12, 13, 8, 7, 14, 15, 6, 5], dtype=np.float64)
    vol = np.array([2, 3, 4, 1, 5, 6, 7, 2, 3, 4, 1], dtype=np.float64)

    p = Input("p")
    q = Input("v")
    bars = multi_resample([p, PosPart()(q)], [Last(), ExpandingSum()], count=3)
    dag = Dag([p, q], [bars])

    vb, kb = dag((price, idx), (vol, idx))

    live = dag.live()
    for k_i, pv, vv in zip(idx, price, vol):
        live.push("p", int(k_i), float(pv))
        live.push("v", int(k_i), float(vv))
    live.flush()
    vs, ks = live.result()

    np.testing.assert_array_equal(kb, ks)
    np.testing.assert_array_equal(vb, vs)


# ---------------------------------------------------------------------------
# 4. Clock port closes empty time-bars straight from the data (fill="nan")
# ---------------------------------------------------------------------------

def test_clock_port_closes_empty_bars_nan():
    price = Input("p")
    clk = Input("clk")
    bars = multi_resample([price], [Last()], clock=clk, every=100, origin=0,
                          fill="nan")
    dag = Dag([price, clk], [bars])

    p_idx = np.array([0], dtype=np.int64)
    p_val = np.array([10.0])
    c_idx = np.array([0, 100, 200, 300], dtype=np.int64)
    c_val = np.array([0.0, 0.0, 0.0, 0.0])

    # No advance() call: the clock ticks themselves close buckets 0/1/2.
    v, k = dag((p_val, p_idx), (c_val, c_idx))

    np.testing.assert_array_equal(k, [0, 100, 200])
    v = np.asarray(v).reshape(-1)   # width-1 output is returned 1-D
    assert v[0] == 10.0
    assert np.isnan(v[1])
    assert np.isnan(v[2])


# ---------------------------------------------------------------------------
# 5. Clock port with fill="carry" repeats the previous bar for empty windows
# ---------------------------------------------------------------------------

def test_clock_port_closes_empty_bars_carry():
    price = Input("p")
    clk = Input("clk")
    bars = multi_resample([price], [Last()], clock=clk, every=100, origin=0,
                          fill="carry")
    dag = Dag([price, clk], [bars])

    p_idx = np.array([0, 50], dtype=np.int64)
    p_val = np.array([10.0, 20.0])          # bucket 0: Last = 20
    c_idx = np.array([0, 100, 200, 300], dtype=np.int64)
    c_val = np.array([0.0, 0.0, 0.0, 0.0])

    v, k = dag((p_val, p_idx), (c_val, c_idx))

    np.testing.assert_array_equal(k, [0, 100, 200])
    v = np.asarray(v).reshape(-1)   # width-1 output is returned 1-D
    np.testing.assert_array_equal(v, [20.0, 20.0, 20.0])   # carried


# ---------------------------------------------------------------------------
# 6. Clock port batch == stream
# ---------------------------------------------------------------------------

def test_clock_port_batch_equals_stream():
    price = Input("p")
    clk = Input("clk")
    bars = multi_resample([price], [Last()], clock=clk, every=100, origin=0,
                          fill="nan")
    dag = Dag([price, clk], [bars])

    p_idx = np.array([0, 130, 150], dtype=np.int64)
    p_val = np.array([10.0, 20.0, 25.0])
    c_idx = np.array([0, 100, 200, 300], dtype=np.int64)
    c_val = np.array([0.0, 0.0, 0.0, 0.0])

    vb, kb = dag((p_val, p_idx), (c_val, c_idx))

    live = dag.live()
    # merge-by-index push order: interleave the two streams by index
    events = sorted(
        [(int(i), "p", float(x)) for i, x in zip(p_idx, p_val)] +
        [(int(i), "clk", float(x)) for i, x in zip(c_idx, c_val)])
    for i, name, x in events:
        live.push(name, i, x)
    live.flush()
    vs, ks = live.result()

    np.testing.assert_array_equal(kb, ks)
    np.testing.assert_array_equal(vb, vs)


# ---------------------------------------------------------------------------
# 7. Regression: a column-only ByIndex graph (no clock) is unchanged
# ---------------------------------------------------------------------------

def test_column_only_byindex_regression():
    W = 5
    idx = np.arange(20, dtype=np.int64)
    vals = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3,
                     5, 8, 9, 7, 9, 3, 2, 3, 8, 4], dtype=np.float64)

    price = Input("p")
    bars = multi_resample(
        [price, price, price, price],
        [First(), ExpandingMax(), ExpandingMin(), Last()],
        every=W)
    v, k = Dag([price], [bars])((vals, idx))

    assert v.shape == (4, 4)
    for col, agg in enumerate(["first", "max", "min", "last"]):
        rv, rk = resample(vals, idx, every=W, agg=agg)
        np.testing.assert_array_equal(k, rk)
        np.testing.assert_array_equal(v[:, col], rv)


# ---------------------------------------------------------------------------
# advance() parks an empty bucket; a later data event in a further bucket fills
# the parked bucket per the fill policy (locks the intended empty-bar semantics
# that the clock port relies on; see task 6a review Minor #1).
# ---------------------------------------------------------------------------

def test_advance_then_later_event_fills_parked_bucket():
    price = Input("p")
    bars = multi_resample([price], [Last()], every=100, fill="nan")
    dag = Dag([price], [bars])

    live = dag.live()
    live.push("p", 0, 10.0)     # bucket 0
    live.advance(150)           # emit bucket 0 (=10); park empty bucket 1 (open)
    live.push("p", 320, 40.0)   # bucket 3 -> crosses past parked bucket 1 and bucket 2
    live.flush()                # emit trailing bucket 3 (=40)
    v, k = live.result()

    v = np.asarray(v).reshape(-1)
    np.testing.assert_array_equal(k, [0, 100, 200, 300])  # parked bucket 1 IS filled
    assert v[0] == 10.0
    assert np.isnan(v[1])       # parked-then-crossed empty bucket -> NaN fill
    assert np.isnan(v[2])
    assert v[3] == 40.0
