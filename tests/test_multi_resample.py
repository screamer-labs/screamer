"""Multi-column collect-and-reset resample bar node (dag::MultiResampleNode).

One bucketing clock shared across N input ports, each reduced by its own EvalOp,
emitting one aligned N-wide row per bar. The core guarantee: an N-wide row from a
single clock equals N separately-run single-column `resample(...)` columns (same
labels AND values). Also proves batch == stream, advance/flush semantics, and
per-column NaN-ignore.
"""
import numpy as np
import pytest

from screamer.dag import Input, Dag
from screamer.streams import multi_resample, resample
from screamer import (First, Last, ExpandingMax, ExpandingMin, ExpandingSum,
                      PosPart, NegPart)


# ---------------------------------------------------------------------------
# 1. Alignment vs stacked single resamples (the core one-clock guarantee)
# ---------------------------------------------------------------------------

def test_alignment_matches_stacked_single_resamples():
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

    assert v.shape == (4, 4)   # 20 / 5 = 4 bars, 4 columns

    for col, agg in enumerate(["first", "max", "min", "last"]):
        rv, rk = resample(vals, idx, every=W, agg=agg)
        np.testing.assert_array_equal(k, rk)
        np.testing.assert_array_equal(v[:, col], rv)


# ---------------------------------------------------------------------------
# 2. Two different inputs (OHLCV2 shape): price + signed volume
# ---------------------------------------------------------------------------

def test_two_inputs_ohlcv2_shape():
    W = 4
    idx = np.arange(12, dtype=np.int64)
    price = np.array([10, 11, 9, 12, 13, 8, 7, 14, 15, 6, 5, 16], dtype=np.float64)
    vol = np.array([2, -3, 4, -1, 5, -6, 7, -2, 3, -4, 1, -8], dtype=np.float64)

    p = Input("p")
    q = Input("v")
    bars = multi_resample(
        [p, p, PosPart()(q), NegPart()(q)],
        [First(), Last(), ExpandingSum(), ExpandingSum()],
        every=W)
    v, k = Dag([p, q], [bars])((price, idx), (vol, idx))

    assert v.shape == (3, 4)

    # column 0/1: first/last of price
    rv, rk = resample(price, idx, every=W, agg="first")
    np.testing.assert_array_equal(k, rk)
    np.testing.assert_array_equal(v[:, 0], rv)
    rv, _ = resample(price, idx, every=W, agg="last")
    np.testing.assert_array_equal(v[:, 1], rv)

    # column 2: sum of PosPart(vol); column 3: sum of NegPart(vol)
    pos = np.where(vol > 0, vol, 0.0)
    neg = np.where(vol < 0, -vol, 0.0)   # NegPart returns positive magnitude
    rv, _ = resample(pos, idx, every=W, agg="sum")
    np.testing.assert_array_equal(v[:, 2], rv)
    rv, _ = resample(neg, idx, every=W, agg="sum")
    np.testing.assert_array_equal(v[:, 3], rv)


# ---------------------------------------------------------------------------
# 3. batch == stream (manual live push + flush coalescing)
# ---------------------------------------------------------------------------

def test_batch_equals_stream_single_input():
    W = 5
    idx = np.arange(20, dtype=np.int64)
    vals = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3,
                     5, 8, 9, 7, 9, 3, 2, 3, 8, 4], dtype=np.float64)

    price = Input("p")
    bars = multi_resample(
        [price, price, price, price],
        [First(), ExpandingMax(), ExpandingMin(), Last()],
        every=W)
    dag = Dag([price], [bars])

    vb, kb = dag((vals, idx))

    live = dag.live()
    for k_i, v_i in zip(idx, vals):
        live.push("p", int(k_i), float(v_i))
    live.flush()
    vs, ks = live.result()

    np.testing.assert_array_equal(kb, ks)
    np.testing.assert_array_equal(vb, vs)


def test_batch_equals_stream_two_inputs():
    W = 4
    idx = np.arange(12, dtype=np.int64)
    price = np.array([10, 11, 9, 12, 13, 8, 7, 14, 15, 6, 5, 16], dtype=np.float64)
    vol = np.array([2, -3, 4, -1, 5, -6, 7, -2, 3, -4, 1, -8], dtype=np.float64)

    p = Input("p")
    q = Input("v")
    bars = multi_resample(
        [p, p, PosPart()(q), NegPart()(q)],
        [First(), Last(), ExpandingSum(), ExpandingSum()],
        every=W)
    dag = Dag([p, q], [bars])

    vb, kb = dag((price, idx), (vol, idx))

    live = dag.live()
    # push price then its co-indexed vol at each timestamp (index order)
    for k_i, pv, vv in zip(idx, price, vol):
        live.push("p", int(k_i), float(pv))
        live.push("v", int(k_i), float(vv))
    live.flush()
    vs, ks = live.result()

    np.testing.assert_array_equal(kb, ks)
    np.testing.assert_array_equal(vb, vs)


# ---------------------------------------------------------------------------
# 4. advance(now) closes empty bars: fill="nan" and fill="carry"
# ---------------------------------------------------------------------------

def test_advance_closes_empty_bars_nan():
    p = Input("p")
    bars = multi_resample([p, p], [First(), Last()], every=100, origin=0, fill="nan")
    dag = Dag([p], [bars])
    live = dag.live()
    live.push("p", 0, 10.0)
    live.advance(350)   # close bucket 0 (real), 100 & 200 (empty), 300 stays open
    live.flush()        # bucket 3 empty -> nothing
    v, k = live.result()
    np.testing.assert_array_equal(k, [0, 100, 200])
    assert v.shape == (3, 2)
    np.testing.assert_array_equal(v[0], [10.0, 10.0])
    assert np.all(np.isnan(v[1]))
    assert np.all(np.isnan(v[2]))


def test_advance_closes_empty_bars_carry():
    p = Input("p")
    bars = multi_resample([p, p], [First(), Last()], every=100, origin=0, fill="carry")
    dag = Dag([p], [bars])
    live = dag.live()
    live.push("p", 0, 10.0)
    live.push("p", 20, 12.0)   # bucket 0: First=10, Last=12
    live.advance(350)
    live.flush()
    v, k = live.result()
    np.testing.assert_array_equal(k, [0, 100, 200])
    np.testing.assert_array_equal(v[0], [10.0, 12.0])
    np.testing.assert_array_equal(v[1], [10.0, 12.0])   # carried
    np.testing.assert_array_equal(v[2], [10.0, 12.0])


# ---------------------------------------------------------------------------
# 5. On-demand flush mid-stream emits the partial N-wide bar once
# ---------------------------------------------------------------------------

def test_flush_mid_stream_emits_partial_once():
    p = Input("p")
    bars = multi_resample([p, p], [First(), Last()], every=100, origin=0)
    dag = Dag([p], [bars])
    live = dag.live()
    live.push("p", 0, 10.0)
    live.push("p", 50, 20.0)   # both bucket 0
    live.flush()               # emit partial bucket 0 once (label 0)
    live.push("p", 120, 30.0)  # bucket 1
    live.flush()               # emit partial bucket 1 (label 100)
    v, k = live.result()
    np.testing.assert_array_equal(k, [0, 100])   # exactly 2 rows, no duplicate
    np.testing.assert_array_equal(v[0], [10.0, 20.0])
    np.testing.assert_array_equal(v[1], [30.0, 30.0])


# ---------------------------------------------------------------------------
# 6. NaN-ignore per column: a port seeing only NaN emits NaN for its columns
# ---------------------------------------------------------------------------

def test_nan_ignore_per_column():
    W = 4
    idx = np.arange(4, dtype=np.int64)
    price = np.array([10.0, 11.0, 12.0, 13.0])
    qvals = np.array([np.nan, np.nan, np.nan, np.nan])   # port sees only NaN

    p = Input("p")
    q = Input("q")
    bars = multi_resample([p, q], [Last(), Last()], every=W)
    v, k = Dag([p, q], [bars])((price, idx), (qvals, idx))

    assert v.shape == (1, 2)
    assert v[0, 0] == 13.0        # price column finite
    assert np.isnan(v[0, 1])      # all-NaN port column is NaN
