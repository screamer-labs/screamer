import itertools

import numpy as np
import pytest
from screamer.dag import Input, Pipeline
from screamer.streams import CombineLatest, Delay, Dropna, Filter, Resample


def _fused_delay_merge_batch_vs_live(duration, idx, vals):
    """Build Pipeline([x], [CombineLatest()(x, Delay(d)(x))]); return (batch, live)
    each as a sorted list of (index, tuple(row)). Live pushes each event then flushes."""
    x = Input("x")
    pipe = Pipeline([x], [CombineLatest()(x, Delay(duration)(x))])

    batch_v, batch_i = pipe((vals, idx))          # batch: whole arrays
    sess = pipe.live()
    for t, v in zip(idx.tolist(), vals.tolist()):
        sess.push("x", int(t), float(v))
    sess.flush()
    live_v, live_i = sess.result()

    def as_rows(v, i):
        v = np.asarray(v); i = np.asarray(i)
        order = np.argsort(i, kind="stable")
        return [(int(i[k]), tuple(np.ravel(v[k]).tolist())) for k in order]
    return as_rows(batch_v, batch_i), as_rows(live_v, live_i)


def test_fused_delay_merge_batch_equals_live_regular():
    idx = (np.arange(12, dtype=np.int64)) * 10          # regular grid 0,10,...,110
    vals = np.arange(12, dtype=float)
    batch, live = _fused_delay_merge_batch_vs_live(30, idx, vals)
    assert live == batch


def test_fused_delay_merge_batch_equals_live_irregular():
    rng = np.random.default_rng(1)
    idx = np.cumsum(rng.integers(1, 9, size=150)).astype(np.int64)
    vals = rng.standard_normal(150)
    for d in (1, 4, 25, 100):
        batch, live = _fused_delay_merge_batch_vs_live(d, idx, vals)
        assert live == batch, f"mismatch at duration {d}"


def test_two_input_delayed_merge_matches_asof_oracle():
    """Two-input delayed merge: live pipeline output matches a hand-derived as-of oracle.

    This test is the primary correctness check for the reorder-buffer fix in
    CombineLatestNode. It uses TWO DISTINCT inputs (a and b), which is the topology
    that exposes the causality bug. A single-input topology does NOT exercise the bug
    because both ports are fed through the same synchronous Broadcast path.

    Setup:
      a at indices [10, 20, 30, 40], values [1.0, 2.0, 3.0, 4.0]
      b at indices [ 8, 15, 25],     values [-1.0, -2.0, -3.0]
      Delay(d=5) applied to b -> delayed-b at indices [13, 20, 30]

    Pipeline: CombineLatest()(a, Delay(5)(b))
    Default emit="when_all": rows are only produced once BOTH slots have a value.

    Oracle derivation (as-of semantics, distinct-index coalescing):
      k=10: a@10=1.0, delayed-b: no index <=10 yet -> skip (when_all)
      k=13: a  as-of <=13 = a@10   = 1.0, delayed-b@13 = -1.0 -> row=[1.0, -1.0]
      k=20: a@20=2.0 and delayed-b@20=-2.0 arrive at the same index
            -> coalesced: row=[2.0, -2.0]
      k=30: a@30=3.0, delayed-b@30=-3.0 (same index again)
            -> coalesced: row=[3.0, -3.0]
      k=40: a@40=4.0, delayed-b as-of <=40 = delayed-b@30 = -3.0
            -> row=[4.0, -3.0]

    Pre-fix bug: delayed-b@13 (raw b@8 shifted by +5) was applied to the merge
    at the wrong time (before its index was safe), so the merge saw it aligned
    against a stale a value (a had no event at or before 8, so NaN), causing
    when_all to never emit k=13, and k=40 to carry a wrong delayed-b value.
    The oracle test would FAIL against pre-fix output.
    """
    a = Input("a")
    b = Input("b")
    pipe = Pipeline([a, b], [CombineLatest()(a, Delay(5)(b))])

    av = np.array([1.0, 2.0, 3.0, 4.0])
    ai = np.array([10, 20, 30, 40], dtype=np.int64)
    bv = np.array([-1.0, -2.0, -3.0])
    bi = np.array([8, 15, 25], dtype=np.int64)

    # Hand-computed oracle (see derivation in docstring)
    expected = [
        (13, (1.0, -1.0)),
        (20, (2.0, -2.0)),
        (30, (3.0, -3.0)),
        (40, (4.0, -3.0)),
    ]

    # --- live path ---
    sess = pipe.live()
    # Interleave events in global raw-arrival order (sorted by raw input index).
    # a arrives at [10,20,30,40], b arrives at [8,15,25]; merged by raw index:
    #   b@8, a@10, b@15, a@20, b@25, a@30, a@40
    events_a = list(zip(ai.tolist(), av.tolist()))
    events_b = list(zip(bi.tolist(), bv.tolist()))
    merged = sorted(
        [("a", t, v) for t, v in events_a] + [("b", t, v) for t, v in events_b],
        key=lambda e: e[1],
    )
    for src, t, v in merged:
        sess.push(src, t, v)
    sess.flush()
    live_v, live_i = sess.result()

    live_v = np.asarray(live_v)
    live_i = np.asarray(live_i)
    order = np.argsort(live_i, kind="stable")
    live_rows = [(int(live_i[k]), tuple(np.ravel(live_v[k]).tolist())) for k in order]

    assert len(live_rows) == len(expected), (
        f"expected {len(expected)} rows, got {len(live_rows)}: {live_rows}"
    )
    for (got_idx, got_row), (exp_idx, exp_row) in zip(live_rows, expected):
        assert got_idx == exp_idx, f"index mismatch: got {got_idx}, expected {exp_idx}"
        for g, e in zip(got_row, exp_row):
            np.testing.assert_equal(np.float64(g), np.float64(e),
                                    err_msg=f"value mismatch at k={exp_idx}")

    # --- batch path (secondary: just checks it agrees with live) ---
    batch_v, batch_i = pipe((av, ai), (bv, bi))
    batch_v = np.asarray(batch_v)
    batch_i = np.asarray(batch_i)
    ob = np.argsort(batch_i, kind="stable")
    batch_rows = [(int(batch_i[k]), tuple(np.ravel(batch_v[k]).tolist())) for k in ob]
    assert batch_rows == live_rows, (
        f"batch and live disagree:\n  batch={batch_rows}\n  live={live_rows}"
    )


def test_two_input_delayed_merge_batch_equals_live():
    """Batch and live give identical results for two-input pipelines over random data.

    This is a secondary invariant: for any delay duration and any pair of random
    indexed streams (a and b as distinct inputs), the batch and live paths of
    Pipeline([a, b], [CombineLatest()(a, Delay(d)(b))]) must agree exactly.

    The test sweeps over d in {1, 7, 40, 200} with a seeded random generator to
    cover short, medium, and large delays relative to typical inter-event gaps.
    """
    rng = np.random.default_rng(42)
    n_a, n_b = 80, 60

    for d in (1, 7, 40, 200):
        ai = np.cumsum(rng.integers(1, 12, size=n_a)).astype(np.int64)
        av = rng.standard_normal(n_a)
        bi = np.cumsum(rng.integers(1, 12, size=n_b)).astype(np.int64)
        bv = rng.standard_normal(n_b)

        a = Input("a")
        b = Input("b")
        pipe = Pipeline([a, b], [CombineLatest()(a, Delay(d)(b))])

        # Batch
        batch_v, batch_i = pipe((av, ai), (bv, bi))
        batch_v = np.asarray(batch_v)
        batch_i = np.asarray(batch_i)
        ob = np.argsort(batch_i, kind="stable")

        # Live: interleave events in raw-arrival order
        events = sorted(
            [("a", int(t), float(v)) for t, v in zip(ai, av)]
            + [("b", int(t), float(v)) for t, v in zip(bi, bv)],
            key=lambda e: e[1],
        )
        sess = pipe.live()
        for src, t, v in events:
            sess.push(src, t, v)
        sess.flush()
        live_v, live_i = sess.result()
        live_v = np.asarray(live_v)
        live_i = np.asarray(live_i)
        ol = np.argsort(live_i, kind="stable")

        np.testing.assert_array_equal(
            batch_i[ob], live_i[ol],
            err_msg=f"index mismatch at d={d}",
        )
        np.testing.assert_allclose(
            batch_v[ob], live_v[ol],
            rtol=0, atol=0,
            err_msg=f"value mismatch at d={d}",
        )


def test_watermark_propagates_through_a_functor():
    # A functor between the input and a delayed merge must forward the watermark,
    # or the merge would stall and diverge from batch.
    from screamer import RollingMean
    x = Input("x")
    pipe = Pipeline([x], [CombineLatest()(RollingMean(3)(x), Delay(20)(x))])
    idx = (np.arange(15, dtype=np.int64)) * 10
    vals = np.arange(15, dtype=float)
    bv, bi = pipe((vals, idx))
    s = pipe.live()
    for t, v in zip(idx.tolist(), vals.tolist()):
        s.push("x", int(t), float(v))
    s.flush()
    lv, li = s.result()
    ob, ol = np.argsort(bi, kind="stable"), np.argsort(li, kind="stable")
    np.testing.assert_array_equal(np.asarray(li)[ol], np.asarray(bi)[ob])
    np.testing.assert_allclose(np.asarray(lv)[ol], np.asarray(bv)[ob])


def test_dropna_before_delayed_merge_releases_incrementally():
    """DropNa before Delay+merge must forward watermarks for dropped frames.

    Without forward-on-drop, a run of NaNs in b stalls b's port watermark at the
    last kept (delayed) index. The merge then holds every row from a that arrived
    during the NaN run, releasing nothing until flush(). With the fix, dropped
    frames emit on_watermark(f.index), so the merge's b-port watermark advances
    and a's rows are released incrementally before flush().

    PRIMARY assertion: result() BEFORE flush() returns >= 5 rows.
    SECONDARY assertion: batch == live after flush().
    """
    a, b = Input("a"), Input("b")
    # b is Dropna'd then delayed by 5 before merging with a.
    pipe = Pipeline([a, b], [CombineLatest()(a, Delay(5)(Dropna()(b)))])
    s = pipe.live()

    # b: one real value at index 0 (delayed to 5), then a long run of NaN.
    s.push("b", 0, 1.0)
    for t in range(1, 20):
        s.push("b", t, float("nan"))   # all dropped; must still advance watermark

    # a: values at even indices 0..38; many land AFTER delayed-b watermark of 5.
    # Without forward-on-drop, b's port watermark is stuck at 5 (last kept delayed
    # index), so nothing beyond index 5 in a can be released incrementally.
    for t in range(0, 40, 2):
        s.push("a", t, float(t))

    # Collect BEFORE flush - this is the observable: rows that were released
    # incrementally (i.e., without needing flush to drain the reorder buffer).
    mid = s.result()
    mid_v = mid[0] if isinstance(mid, tuple) else mid
    mid_arr = np.asarray(mid_v) if mid_v is not None else np.array([])

    # Must have released at least 5 rows before flush; without the fix this is 0
    # or 1 (only the row at the delayed-b index 5 is safe).
    assert len(mid_arr) >= 5, (
        f"Expected >= 5 incrementally released rows before flush, got {len(mid_arr)}. "
        "DropNa is not forwarding watermarks past dropped frames."
    )

    # Secondary: batch == live (full agreement). result() drains incrementally, so
    # we must combine pre-flush (mid) and post-flush results to get the full live
    # output, then compare against batch.
    s.flush()
    rest_v, rest_i = s.result()

    # mid is (values_2d, index_1d); rest is the same shape.
    mid_v_arr = np.asarray(mid[0]) if len(mid[0]) > 0 else np.empty((0, 2))
    mid_i_arr = np.asarray(mid[1])
    rest_v_arr = np.asarray(rest_v) if len(rest_v) > 0 else np.empty((0, 2))
    rest_i_arr = np.asarray(rest_i)
    live_v = np.concatenate([mid_v_arr, rest_v_arr], axis=0)
    live_i = np.concatenate([mid_i_arr, rest_i_arr])

    a2, b2 = Input("a"), Input("b")
    pipe2 = Pipeline([a2, b2], [CombineLatest()(a2, Delay(5)(Dropna()(b2)))])
    av = np.array([float(t) for t in range(0, 40, 2)], dtype=float)
    ai = np.arange(0, 40, 2, dtype=np.int64)
    bv = np.concatenate([[1.0], [float("nan")] * 19])
    bi = np.arange(0, 20, dtype=np.int64)
    batch_v, batch_i = pipe2((av, ai), (bv, bi))

    batch_i = np.asarray(batch_i)
    ob = np.argsort(batch_i, kind="stable")
    ol = np.argsort(live_i, kind="stable")
    np.testing.assert_array_equal(batch_i[ob], live_i[ol],
                                  err_msg="batch and live index mismatch")
    np.testing.assert_allclose(np.asarray(batch_v)[ob], live_v[ol],
                               rtol=0, atol=0,
                               err_msg="batch and live value mismatch")


def test_reorder_buffer_overflow_raises():
    from screamer.dag import Input, Pipeline
    from screamer.streams import CombineLatest, Delay
    a, b = Input("a"), Input("b")
    pipe = Pipeline([a, b], [CombineLatest()(a, Delay(1)(b))], max_pending=8)
    s = pipe.live()
    with pytest.raises(RuntimeError):
        for t in range(100):
            s.push("b", t, float(t))    # a never advances -> b's delayed rows pile up


def test_advance_releases_idle_delayed_buffer():
    """advance(now) must release buffered delayed-b rows when a goes idle.

    When input a stops producing events, the delayed-b rows accumulate in
    CombineLatest's reorder buffer. Calling advance(now) injects a watermark at
    input a's sink, which propagates through the graph and unblocks the merge.
    Without the Task-3 fix, advance() only reaches resample nodes directly and
    the merge never sees the watermark, so result() stays empty.
    """
    a, b = Input("a"), Input("b")
    # b is delayed and merged against a. a stops early; advance(now) must let the
    # buffered delayed-b rows settle against a's last value.
    pipe = Pipeline([a, b], [CombineLatest()(a, Delay(5)(b))])
    s = pipe.live()
    s.push("a", 0, 10.0)
    s.push("b", 0, 1.0)     # delayed to index 5
    s.push("b", 3, 2.0)     # delayed to index 8
    # Before advance: delayed-b rows are buffered; a is idle so merge cannot
    # safely release them (it does not know a has stopped).
    v_before, i_before = s.result()
    assert (i_before is None or len(np.asarray(i_before)) == 0), (
        "Expected no rows before advance; merge should not have released anything yet"
    )
    s.advance(100)          # a is idle; advancing time releases the buffered rows
    v, i = s.result()
    # both delayed b rows are now settled against a-as-of (a=10.0 at every index >=0)
    assert i is not None and len(np.asarray(i)) >= 2, (
        f"Expected >= 2 rows after advance(100), got {len(np.asarray(i)) if i is not None else 0}. "
        "advance(now) is not propagating as a watermark through the graph."
    )


@pytest.mark.parametrize("d,emit", list(itertools.product([1, 7, 50], ["on_any", "when_all"])))
def test_two_input_delayed_merge_sweep(d, emit):
    from screamer.dag import Input, Pipeline
    from screamer.streams import CombineLatest, Delay
    rng = np.random.default_rng((d * 7 + len(emit)) % (2**32))
    n = 120
    ia = np.cumsum(rng.integers(1, 6, size=n)).astype(np.int64)
    va = rng.standard_normal(n)
    ib = np.cumsum(rng.integers(1, 6, size=n)).astype(np.int64)
    vb = rng.standard_normal(n)
    a, b = Input("a"), Input("b")
    cl = CombineLatest(emit=emit)
    pipe = Pipeline([a, b], [cl(a, Delay(d)(b))])
    bv, bi = pipe((va, ia), (vb, ib))
    s = pipe.live()
    # feed a and b events interleaved by raw arrival index (a real event loop order)
    events = sorted([("a", int(t), float(v)) for t, v in zip(ia, va)] +
                    [("b", int(t), float(v)) for t, v in zip(ib, vb)],
                    key=lambda e: e[1])
    for name, t, v in events:
        s.push(name, t, v)
    s.flush()
    lv, li = s.result()
    ob, ol = np.argsort(bi, kind="stable"), np.argsort(li, kind="stable")
    np.testing.assert_array_equal(np.asarray(li)[ol], np.asarray(bi)[ob])
    np.testing.assert_allclose(np.asarray(lv)[ol], np.asarray(bv)[ob], equal_nan=True)


def test_filter_with_delayed_input_asof_oracle():
    """Filter with a DELAYED mask input must gate as-of the merged index.

    FilterNode is a 2-port fan-in (port 0 = data, port 1 = mask) built on
    CombineLatest(2, when_all=true). A delayed sibling (Delay on the mask) makes the
    mask's frames future-dated relative to its raw index. Without a reorder buffer,
    the node applies events in ARRIVAL order and merges the future-dated mask frame
    too early against a stale data value, gating rows at the wrong index. This is the
    same causality bug the reorder buffer fixed in CombineLatestNode.

    Setup (two DISTINCT inputs, the topology that exposes the bug):
      data d at indices [10, 20, 30, 40], values [1.0, 2.0, 3.0, 4.0]
      mask m at indices [ 8, 15, 25],     values [1.0, 0.0, 1.0]
      Delay(5) applied to m -> delayed-m at indices [13, 20, 30]

    Pipeline: Filter()(d, Delay(5)(m))  (when_all: fires once both ports seen)

    Oracle derivation (as-of gate, distinct-index coalescing):
      k=10: data d@10=1.0, delayed-m: no index <=10 -> when_all not yet fired -> skip
      k=13: data as-of <=13 = d@10 = 1.0, mask delayed-m@13 (raw m@8) = 1.0
            -> gate KEEP -> emit data 1.0 at index 13
      k=20: data d@20=2.0, mask delayed-m@20 (raw m@15) = 0.0 -> gate DROP
      k=30: data d@30=3.0, mask delayed-m@30 (raw m@25) = 1.0 -> gate KEEP -> emit 3.0
      k=40: data d@40=4.0, mask as-of <=40 = delayed-m@30 = 1.0 -> KEEP -> emit 4.0

    Pre-fix (arrival order) the first KEEP lands at index 10 (data merged with a mask
    that had not yet reached its true index 13). The oracle KEEP indices are 13,30,40.
    """
    d = Input("d")
    m = Input("m")
    pipe = Pipeline([d, m], [Filter()(d, Delay(5)(m))])

    dv = np.array([1.0, 2.0, 3.0, 4.0])
    di = np.array([10, 20, 30, 40], dtype=np.int64)
    mv = np.array([1.0, 0.0, 1.0])
    mi = np.array([8, 15, 25], dtype=np.int64)

    expected = [(13, 1.0), (30, 3.0), (40, 4.0)]  # 20 dropped

    # --- live path ---
    merged = sorted(
        [("d", int(t), float(v)) for t, v in zip(di, dv)]
        + [("m", int(t), float(v)) for t, v in zip(mi, mv)],
        key=lambda e: e[1],
    )
    sess = pipe.live()
    for src, t, v in merged:
        sess.push(src, t, v)
    sess.flush()
    live_v, live_i = sess.result()
    live_v = np.asarray(live_v)
    live_i = np.asarray(live_i)
    order = np.argsort(live_i, kind="stable")
    live_rows = [(int(live_i[k]), float(np.ravel(live_v[k])[0])) for k in order]

    assert live_rows == expected, f"live {live_rows} != oracle {expected}"

    # --- batch path (must agree with live) ---
    batch_v, batch_i = pipe((dv, di), (mv, mi))
    batch_v = np.asarray(batch_v)
    batch_i = np.asarray(batch_i)
    ob = np.argsort(batch_i, kind="stable")
    batch_rows = [(int(batch_i[k]), float(np.ravel(batch_v[k])[0])) for k in ob]
    assert batch_rows == expected, f"batch {batch_rows} != oracle {expected}"
