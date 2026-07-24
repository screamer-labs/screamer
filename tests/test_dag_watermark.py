import numpy as np
import pytest
from screamer.dag import Input, Pipeline
from screamer.streams import CombineLatest, Delay


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
