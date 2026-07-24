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
