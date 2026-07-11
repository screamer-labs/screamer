import numpy as np
from screamer import RollingMean, Diff, Sub, CombineLatest
from screamer import screamer_bindings as _b


def _row(v):
    v = np.ascontiguousarray(v, dtype=np.float64)
    return np.arange(v.size, dtype=np.int64), v


def test_compile_chain_equals_eager():
    x = np.random.default_rng(0).standard_normal(200)
    g = _b._GraphBuilder()
    xi = g.add_input()
    a = g.add_functor(RollingMean(5), [xi])
    b = g.add_functor(Diff(1), [a])
    g.set_outputs([b])
    (out_k, out_v), = g.run_batch([_row(x)])
    exp = Diff(1)(RollingMean(5)(x))
    np.testing.assert_array_equal(out_v.reshape(-1), exp)
    np.testing.assert_array_equal(out_k, np.arange(x.size, dtype=np.int64))


def test_compile_single_functor():
    x = np.random.default_rng(1).standard_normal(50)
    g = _b._GraphBuilder()
    xi = g.add_input()
    g.set_outputs([g.add_functor(RollingMean(3), [xi])])
    (out_k, out_v), = g.run_batch([_row(x)])
    np.testing.assert_array_equal(out_v.reshape(-1), RollingMean(3)(x))


def test_compile_fanout_and_multi_output():
    x = np.random.default_rng(2).standard_normal(200)
    g = _b._GraphBuilder()
    xi = g.add_input()
    shared = g.add_functor(RollingMean(5), [xi])   # one node...
    d = g.add_functor(Diff(1), [shared])           # ...two consumers
    m = g.add_functor(RollingMean(3), [shared])
    g.set_outputs([d, m])
    (dk, dv), (mk, mv) = g.run_batch([_row(x)])
    sm = RollingMean(5)(x)
    np.testing.assert_array_equal(dv.reshape(-1), Diff(1)(sm))
    np.testing.assert_array_equal(mv.reshape(-1), RollingMean(3)(sm))


def test_compile_output_is_also_intermediate():
    x = np.random.default_rng(3).standard_normal(100)
    g = _b._GraphBuilder()
    xi = g.add_input()
    a = g.add_functor(RollingMean(4), [xi])        # both an output AND feeds b
    b = g.add_functor(Diff(1), [a])
    g.set_outputs([a, b])
    (ak, av), (bk, bv) = g.run_batch([_row(x)])
    np.testing.assert_array_equal(av.reshape(-1), RollingMean(4)(x))
    np.testing.assert_array_equal(bv.reshape(-1), Diff(1)(RollingMean(4)(x)))


def test_compile_combine_then_functor_equals_eager():
    rng = np.random.default_rng(4)
    a_k = np.sort(rng.integers(0, 500, size=150)).astype(np.int64)
    a_v = rng.standard_normal(150)
    b_k = np.sort(rng.integers(0, 500, size=150)).astype(np.int64)
    b_v = rng.standard_normal(150)

    g = _b._GraphBuilder()
    ai, bi = g.add_input(), g.add_input()
    c = g.add_combine_latest([ai, bi], True)       # width-2 aligned
    spread = g.add_functor(Sub(), [c])             # 2-input functor over the width-2 edge
    z = g.add_functor(RollingMean(10), [spread])   # smooth the spread
    g.set_outputs([z])
    (zk, zv), = g.run_batch([(a_k, a_v), (b_k, b_v)])

    aligned, keys = CombineLatest()(a_v, b_v, index=[a_k, b_k])   # when_all, values-first
    exp = RollingMean(10)(aligned[:, 0] - aligned[:, 1])
    np.testing.assert_array_equal(zk, keys)
    np.testing.assert_array_equal(zv.reshape(-1), exp)


def test_compile_combine_same_input_twice():
    from screamer import Sub, CombineLatest as _CL
    x = np.random.default_rng(9).standard_normal(40)
    xk = np.arange(x.size, dtype=np.int64)
    g = _b._GraphBuilder()
    xi = g.add_input()
    c = g.add_combine_latest([xi, xi], True)
    d = g.add_functor(Sub(), [c])
    g.set_outputs([d])
    (dk, dv), = g.run_batch([(xk, x)])
    # Oracle: DAG-1 evaluates CombineLatest()(xi, xi) as CombineLatest()(x, x, index=[xk, xk]).
    aligned, exp_k = _CL()(x, x, index=[xk, xk])
    np.testing.assert_array_equal(dk, exp_k)
    np.testing.assert_array_equal(dv.reshape(-1), aligned[:, 0] - aligned[:, 1])


def test_compile_combine_latest_reset_between_runs():
    """Running the same compiled combine_latest graph twice must give identical output.
    A missed reset leaks buffered state from run 1 into run 2 (wrong values on run 2)."""
    rng = np.random.default_rng(10)
    a_k = np.sort(rng.integers(0, 200, 80)).astype(np.int64)
    a_v = rng.standard_normal(80)
    b_k = np.sort(rng.integers(0, 200, 80)).astype(np.int64)
    b_v = rng.standard_normal(80)
    g = _b._GraphBuilder()
    ai, bi = g.add_input(), g.add_input()
    c = g.add_combine_latest([ai, bi], True)
    z = g.add_functor(RollingMean(5), [g.add_functor(Sub(), [c])])
    g.set_outputs([z])
    (zk1, zv1), = g.run_batch([(a_k, a_v), (b_k, b_v)])
    (zk2, zv2), = g.run_batch([(a_k, a_v), (b_k, b_v)])
    np.testing.assert_array_equal(zk1, zk2)
    np.testing.assert_allclose(np.array(zv1), np.array(zv2), equal_nan=True)


def test_compile_resample_reset_between_runs():
    """Running the same compiled resample graph twice must give identical output."""
    x = np.arange(20.0)
    xk = np.arange(20, dtype=np.int64)
    g = _b._GraphBuilder()
    xi = g.add_input()
    # mode=0 (ByIndex), agg=1 (Last), label=0 (Left), width=5, origin=0, count=0
    rn = g.add_resample([xi], 0, 1, 0, 5, 0, 0)
    g.set_outputs([rn])
    (rk1, rv1), = g.run_batch([(xk, x)])
    (rk2, rv2), = g.run_batch([(xk, x)])
    np.testing.assert_array_equal(rk1, rk2)
    np.testing.assert_allclose(np.array(rv1), np.array(rv2), equal_nan=True)


def test_compile_multi_resample_reset_between_runs():
    """Running the same compiled multi_resample graph twice must give identical output.
    A missed reset leaves buckets half-filled from run 1 (wrong values on run 2)."""
    from screamer import First, Last
    idx = np.arange(20, dtype=np.int64)
    vals = np.arange(20.0)
    g = _b._GraphBuilder()
    xi = g.add_input()
    # mode=0 (ByIndex), label=0 (Left), width=5, origin=0, count=0, fill=0 (skip)
    mn = g.add_multicolumn_resample([xi, xi], 0, 0, 5, 0, 0, 0, [First(), Last()])
    g.set_outputs([mn])
    (mk1, mv1), = g.run_batch([(idx, vals)])
    (mk2, mv2), = g.run_batch([(idx, vals)])
    np.testing.assert_array_equal(mk1, mk2)
    np.testing.assert_allclose(np.array(mv1), np.array(mv2), equal_nan=True)
