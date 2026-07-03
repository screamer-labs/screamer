import numpy as np
from screamer import RollingMean, Diff
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
