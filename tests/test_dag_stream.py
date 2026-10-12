import numpy as np
from screamer import RollingMean, Diff, Sub, Merge
from screamer import screamer_bindings as _b


def test_stream_equals_batch_chain():
    x = np.random.default_rng(0).standard_normal(200)
    xk = np.arange(x.size, dtype=np.int64)

    b = _b._GraphBuilder()
    xi = b.add_input()
    y = b.add_functor(Diff(1), [b.add_functor(RollingMean(5), [xi])])
    b.set_outputs([y])

    cg = b.compile()
    (bk, bv), = cg.run_batch([(xk, x)])          # batch

    cg.reset()
    for k, v in zip(xk, x):                        # streaming: one event at a time
        cg.push_event(0, int(k), float(v))
    (sk, sv), = cg.drain()
    np.testing.assert_array_equal(sk, bk)
    np.testing.assert_array_equal(sv, bv)


def test_stream_equals_batch_combine():
    rng = np.random.default_rng(1)
    a_k = np.sort(rng.integers(0, 500, size=120)).astype(np.int64)
    a_v = rng.standard_normal(120)
    b_k = np.sort(rng.integers(0, 500, size=120)).astype(np.int64)
    b_v = rng.standard_normal(120)

    gb = _b._GraphBuilder()
    ai, bi = gb.add_input(), gb.add_input()
    spread = gb.add_functor(Sub(), [gb.add_combine_latest([ai, bi], True)])
    gb.set_outputs([spread])
    cg = gb.compile()

    (bk, bv), = cg.run_batch([(a_k, a_v), (b_k, b_v)])   # batch

    cg.reset()
    mv, ms, mk = Merge()(a_v, b_v, index=[a_k, b_k])         # merged index-ordered arrays; values-first
    for k, v, src in zip(mk, mv, ms):                      # iterate event-by-event
        cg.push_event(int(src), int(k), float(v))
    cg.flush()                                              # flush pending combine_latest state
    (sk, sv), = cg.drain()
    np.testing.assert_array_equal(sk, bk)
    np.testing.assert_array_equal(sv, bv)


def _wide_live():
    from screamer import Input, Pipeline, Select
    a = Input("bars")
    return Pipeline([a], [Select(columns=[3])(a)]).live()


def test_live_push_accepts_a_wide_event():
    live = _wide_live()
    live.push("bars", 0, [100.0, 102.0, 99.5, 101.5])
    live.push("bars", 1, np.array([101.5, 103.0, 101.0, 102.0]))
    live.push("bars", 2, (1, 2, 3, 4))
    live.push("bars", 3, np.array([True, False, True, True]))
    vals, idx = live.result()
    np.testing.assert_array_equal(np.ravel(vals), [101.5, 102.0, 4.0, 1.0])
    np.testing.assert_array_equal(idx, [0, 1, 2, 3])


def test_live_wide_push_equals_batch():
    from screamer import Input, Pipeline, Select
    a = Input("bars")
    p = Pipeline([a], [Select(columns=[3])(a)])
    rng = np.random.default_rng(1)
    x = rng.standard_normal((20, 4))
    bv, bi = p(x)
    live = p.live()
    for k, row in enumerate(x):
        live.push("bars", k, row)
    sv, si = live.result()
    np.testing.assert_array_equal(sv, bv)
    np.testing.assert_array_equal(si, bi)


def test_live_push_scalar_still_works():
    from screamer import Input, Pipeline, RollingMean
    a = Input("x")
    live = Pipeline([a], [RollingMean(2)(a)]).live()
    for k, v in enumerate([1.0, 2, np.float64(3), True]):
        live.push("x", k, v)
    vals, _ = live.result()
    np.testing.assert_array_equal(np.ravel(vals), [np.nan, 1.5, 2.5, 2.0])


def test_live_push_wide_errors():
    import pytest
    live = _wide_live()
    with pytest.raises(ValueError, match="at least one value"):
        live.push("bars", 0, [])
    with pytest.raises(ValueError, match="1-D"):
        live.push("bars", 0, np.ones((2, 4)))
    with pytest.raises(TypeError):
        live.push("bars", 0, ["a", "b"])
    live.push("bars", 0, [1.0, 2, 3, 4])
    with pytest.raises(ValueError, match="before"):
        live.push("bars", 1, [1.0, 2])
