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


def _identity_live():
    from screamer import Input, Pipeline
    a = Input("x")
    p = Pipeline([a], [a])
    return p, p.live()


def test_live_push_wide_errors():
    import pytest
    live = _wide_live()
    with pytest.raises(ValueError, match="at least one value"):
        live.push("bars", 0, [])
    with pytest.raises(ValueError, match="1-D"):
        live.push("bars", 0, np.ones((2, 4)))
    with pytest.raises(ValueError, match="1-D"):
        live.push("bars", 0, [[1.0, 2.0]])
    with pytest.raises(ValueError):
        live.push("bars", 0, ["a", "b"])
    with pytest.raises(KeyError):
        live.push("nope", 0, 1.0)


def test_live_width_is_locked_per_input_until_reset():
    import pytest
    p, live = _identity_live()
    live.push("x", 0, [1.0, 2.0, 3.0, 4.0])
    for bad in (5.0, [1.0, 2.0], np.float64(1.0), True):
        with pytest.raises(ValueError, match="values after events of 4"):
            live.push("x", 1, bad)
    live.push("x", 1, [5.0, 6.0, 7.0, 8.0])
    vals, idx = live.result()
    np.testing.assert_array_equal(vals, [[1, 2, 3, 4], [5, 6, 7, 8]])
    np.testing.assert_array_equal(idx, [0, 1])
    # scalar first, then wide
    live = p.live()
    live.push("x", 0, 5.0)
    with pytest.raises(ValueError, match="values after events of 1"):
        live.push("x", 1, [1.0, 2.0, 3.0, 4.0])
    live.push("x", 1, 6)
    np.testing.assert_array_equal(np.ravel(live.result()[0]), [5.0, 6.0])
    # a new session starts with a fresh width
    live = p.live()
    live.push("x", 0, [1.0, 2.0])
    np.testing.assert_array_equal(live.result()[0], [[1.0, 2.0]])
    live = p.live()
    live.push("x", 0, 9.0)
    np.testing.assert_array_equal(np.ravel(live.result()[0]), [9.0])


def test_live_cg_reset_clears_the_width():
    p, live = _identity_live()
    live.push("x", 0, [1.0, 2.0])
    live._cg.reset()
    live.push("x", 0, 3.0)
    np.testing.assert_array_equal(np.ravel(live.result()[0]), [3.0])


def test_live_wide_missing_values_and_strings_follow_the_batch_feed():
    p, live = _identity_live()
    live.push("x", 0, [None, 1.0])
    live.push("x", 1, ["2", "3"])
    live.push("x", 2, np.array([np.nan, 4.0]))
    batch = np.asarray([[None, 1.0], ["2", "3"], [np.nan, 4.0]], dtype=np.float64)
    np.testing.assert_array_equal(live.result()[0], p(batch)[0])
    _, s = _identity_live()
    s.push("x", 0, None)
    s.push("x", 1, "3")
    np.testing.assert_array_equal(np.ravel(s.result()[0]), [np.nan, 3.0])


def test_live_push_index_must_be_integer_valued():
    import pytest
    p, live = _identity_live()
    live.push("x", 1.0, 1.0)
    live.push("x", np.int64(2), np.float32(2.5))
    with pytest.raises(TypeError, match="integer-valued"):
        live.push("x", 3.5, 1.0)
    np.testing.assert_array_equal(live.result()[1], [1, 2])


def test_live_wide_event_with_a_reordering_select():
    from screamer import Input, Pipeline, Select
    a = Input("a")
    x = np.random.default_rng(3).standard_normal((15, 3))
    b = Pipeline([a], [Select(columns=[2, 0])(a)])
    live = b.live()
    for k, row in enumerate(x):
        live.push("a", k, row)
    np.testing.assert_array_equal(live.result()[0], b(x)[0])


def test_live_width_change_after_a_drain_still_raises():
    import pytest
    p, live = _identity_live()
    live.push("x", 0, [1.0, 2.0])
    live.result()
    with pytest.raises(ValueError):
        live.push("x", 1, 3.0)
