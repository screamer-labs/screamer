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


def test_live_rejected_first_event_sets_no_width():
    import pytest
    from screamer import Input, Pipeline, Select, RollingCorr
    a = Input("a")
    live = Pipeline([a], [Select(columns=[3])(a)]).live()
    with pytest.raises(Exception):
        live.push("a", 0, [1.0, 2.0])          # too narrow for Select(3)
    live.push("a", 0, [1.0, 2.0, 3.0, 4.0])    # a correct event still works
    np.testing.assert_array_equal(np.ravel(live.result()[0]), [4.0])
    b = Input("b")
    live = Pipeline([b], [RollingCorr(3)(b)]).live()
    with pytest.raises(Exception):
        live.push("b", 0, 1.0)                 # a scalar into a two-column edge
    for k in range(4):
        live.push("b", k, [float(k), float(k * k)])
    assert len(live.result()[0]) == 4


def test_live_wide_edge_into_a_multi_input_functor_equals_batch():
    from screamer import Input, Pipeline, RollingCorr
    a = Input("a")
    p = Pipeline([a], [RollingCorr(5)(a)])
    x = np.random.default_rng(5).standard_normal((30, 2))
    live = p.live()
    for k, row in enumerate(x):
        live.push("a", k, row)
    np.testing.assert_array_equal(live.result()[0], p(x)[0])


def test_live_nested_sequence_is_a_value_error():
    import pytest
    p, live = _identity_live()
    with pytest.raises(ValueError):
        live.push("x", 0, [[1.0, 2.0]])


def test_live_index_kinds_and_bytes():
    import pytest
    p, live = _identity_live()
    live.push("x", np.float32(1.0), 1.0)
    live.push("x", np.float64(2.0), 2.0)
    live.push("x", np.int16(3), 3.0)
    for bad in (float("nan"), np.float32(1.5), float("inf")):
        with pytest.raises(TypeError, match="integer-valued"):
            live.push("x", bad, 1.0)
    for bad in (b"12", bytearray(b"12")):
        with pytest.raises(TypeError, match="bytes"):
            live.push("x", 4, bad)
    np.testing.assert_array_equal(live.result()[1], [1, 2, 3])


def test_live_push_by_position_does_not_leak_references():
    import sys
    p, live = _identity_live()
    big = int("1000007")        # a position that is not a cached small int
    before = sys.getrefcount(big)
    for i in range(1000):
        try:
            live.push(big, i, 1.0)   # out of range: the engine rejects it
        except Exception:
            pass
    assert sys.getrefcount(big) - before < 5


def test_live_wide_list_tuple_and_array_agree():
    p, live = _identity_live()
    live.push("x", 0, [1, 2.5, True])
    live.push("x", 1, (1, 2.5, 1))
    live.push("x", 2, np.array([1, 2.5, 1.0]))
    live.push("x", 3, np.array([1, 2.5, 1], dtype=np.float32))
    live.push("x", 4, np.arange(6.0)[::2])
    v = live.result()[0]
    np.testing.assert_array_equal(v[:4], np.tile([1, 2.5, 1.0], (4, 1)))
    np.testing.assert_array_equal(v[4], [0, 2, 4])


def _reject_then_valid_equals_fresh(build, bad_first, n_cols=4):
    """A rejected first event leaves no trace: the session then equals a fresh one.
    (Sessions of one Pipeline share its graph, so they run one after the other.)"""
    import pytest
    rows = np.random.default_rng(11).standard_normal((6, n_cols)) + 100
    p = build()
    fresh = p.live()
    for k, r in enumerate(rows):
        fresh.push("a", k, r)
    want = fresh.result()
    s1 = p.live()
    with pytest.raises(Exception):
        s1.push("a", 0, bad_first)
    for k, r in enumerate(rows):
        s1.push("a", k, r)
    return s1.result(), want


def test_live_rejected_first_event_with_a_stateful_sibling_equals_fresh():
    from screamer import Input, Pipeline, Select, RollingMean, RollingSum, Diff, RollingCorr
    from screamer.streams import CombineLatest

    def g1():
        a = Input("a")
        return Pipeline([a], [RollingMean(2)(Select(columns=[0])(a)), Select(columns=[3])(a)])

    def g2():
        a = Input("a")
        return Pipeline([a], [RollingSum(3)(Select(columns=[1])(a)),
                              Diff()(Select(columns=[0])(a)), Select(columns=[3])(a)])

    def g3():
        a = Input("a")
        return Pipeline([a], [CombineLatest()(RollingMean(2)(Select(columns=[0])(a)),
                                              Select(columns=[3])(a))])

    def g4():
        a = Input("a")
        return Pipeline([a], [RollingMean(3)(Select(columns=[0])(a)), RollingCorr(3)(a)])

    for build, bad, cols in ((g1, [100.0, 200.0], 4), (g2, [100.0, 200.0], 4),
                             (g3, [100.0, 200.0], 4), (g4, 100.0, 2)):
        v1, v2 = _reject_then_valid_equals_fresh(build, bad, cols)
        out1 = v1 if isinstance(v1[0], tuple) else (v1,)
        out2 = v2 if isinstance(v2[0], tuple) else (v2,)
        assert len(out1) == len(out2)
        for (x1, i1), (x2, i2) in zip(out1, out2):
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(i1, i2)


def test_live_batch_checks_widths_before_feeding():
    import pytest
    from screamer import Input, Pipeline, Select, RollingMean
    a = Input("a")
    p = Pipeline([a], [RollingMean(2)(Select(columns=[0])(a)), Select(columns=[3])(a)])
    with pytest.raises(Exception):
        p(np.ones((5, 2)))
    out = p(np.ones((5, 4)))
    assert len(out) == 2


def test_live_a_rejection_after_part_of_the_graph_took_the_event_needs_reset():
    import pytest
    import screamer.screamer_bindings as _b
    gb = _b._GraphBuilder()
    a, b = gb.add_input(), gb.add_input()
    gb.set_outputs([gb.add_combine_latest([a, b], True, 2)])
    cg = gb.compile()
    with pytest.raises(RuntimeError, match="buffer overflow"):
        for k in range(5):
            cg.push_event(0, k, 1.0)
    with pytest.raises(RuntimeError, match="reset"):
        cg.push_event(1, 0, 1.0)
    cg.reset()
    cg.push_event(1, 0, 1.0)


def test_live_index_must_be_a_real_number():
    import pytest
    from decimal import Decimal
    from fractions import Fraction
    import screamer.screamer_bindings as _b
    p, live = _identity_live()
    for bad in ("5", b"6", Decimal(7), Fraction(8), float("inf"), 2.0**63, 1e300, float("nan")):
        with pytest.raises(TypeError):
            live.push("x", bad, 1.0)
    with pytest.raises(TypeError, match="integer-valued"):
        _b._checked_index(1.5)
    with pytest.raises(TypeError, match="integer-valued"):
        p((np.array([1.0, 2.0]), np.array([0.0, np.nan])))
    live.push("x", 0, 1.0)
    assert len(live.result()[1]) == 1


def test_live_memoryview_and_bytes_like_values_raise():
    import pytest
    p, live = _identity_live()
    for bad in (b"12", bytearray(b"12"), memoryview(b"12")):
        with pytest.raises(TypeError):
            live.push("x", 0, bad)


def test_live_wide_arrays_of_other_dtypes():
    p, live = _identity_live()
    live.push("x", 0, np.array([1, 2, 3, 4], dtype=np.float32))
    live.push("x", 1, np.array([1, 2, 3, 4], dtype=np.int64))
    live.push("x", 2, np.array([True, False, True, True]))
    v = live.result()[0]
    np.testing.assert_array_equal(v, [[1, 2, 3, 4], [1, 2, 3, 4], [1, 0, 1, 1]])
