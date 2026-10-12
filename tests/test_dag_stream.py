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
    with pytest.raises(TypeError, match="must be a number"):   # G13
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
    for bad, msg in ((float("nan"), "finite"), (np.float32(1.5), "integer-valued"),
                     (float("inf"), "finite")):
        with pytest.raises(TypeError, match=msg):
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


def test_live_bytes_like_values_raise():
    import pytest
    p, live = _identity_live()
    for bad in (b"12", bytearray(b"12"), memoryview(b"12"), memoryview(bytearray(b"12")),
                memoryview(b"12").cast("c")):
        with pytest.raises(TypeError):
            live.push("x", 0, bad)


def test_live_numeric_memoryview_is_a_wide_event():
    import array
    p, live = _identity_live()
    live.push("x", 0, memoryview(np.array([1.0, 2.0])))
    live.push("x", 1, memoryview(array.array("i", [3, 4])))
    live.push("x", 2, memoryview(np.array([5, 6], dtype=np.uint8)))
    np.testing.assert_array_equal(live.result()[0], [[1, 2], [3, 4], [5, 6]])


def test_live_misaligned_buffers_read_correctly():
    p, live = _identity_live()
    raw = bytearray(8 * 3 + 1)
    d = np.frombuffer(raw, dtype=np.float64, count=3, offset=1)
    d[:] = [1.5, 2.5, 3.5]
    i = np.frombuffer(bytearray(4 * 3 + 1), dtype=np.int32, count=3, offset=1)
    i[:] = [7, 8, 9]
    assert not d.flags.aligned and not i.flags.aligned
    live.push("x", 0, d)
    live.push("x", 1, i)
    np.testing.assert_array_equal(live.result()[0], [[1.5, 2.5, 3.5], [7, 8, 9]])


# --- the index rule (checked_index.h): every Python entry point -------------

_BAD_INDEX = [
    ("5", "must be a number"), (None, "must be a number"), (True, "must be a number"),
    (np.bool_(True), "must be a number"), (b"6", "must be a number"),
    ("decimal", "must be a number"), ("fraction", "must be a number"),
    (1.5, "integer-valued"), (np.float32(2.5), "integer-valued"),
    (float("nan"), "finite"), (float("inf"), "finite"), (-float("inf"), "finite"),
    (2**63, "int64 range"), (2.0**63, "int64 range"), (1e300, "int64 range"),
    (np.uint64(2**63), "int64 range"), (-2**63 - 1, "int64 range"),
]


def _bad_index(v):
    from decimal import Decimal
    from fractions import Fraction
    return {"decimal": Decimal(5), "fraction": Fraction(5)}.get(v, v) if isinstance(v, str) else v


def test_index_rule_live_push_and_advance():
    import pytest
    from screamer import Input, Pipeline, Resample
    a = Input("x")
    p = Pipeline([a], [Resample(every=2, agg="last")(a)])
    for bad, msg in _BAD_INDEX:
        live = p.live()
        with pytest.raises(TypeError, match=msg):
            live.push("x", _bad_index(bad), 1.0)
        with pytest.raises(TypeError, match=msg):
            live.advance(_bad_index(bad))
    live = p.live()
    for k in (0, 1.0, np.int64(2), np.uint64(3), np.float32(4.0), 2**63 - 1):
        live.push("x", k, 1.0)
    live.advance(np.int32(5))
    live.advance(2.0**62)


def test_index_rule_batch():
    import pytest
    from decimal import Decimal
    from screamer import Input, Pipeline
    a = Input("x")
    p = Pipeline([a], [a])
    v = np.array([1.0, 2.0])
    for idx, msg in [
        (np.array([0.0, 1.5]), "integer-valued"),
        (np.array([0.0, np.nan]), "finite"),
        (np.array([0.0, np.inf]), "finite"),
        (np.array([0.0, 2.0**63]), "int64 range"),
        (np.array([0, 2**63], dtype=np.uint64), "int64 range"),
        ([0, 2**63], "int64 range"),
        (np.array([True, False]), "must be a number"),
        (["0", "1"], "must be a number"),
        ([Decimal(0), Decimal(1)], "must be a number"),
        ([0, None], "must be a number"),
        (np.array([0, 1j]), "must be a number"),
    ]:
        with pytest.raises(TypeError, match=msg):
            p((v, idx))
    for idx in (np.array([0, 1]), np.array([0.0, 1.0]), np.array([0, 1], dtype=np.uint64),
                np.array([0, 1], dtype=np.int8), np.array([0.0, 1.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float16), [0, 1], np.array([0, 9, 1])[::2]):
        np.testing.assert_array_equal(p((v, idx))[1], [0, 1])
    t = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(p((v, t))[1], t.view(np.int64))


def test_index_rule_lazy_feed():
    import pytest
    from screamer import Input, Pipeline
    a = Input("x")
    p = Pipeline([a], [a])
    for bad, msg in [(1.5, "integer-valued"), (float("nan"), "finite"), (2.0**63, "int64 range")]:
        with pytest.raises(TypeError, match=msg):
            list(p(iter([(1.0, 0), (2.0, bad)])))


# --- complex values ----------------------------------------------------------

def test_complex_values_raise_in_live_push_and_batch():
    import pytest
    p, live = _identity_live()
    for bad in (1j, np.complex128(1), np.complex64(1), np.clongdouble(1), [1j, 2.0],
                [np.complex128(1), 2.0], np.array([1 + 0j, 2]), np.array(1 + 0j)):
        with pytest.raises(TypeError, match="complex"):
            live.push("x", 0, bad)
    with pytest.raises(TypeError, match="complex"):
        p(np.array([1 + 0j, 2]))
    with pytest.raises(TypeError, match="complex"):
        p((np.array([[1 + 0j, 2]]), np.array([0])))
    # object arrays keep the operators' behavior
    np.testing.assert_array_equal(p(np.array([1, 2], dtype=object))[0], [1.0, 2.0])


# --- a functor fed by several nodes --------------------------------------------

def test_functor_inputs_must_each_carry_its_columns():
    import pytest
    from screamer import Input, Pipeline, Select, Sub, RollingCorr
    a, b = Input("a"), Input("b")
    p = Pipeline([a], [Sub()(Select(columns=[0])(a), Select(columns=[1])(a))])
    with pytest.raises(ValueError, match="CombineLatest"):
        p(np.ones((4, 2)))
    with pytest.raises(ValueError, match="CombineLatest"):
        p.live().push("a", 0, [1.0, 2.0])
    # Two producers that each carry the operator's columns run (interleaved).
    q = Pipeline([a, b], [RollingCorr(3)(a, b)])
    x = np.arange(12.0).reshape(6, 2)
    assert len(q(x, x + 100)[0]) == 12


# --- learned resample plans are forgotten on reset ---------------------------

def _clock_and_bars():
    from screamer import Input, Pipeline, Resample
    x, c = Input("x"), Input("c")
    return Pipeline([x, c], [Resample(x, c, clock=True, agg="last"),
                             Resample(every=100, agg="ohlcv_bars")(x)],
                    align_outputs=False)


def _bars(width, n=12):
    v = np.arange(float(n * width)).reshape(n, width)
    return (v, np.arange(n)), (np.ones(4), np.array([2, 5, 8, 11]))


def _same(a, b):
    for (av, ak), (bv, bk) in zip(a, b):
        np.testing.assert_array_equal(av, bv)
        np.testing.assert_array_equal(ak, bk)


def test_reused_pipeline_forgets_learned_resample_plans():
    p = _clock_and_bars()
    for widths in ((5, 6), (6, 5), (5, 4)):
        for w in widths:
            _same(p(*_bars(w)), _clock_and_bars()(*_bars(w)))


def test_live_after_a_rejected_event_equals_a_fresh_session():
    import pytest
    p = _clock_and_bars()
    live = p.live()
    with pytest.raises(ValueError, match="ohlcv_bars"):
        live.push("x", 0, [1.0, 2.0])          # rejected before any node sees it
    for width in (4, 6, 5):
        (v, k), (cv, ck) = _bars(width)
        live = p.live()
        for row, i in zip(v, k):
            live.push("x", int(i), row)
            if i in ck:
                live.push("c", int(i), 1.0)
        live.flush()
        _same(live.result(), _clock_and_bars()(*_bars(width)))


def test_live_rejected_fixed_plan_width_does_not_poison():
    import pytest
    from screamer import Input, Pipeline, Resample
    a = Input("a")
    p = Pipeline([a], [Resample(every=2, agg="ohlcv")(a)])
    live = p.live()
    for bad in ([1.0, 2.0, 3.0], 1.0):
        with pytest.raises(ValueError, match="plan takes 2"):
            live.push("a", 0, bad)
    x = np.arange(12.0).reshape(6, 2)
    for k, row in enumerate(x):
        live.push("a", k, row)
    live.flush()
    _same([live.result()], [p(x)])


def test_clock_resample_of_wide_values():
    import pytest
    from screamer import Input, Pipeline, Resample
    x, c = Input("x"), Input("c")
    (v, k), clk = _bars(3)
    ohlc = Pipeline([x, c], [Resample(x, c, clock=True, agg="ohlc")])(( v, k), clk)[0]
    first = Pipeline([x, c], [Resample(x, c, clock=True, agg="first")])((v, k), clk)[0]
    last = Pipeline([x, c], [Resample(x, c, clock=True, agg="last")])((v, k), clk)[0]
    assert ohlc.shape == (4, 12)
    np.testing.assert_array_equal(ohlc[:, 0::4], first)
    np.testing.assert_array_equal(ohlc[:, 3::4], last)
    (v5, k5), clk5 = _bars(5)
    bars = Pipeline([x, c], [Resample(x, c, clock=True, agg="ohlcv_bars")])((v5, k5), clk5)[0]
    assert bars.shape == (4, 5)
    with pytest.raises(ValueError, match="plan takes 2"):
        Pipeline([x, c], [Resample(x, c, clock=True, agg="ohlcv")])((v, k), clk)


# --- a poisoned session -------------------------------------------------------

def test_poisoned_session_result_raises_until_reset():
    import pytest
    from screamer import Input, Pipeline, CombineLatest
    a, b = Input("a"), Input("b")
    p = Pipeline([a, b], [CombineLatest()(a, b)], max_pending=2)
    live = p.live()
    with pytest.raises(RuntimeError, match="buffer overflow"):
        for k in range(5):
            live.push("a", k, 1.0)
    for call in (live.result, live.flush, lambda: live.advance(9),
                 lambda: live.push("b", 0, 1.0)):
        with pytest.raises(RuntimeError, match="reset"):
            call()
    live = p.live()
    live.push("a", 0, 1.0)
    live.push("b", 0, 2.0)
    live.flush()
    np.testing.assert_array_equal(live.result()[0], [[1.0, 2.0]])


def test_live_wide_arrays_of_other_dtypes():
    p, live = _identity_live()
    live.push("x", 0, np.array([1, 2, 3, 4], dtype=np.float32))
    live.push("x", 1, np.array([1, 2, 3, 4], dtype=np.int64))
    live.push("x", 2, np.array([True, False, True, True]))
    v = live.result()[0]
    np.testing.assert_array_equal(v, [[1, 2, 3, 4], [1, 2, 3, 4], [1, 0, 1, 1]])


def test_pandas_nullable_values_read_na_as_nan():
    import pandas as pd
    p, _ = _identity_live()
    for v, want in ((pd.Series([True, None], dtype="boolean"), [1.0, np.nan]),
                    (pd.array([True, None], dtype="boolean"), [1.0, np.nan]),
                    (pd.array([1, None], dtype="Int64"), [1.0, np.nan]),
                    (pd.array([1.5, None], dtype="Float64"), [1.5, np.nan]),
                    ([True, None], [1.0, np.nan])):
        np.testing.assert_array_equal(p(v)[0], want)
        live = p.live()
        live.push("x", 0, v)
        np.testing.assert_array_equal(live.result()[0], [want])


def test_index_rule_iterator_feeds():
    import pytest
    from decimal import Decimal
    from fractions import Fraction
    from screamer import Merge
    p, _ = _identity_live()
    for bad, msg in ((np.float32(1.5), "integer-valued"), (Decimal("1.5"), "must be a number"),
                     (Fraction(3, 2), "must be a number"), ("5", "must be a number"),
                     (b"6", "must be a number"), (True, "must be a number"),
                     (np.bool_(True), "must be a number"), (2**63, "int64 range"),
                     (np.uint64(2**63), "int64 range"), (float("nan"), "finite")):
        with pytest.raises(TypeError, match=msg):
            list(p(iter([(1.0, 0), (2.0, bad)])))
        with pytest.raises(TypeError, match=msg):
            list(Merge()(iter([(1.0, 0), (2.0, bad)]), iter([(3.0, 1)])))
    out = list(p(iter([(1.0, np.int64(0)), (2.0, np.float32(1.0)), (3.0, 2.0)])))
    assert [k for _, k in out] == [0, 1, 2]


def test_object_index_array_follows_the_live_rule():
    import pytest
    p, _ = _identity_live()
    v = np.array([1.0, 2.0])
    np.testing.assert_array_equal(p((v, np.array([0, 1], dtype=object)))[1], [0, 1])
    np.testing.assert_array_equal(p((v, [0, np.int64(1)]))[1], [0, 1])
    with pytest.raises(TypeError, match="int64 range"):
        p((v, [0, 2**64]))
    with pytest.raises(TypeError, match="must be a number"):
        p((v, np.array([0, True], dtype=object)))
