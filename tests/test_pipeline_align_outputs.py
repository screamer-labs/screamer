"""Pipeline output alignment in the engine (B7).

align_outputs=True puts the M outputs on one index in C++ (dag::align_outputs
for batch and live, dag::OutputJoin for lazy). These tests check it against the
reference (a batch combine_latest of the outputs, tests/_dag_oracle.py) and
that batch, lazy and live agree. They also check the cumulative resample with
a driver, which the C++ graph builder now joins with its driver.
"""
import numpy as np
import pytest

import screamer as sc
from screamer import Input, Pipeline
from screamer import screamer_bindings as _b
from tests._dag_oracle import _align_results


def _same(got, want):
    assert len(got) == len(want)
    for (gv, gk), (wv, wk) in zip(got, want):
        assert gv.dtype == wv.dtype and gv.shape == wv.shape
        assert gv.tobytes() == wv.tobytes()
        np.testing.assert_array_equal(gk, wk)


def _reference(p, feeds):
    """The reference alignment of the independent outputs of `p`."""
    q = Pipeline(p.inputs, p.outputs, align_outputs=False)
    results = [(k, v) for v, k in q(*feeds)]
    return _align_results(results, True)


def _events(feeds, names):
    evs = []
    for n, (v, k) in zip(names, feeds):
        for j in range(len(k)):
            evs.append((int(k[j]), names.index(n), j, n, v[j]))
    evs.sort(key=lambda e: e[:3])
    return evs


def _lazy(p, feeds):
    gens = [iter([(v[j].tolist() if np.ndim(v[j]) else float(v[j]), int(k[j]))
                  for j in range(len(k))]) for v, k in feeds]
    rows = list(p(*gens))
    m = len(p.outputs)
    k = np.array([r[-1] for r in rows], dtype=np.int64)
    return tuple((np.array([r[j] for r in rows], dtype=np.float64), k) for j in range(m))


def _graphs(rng):
    n = 300
    x = rng.normal(size=n)
    x[rng.random(n) < 0.05] = np.nan
    xi = np.cumsum(rng.integers(0, 3, n)).astype(np.int64)      # repeats
    y = rng.normal(size=(n // 3, 2))
    yi = np.cumsum(rng.integers(1, 5, n // 3)).astype(np.int64)
    m = (rng.random(n) > 0.5).astype(float)
    a, b, mm = Input("a"), Input("b"), Input("m")
    yield [a, b, mm], [sc.RollingMean(4)(a), b, sc.Select(1)(b),
                       sc.Filter()(a, mm), sc.Resample(every=9, agg="ohlc")(a)], \
        [(x, xi), (y, yi), (m, xi)]
    a, b = Input("a"), Input("b")
    yield [a, b], [sc.Delay(7)(a), sc.Resample(count=4, agg="mean")(b)], [(x, xi), (y, yi)]


@pytest.mark.parametrize("seed", range(6))
def test_engine_alignment_matches_the_reference_batch_live_lazy(seed):
    for inputs, outputs, feeds in _graphs(np.random.default_rng(seed)):
        p = Pipeline(inputs, outputs)
        want = _reference(p, feeds)
        got = p(*feeds)
        _same(got, want)
        names = [i.op[1] for i in inputs]
        s = p.live()
        for t, _, _, n, v in _events(feeds, names):
            s.push(n, t, v.tolist() if np.ndim(v) else float(v))
        s.flush()
        _same(s.result(), want)
        for (lv, lk), (wv, wk) in zip(_lazy(p, feeds), want):
            np.testing.assert_array_equal(lk, wk)
            np.testing.assert_array_equal(lv, wv)


def test_aligned_outputs_share_one_index():
    a = Input("a")
    p = Pipeline([a], [a, sc.RollingMean(2)(a)])
    (_, k0), (_, k1) = p(np.arange(5.0))
    assert k0 is k1
    assert k0.dtype == np.int64


def test_lazy_alignment_waits_for_a_late_output_without_a_cap():
    # Output a emits 50 rows before output b emits any; the rows wait, as
    # they do in batch, whatever max_pending is (it caps the join nodes only).
    a, b = Input("a"), Input("b")
    feeds = [(np.arange(50.0), np.arange(50)), (np.array([7.0]), np.array([100]))]
    p = Pipeline([a, b], [a, b], max_pending=10)
    rows = list(p(*[iter([(float(v), int(k)) for v, k in zip(*f)]) for f in feeds]))
    assert rows == [(49.0, 7.0, 100)]
    _same(_lazy(p, feeds), p(*feeds))


def test_lazy_alignment_frees_settled_rows():
    # Two outputs on one input settle every row once both emitted it.
    a = Input("a")
    p = Pipeline([a], [a, sc.RollingMean(3)(a)], max_pending=1)
    n = 10_000
    rows = list(p(iter((float(i), i) for i in range(n))))
    assert len(rows) == n
    assert rows[-1] == (n - 1.0, n - 2.0, n - 1)


def test_threshold_resample_with_a_driver_is_one_node():
    # The graph builder (C++) joins value and driver; the Python graph keeps the
    # node the user wrote, with both streams as inputs.
    v, d = Input("v"), Input("d")
    node = sc.Resample(v, d, threshold=3.0, agg="sum")
    assert node.op[1].__name__ == "Resample"
    assert node.inputs == (v, d)
    p = Pipeline([v, d], [node])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    dr = np.array([1.0, 1.0, 2.0, 0.5, 0.5, 4.0])
    k = np.arange(6)
    # The same as the explicit join.
    j = Input("j")
    q = Pipeline([j], [sc.Resample(j, threshold=3.0, agg="sum")])
    got = p((x, k), (dr, k))
    want = q((np.column_stack([x, dr]), k))
    _same([got], [want])
    np.testing.assert_array_equal(got[0], [6.0, 15.0])
    # A round trip through to_dict rebuilds the same graph.
    r = Pipeline.from_dict(p.to_dict())
    _same([r((x, k), (dr, k))], [got])


def test_threshold_resample_from_the_old_serialized_form():
    # Graphs serialized before B7 hold an explicit CombineLatest node.
    data = {"screamer_dag": 1, "inputs": ["v", "d"], "align_outputs": True,
            "nodes": [{"id": 0, "kind": "input", "name": "v"},
                      {"id": 1, "kind": "input", "name": "d"},
                      {"id": 2, "kind": "operator", "op": "CombineLatest",
                       "params": {"emit": "when_all"}, "in": [0, 1]},
                      {"id": 3, "kind": "operator", "op": "Resample",
                       "params": {"threshold": 3.0, "agg": "sum", "origin": 0,
                                  "label": "left", "fill": "skip"}, "in": [2]}],
            "outputs": [3]}
    p = Pipeline.from_dict(data)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    dr = np.array([1.0, 1.0, 2.0, 0.5, 0.5, 4.0])
    k = np.arange(6)
    v, d = Input("v"), Input("d")
    q = Pipeline([v, d], [sc.Resample(v, d, threshold=3.0, agg="sum")])
    _same([p((x, k), (dr, k))], [q((x, k), (dr, k))])


def test_graph_builder_joins_a_cumulative_driver():
    def run(two_inputs):
        gb = _b._GraphBuilder()
        v, d = gb.add_input(), gb.add_input()
        src = [v, d] if two_inputs else [gb.add_combine_latest([v, d], True)]
        r = gb.add_resample(src, mode=2, agg=4, label=0, width=1, origin=0, count=1,
                            threshold=3.0)
        gb.set_outputs([r])
        k = np.arange(6, dtype=np.int64)
        return gb.compile().run_batch([(k, np.array([1.0, 2, 3, 4, 5, 6])),
                                       (k, np.array([1.0, 1, 2, 0.5, 0.5, 4]))])
    (k2, v2), = run(True)
    (k1, v1), = run(False)
    np.testing.assert_array_equal(k2, k1)
    np.testing.assert_array_equal(v2, v1)


# ---------------------------------------------------------------------------
# Fix round 1: repeated live drains, the overflow, max_pending
# ---------------------------------------------------------------------------

def _concat(parts):
    # A drain with no rows gives an output of unknown width as (0,), as a batch
    # call with no rows does; it adds nothing.
    full = [p for p in parts if len(p[0][1])] or parts[-1:]
    out = []
    for j in range(len(parts[0])):
        vs = [p[j][0] for p in full]
        ks = [p[j][1] for p in full]
        out.append((np.concatenate(vs), np.concatenate(ks)))
    return tuple(out)


@pytest.mark.parametrize("period", [1, 2, 3, 7, 10_000])
def test_repeated_live_drains_concatenated_equal_batch(period):
    for inputs, outputs, feeds in _graphs(np.random.default_rng(11)):
        p = Pipeline(inputs, outputs)
        want = p(*feeds)
        names = [i.op[1] for i in inputs]
        s = p.live()
        parts = []
        for n, (t, _, _, name, v) in enumerate(_events(feeds, names), 1):
            s.push(name, t, v.tolist() if np.ndim(v) else float(v))
            if n % period == 0:
                parts.append(s.result())
        s.flush()
        parts.append(s.result())
        _same(_concat(parts), want)


def test_a_drain_holds_back_rows_another_output_may_still_change():
    # The reported case: a0 b1 a2 b3 a4 a5 b6, result() after events 4 and 6.
    a, b = Input("a"), Input("b")
    p = Pipeline([a, b], [a, b])
    s = p.live()
    ev = [("a", 0, 1.0), ("b", 1, 10.0), ("a", 2, 2.0), ("b", 3, 20.0),
          ("a", 4, 3.0), ("a", 5, 4.0), ("b", 6, 30.0)]
    parts = []
    for n, (name, k, v) in enumerate(ev, 1):
        s.push(name, k, v)
        if n in (4, 6):
            parts.append(s.result())
    s.flush()
    parts.append(s.result())
    assert [list(part[0][1]) for part in parts] == [[1], [2], [3, 4, 5, 6]]
    _same(_concat(parts), p((np.array([1.0, 2, 3, 4]), np.array([0, 2, 4, 5])),
                            (np.array([10.0, 20, 30]), np.array([1, 3, 6]))))


def test_lazy_combine_latest_overflow_ends_the_session():
    # A rejected event ends a lazy call: later next() raises and pulls nothing.
    a, b = Input("a"), Input("b")
    pulled = []

    def gen():
        for i in range(100):
            pulled.append(i)
            yield float(i), i

    p = Pipeline([a, b], [sc.CombineLatest()(a, b), a], max_pending=10)
    it = p(gen(), iter([(1.0, 1000)]))
    with pytest.raises(RuntimeError, match="reorder buffer overflow"):
        next(it)
    n = len(pulled)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="rejected an event"):
            next(it)
    assert len(pulled) == n


def _drain_cases():
    rng = np.random.default_rng(5)
    n = 1000
    x = rng.normal(size=n)
    k = np.arange(n)

    def filter_drops_all(a):         # x has no NaN: the mask is always 0
        return [a, sc.Filter()(a, sc.IsNan()(a))]

    def rare_resample(a):
        return [sc.RollingMean(5)(a), sc.Resample(every=500, agg="mean")(a)]

    def filter_of_an_input(a, m):
        return [a, sc.Filter()(a, m)]

    yield filter_drops_all, [(x, k)]
    yield rare_resample, [(x, k)]
    yield filter_of_an_input, [(x, k), (np.zeros(n), k)]


@pytest.mark.parametrize("case", list(range(3)))
@pytest.mark.parametrize("drains", ["every 1", "every 7", "every 50", "odd", "end"])
def test_live_drains_and_lazy_equal_batch_whatever_the_timing(case, drains):
    make, feeds = list(_drain_cases())[case]
    inputs = [Input(nm) for nm in ("a", "m")[:len(feeds)]]
    p = Pipeline(inputs, make(*inputs), max_pending=100)
    want = p(*feeds)
    names = [i.op[1] for i in inputs]
    evs = _events(feeds, names)
    at = {"every 1": lambda n: True, "every 7": lambda n: n % 7 == 0,
          "every 50": lambda n: n % 50 == 0, "odd": lambda n: n in (5, 6, 333, 999),
          "end": lambda n: False}[drains]
    s = p.live()
    parts = []
    for n, (t, _, _, nm, v) in enumerate(evs, 1):
        s.push(nm, t, float(v))
        if at(n):
            parts.append(s.result())
    s.flush()
    parts.append(s.result())
    _same(_concat(parts), want)
    for (lv, lk), (wv, wk) in zip(_lazy(p, feeds), want):
        np.testing.assert_array_equal(lk, wk)
        np.testing.assert_array_equal(lv, wv)


def test_every_node_kind_round_trips_through_to_dict():
    a, b, c, m = Input("a"), Input("b"), Input("c"), Input("m")
    outs = [sc.RollingMean(3)(a), sc.CombineLatest()(a, b), sc.Dropna()(a),
            sc.Select([1, 0])(sc.CombineLatest()(a, b)),
            sc.Resample(every=5, agg="ohlc")(a), sc.Resample(count=3, agg="mean")(b),
            sc.Resample(a, b, threshold=2.0, agg="sum"),
            sc.Resample(a, c, clock=True, agg="last"), sc.Filter()(a, m), sc.Delay(2)(a)]
    for align in (True, False):
        p = Pipeline([a, b, c, m], outs, align_outputs=align, max_pending=777)
        q = Pipeline.from_json(p.to_json())
        assert q.to_dict() == p.to_dict() and q.max_pending == 777
        rng = np.random.default_rng(1)
        feeds = [(rng.normal(size=50), np.arange(50)), (rng.random(50), np.arange(50)),
                 (np.ones(10), np.arange(0, 50, 5)), ((rng.random(50) > 0.5) * 1.0, np.arange(50))]
        _same(q(*feeds), p(*feeds))


@pytest.mark.parametrize("ok", [1, 5, 2**53, np.int64(7), 5.0, np.float64(9.0)])
def test_max_pending_accepts_an_integer_from_1(ok):
    a = Input("a")
    assert Pipeline([a], [a], max_pending=ok).max_pending == int(ok)


@pytest.mark.parametrize("bad", [0, -1, 2**53 + 1, 10**30, np.int64(0)])
def test_max_pending_out_of_range_is_a_value_error(bad):
    a = Input("a")
    with pytest.raises(ValueError, match="max_pending must be at least 1 and at most 2\\*\\*53"):
        Pipeline([a], [a], max_pending=bad)


@pytest.mark.parametrize("bad", [1.5, float("nan"), float("inf"), True, None, "5",
                                 1 + 0j, np.float64(2.5), [5]])
def test_max_pending_non_integer_is_a_type_error(bad):
    a = Input("a")
    with pytest.raises(TypeError, match="max_pending must be an integer, got"):
        Pipeline([a], [a], max_pending=bad)


def test_max_pending_default_is_the_engine_default():
    a = Input("a")
    assert Pipeline([a], [a]).max_pending == _b._DEFAULT_MAX_PENDING == 1_000_000


@pytest.mark.parametrize("make", [
    lambda a, b: sc.CombineLatest()(a, b),
    lambda a, b: sc.Filter()(a, b),
    lambda a, b: sc.Resample(a, b, threshold=3.0, agg="sum"),
])
def test_max_pending_caps_every_join(make):
    # b arrives only after 10 events of a, so 10 events wait in the join.
    a, b = Input("a"), Input("b")
    feeds = ((np.arange(1.0, 11.0), np.arange(10)), (np.array([1.0]), np.array([20])))
    with pytest.raises(RuntimeError, match="overflow"):
        Pipeline([a, b], [make(a, b)], max_pending=5)(*feeds)
    Pipeline([a, b], [make(a, b)], max_pending=20)(*feeds)


def test_max_pending_round_trips_through_to_dict():
    a = Input("a")
    p = Pipeline([a], [sc.RollingMean(2)(a)], max_pending=12345)
    d = p.to_dict()
    assert d["max_pending"] == 12345
    assert Pipeline.from_dict(d).max_pending == 12345
    assert Pipeline.from_json(p.to_json()).max_pending == 12345
    del d["max_pending"]               # a graph saved before 3.0.0
    assert Pipeline.from_dict(d).max_pending == 1_000_000


# ---------------------------------------------------------------------------
# Fix round 3: watermarks never pass a row a node may still emit; no fill
# toward the int64 maximum
# ---------------------------------------------------------------------------

def _live_every_event(p, feeds, names, advance_at=()):
    s = p.live()
    parts = []
    for t, _, _, nm, v in _events(feeds, names):
        if t in advance_at:
            s.advance(t)
        s.push(nm, t, v.tolist() if np.ndim(v) else float(v))
        parts.append(s.result())
    s.flush()
    parts.append(s.result())
    return _concat(parts)


def test_resample_before_its_first_event_holds_the_watermark_at_its_bucket():
    # The reported case: a NaN prefix that Dropna drops sends watermarks
    # before the resample's first event, which opens the bucket [0, 100).
    a = Input("a")
    x = np.array([np.nan] * 10 + [1.0, 2, 3, 4, 5])
    k = np.arange(15)
    p = Pipeline([a], [a, sc.Resample(every=100, agg="mean")(sc.Dropna()(a))])
    want = p((x, k))
    np.testing.assert_array_equal(want[0][1], k)
    _same(_live_every_event(p, [(x, k)], ["a"]), want)
    _same(_live_every_event(p, [(x, k)], ["a"], advance_at={3, 9}), want)
    for (lv, lk), (wv, wk) in zip(_lazy(p, [(x, k)]), want):
        np.testing.assert_array_equal(lk, wk)
        np.testing.assert_array_equal(lv, wv)
    # A CombineLatest downstream of the same resample keeps the rows too.
    q = Pipeline([a], [sc.CombineLatest()(a, sc.Resample(every=100)(sc.Dropna()(a)))])
    np.testing.assert_array_equal(q((x, k))[1], k)


def _producers():
    """Each watermark producer, feeding a label-left resample beside `a`."""
    def mask_of(a):
        return sc.IsNan()(a)
    return {
        "dropna": lambda a: sc.Dropna()(a),
        "filter": lambda a: sc.Filter()(a, sc.Not()(mask_of(a))),
        "combine_latest": lambda a: sc.Select(0)(sc.CombineLatest()(sc.Dropna()(a), a)),
        "delay": lambda a: sc.Delay(3)(sc.Dropna()(a)),
        "select": lambda a: sc.Select(0)(sc.Dropna()(a)),
        "resample every left": lambda a: sc.Resample(every=7, agg="last")(sc.Dropna()(a)),
        "resample every right": lambda a: sc.Resample(every=7, agg="last", label="right")(sc.Dropna()(a)),
        "resample count": lambda a: sc.Resample(count=3, agg="mean")(sc.Dropna()(a)),
    }


@pytest.mark.parametrize("name", list(_producers()))
@pytest.mark.parametrize("label", ["left", "right"])
@pytest.mark.parametrize("fill", ["skip", "nan", "carry"])
def test_each_watermark_producer_feeding_a_resample(name, label, fill):
    a = Input("a")
    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    x[:12] = np.nan
    x[rng.random(80) < 0.2] = np.nan
    k = np.cumsum(rng.integers(0, 3, 80))
    up = _producers()[name](a)
    p = Pipeline([a], [a, sc.Resample(every=25, agg="mean", label=label, fill=fill)(up)])
    want = p((x, k))
    _same(_live_every_event(p, [(x, k)], ["a"]), want)
    for (lv, lk), (wv, wk) in zip(_lazy(p, [(x, k)]), want):
        np.testing.assert_array_equal(lk, wk)
        np.testing.assert_array_equal(lv, wv)


@pytest.mark.parametrize("fill", ["carry", "nan", "skip"])
@pytest.mark.parametrize("gate", ["combine_latest", "filter"])
def test_flush_does_not_fill_toward_the_int64_maximum(fill, gate):
    # The end-of-input watermark of a CombineLatest or Filter flush closes the
    # open bucket only, as a resample fed directly does (2.8.0 hung here).
    b, m = Input("b"), Input("m")
    v, k = np.array([1.0, 2, 3, 4]), np.array([42, 45, 48, 50])
    if gate == "combine_latest":
        up = sc.Select(0)(sc.CombineLatest()(b, b))
        p = Pipeline([b], [sc.Resample(every=4, agg="last", fill=fill)(up)])
        feeds = [(v, k)]
    else:
        p = Pipeline([b, m], [sc.Resample(every=4, agg="last", fill=fill)(sc.Filter()(b, m))])
        feeds = [(v, k), (np.ones(4), k)]
    direct = Pipeline([b], [sc.Resample(every=4, agg="last", fill=fill)(b)])((v, k))
    got = p(*feeds)
    _same([got], [direct])
    s = p.live()
    names = [i.op[1] for i in p.inputs]
    for t, _, _, nm, val in _events(feeds, names):
        s.push(nm, t, float(val))
    s.flush()
    _same([s.result()], [direct])
    np.testing.assert_array_equal(direct[1], [40, 44, 48])


def test_advance_emits_the_buckets_up_to_now_only():
    a = Input("a")
    p = Pipeline([a], [sc.Resample(every=10, agg="last", fill="nan")(a)])
    s = p.live()
    s.push("a", 0, 1.0)
    s.advance(1_000_000)               # a large jump: 100_000 buckets, no more
    v, k = s.result()
    assert len(k) == 100_000 and k[0] == 0 and k[-1] == 999_990
    assert v[0] == 1.0 and np.isnan(v[1:]).all()
    s.push("a", 1_000_005, 2.0)
    s.advance(2**63 - 1)               # the end of input: the open bucket only
    v, k = s.result()
    assert list(k) == [1_000_000] and list(v) == [2.0]
    s.advance(2**62)                   # no bucket is open: nothing more
    assert len(s.result()[1]) == 0
    q = Pipeline([a], [sc.Resample(every=10, agg="last", fill="carry")(sc.Delay(5)(a))])
    t = q.live()
    t.push("a", 0, 1.0)
    t.advance(2**63 - 1)               # Delay keeps the end-of-input watermark
    assert list(t.result()[1]) == [0]


# ---------------------------------------------------------------------------
# Fix round 4: the end of input emits the open bucket only if it has data, so
# the bars do not depend on what drops the last rows
# ---------------------------------------------------------------------------

def _tail_graphs(fill, label, agg):
    """The same resample fed the rows of `a` that are not NaN, four ways."""
    def R():
        return sc.Resample(every=4, fill=fill, label=label, agg=agg)
    a, m = Input("a"), Input("m")
    return {
        "dropna": Pipeline([a], [R()(sc.Dropna()(a))]),
        "filter": Pipeline([a, m], [R()(sc.Filter()(a, m))]),
        "combine_latest": Pipeline([a], [R()(sc.Select(0)(sc.CombineLatest()(sc.Dropna()(a), sc.Dropna()(a))))]),
        "direct": Pipeline([a], [R()(a)]),
    }


def _tail_feeds(name, x, k):
    keep = ~np.isnan(x)
    if name == "direct":
        return [(x[keep], k[keep])]
    if name == "filter":
        return [(x, k), (keep.astype(float), k)]
    return [(x, k)]


def _live_tail(p, feeds, end):
    names = [i.op[1] for i in p.inputs]
    s = p.live()
    parts = []
    for t, _, _, nm, v in _events(feeds, names):
        s.push(nm, t, float(v))
        parts.append(s.result())
    if end == "advance":
        s.advance(2**63 - 1)
    else:
        s.flush()
    parts.append(s.result())
    return _concat([(q,) for q in parts])[0]


@pytest.mark.parametrize("agg", ["mean", sc.Last()])
@pytest.mark.parametrize("label", ["left", "right"])
@pytest.mark.parametrize("fill", ["skip", "nan", "carry"])
def test_input_that_ends_with_dropped_rows_fills_no_trailing_bucket(fill, label, agg):
    # The reviewer's case: rows 6 to 9 are dropped, so the bucket [8, 12) has
    # no data. Every graph gives bars 0 and 4 (label left), in batch, lazy and
    # live, whether live ends with flush() or advance(2**63 - 1).
    x = np.array([1.0, 2, 3, 4, 5, 6] + [np.nan] * 4)
    k = np.arange(10, dtype=np.int64)
    graphs = _tail_graphs(fill, label, agg)
    want = graphs["direct"](*_tail_feeds("direct", x, k))
    np.testing.assert_array_equal(want[1], [0, 4] if label == "left" else [4, 8])
    for name, p in graphs.items():
        feeds = _tail_feeds(name, x, k)
        _same([p(*feeds)], [want])
        for end in ("flush", "advance"):
            _same([_live_tail(p, feeds, end)], [want])
        (lv, lk), = _lazy(p, feeds)
        np.testing.assert_array_equal(lk, want[1])
        np.testing.assert_array_equal(lv, np.asarray(want[0]).ravel())


@pytest.mark.parametrize("fill", ["nan", "carry"])
def test_dropped_rows_between_bars_still_fill_the_gap(fill):
    a = Input("a")
    x = np.array([1.0] + [np.nan] * 10 + [2.0])
    k = np.arange(12, dtype=np.int64)
    p = Pipeline([a], [a, sc.Resample(every=4, fill=fill, agg="last")(sc.Dropna()(a))])
    want = p((x, k))
    direct = Pipeline([a], [sc.Resample(every=4, fill=fill, agg="last")(a)])((x[[0, 11]], k[[0, 11]]))
    np.testing.assert_array_equal(direct[1], [0, 4, 8])
    alone = Pipeline([a], [sc.Resample(every=4, fill=fill, agg="last")(sc.Dropna()(a))])
    _same([alone((x, k))], [direct])
    _same(_live_every_event(p, [(x, k)], ["a"]), want)
    _same(_live_every_event(p, [(x, k)], ["a"], advance_at={5, 9, 11}), want)
