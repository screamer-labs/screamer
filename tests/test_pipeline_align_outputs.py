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


def test_lazy_alignment_is_bounded_by_max_pending():
    # Output a emits 50 rows before output b emits any, so 50 rows wait.
    a, b = Input("a"), Input("b")
    feeds = [(np.arange(50.0), np.arange(50)), (np.array([7.0]), np.array([100]))]

    def gens():
        return [iter([(float(v), int(k)) for v, k in zip(*f)]) for f in feeds]

    p = Pipeline([a, b], [a, b], max_pending=10)
    with pytest.raises(RuntimeError, match="output alignment overflow: more than 10 output rows"):
        list(p(*gens()))
    q = Pipeline([a, b], [a, b], max_pending=50)
    rows = list(q(*gens()))
    assert rows == [(49.0, 7.0, 100)]
    _same(_lazy(q, feeds), q(*feeds))


def test_lazy_alignment_frees_settled_rows():
    # Two outputs on one input settle every row once both emitted it, so a long
    # run never holds more than a few rows, far below max_pending.
    a = Input("a")
    p = Pipeline([a], [a, sc.RollingMean(3)(a)], max_pending=4)
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
