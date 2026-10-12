"""Pipeline node widths and outputs (Task 18c: A11, G10, G11, G14, G17, Q2).

- A11: co-indexed outputs of different widths keep their own columns.
- G10: Filter gates one value by one mask value; a wide data or mask stream
  raises ValueError before any node sees the event.
- G11: a width mismatch inside a Pipeline is a ValueError at every node.
- G14: ohlcv_bars takes at least 5 columns on every path.
- G17: a batch merge of an int index and a float index runs like the iterator.
- Q2: the float-index iterator merge applies the index kind rule.
"""
import decimal
import fractions

import numpy as np
import pytest

from screamer import (ExpandingMax, Filter, Input, Merge, Pipeline, Resample,
                      RollingMean, Select)
from screamer import screamer_bindings as _b
from screamer.streams import combine_latest, merge, resample


def _same(got, want):
    assert len(got) == len(want)
    for (gv, gk), (wv, wk) in zip(got, want):
        np.testing.assert_array_equal(np.asarray(gv), np.asarray(wv))
        np.testing.assert_array_equal(np.asarray(gk), np.asarray(wk))


def _live(p, events):
    """Push (input, index, value) events into a new session, then result()."""
    s = p.live()
    for name, i, v in events:
        s.push(name, i, v)
    s.flush()
    return s.result()


# ---------------------------------------------------------------------------
# A11: co-indexed outputs of mixed widths
# ---------------------------------------------------------------------------

X3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
K3 = np.arange(3)


def test_mixed_width_outputs_keep_their_columns_batch_and_live():
    a = Input("a")
    p = Pipeline([a], [a, Select(columns=[1])(a), Select(columns=[2, 0])(a), RollingMean(2)(Select(columns=[0])(a))])
    want = [
        (X3, K3),
        (X3[:, 1], K3),
        (X3[:, [2, 0]], K3),
        (np.array([np.nan, 2.5, 5.5]), K3),
    ]
    np.testing.assert_array_equal(want[3][0][1:], [2.5, 5.5])
    got = p((X3, K3))
    for (gv, gk), (wv, wk) in zip(got, want):
        np.testing.assert_array_equal(gv, wv)
        np.testing.assert_array_equal(gk, wk)
    _same(_live(p, [("a", int(k), row) for k, row in zip(K3, X3)]), got)
    lazy = list(p(iter(zip(X3.tolist(), K3.tolist()))))
    assert [r[-1] for r in lazy] == [0, 1, 2]
    for t, row in enumerate(lazy):
        assert row[0] == tuple(X3[t])
        assert row[1] == X3[t, 1]
        assert row[2] == tuple(X3[t, [2, 0]])


def test_the_reported_case():
    a = Input("a")
    p = Pipeline([a], [a, Select(columns=[0])(a)])
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    (v0, k0), (v1, k1) = p(x)
    np.testing.assert_array_equal(v0, x)
    np.testing.assert_array_equal(v1, [1.0, 3.0])
    np.testing.assert_array_equal(k0, [0, 1])
    np.testing.assert_array_equal(k1, [0, 1])


def test_outputs_on_different_indexes_align_as_of():
    # a is 2 columns at 0, 2, 4; b is 1 column at 1, 2, 5; c is 3 columns at 3.
    a, b, c = Input("a"), Input("b"), Input("c")
    p = Pipeline([a, b, c], [a, b, c])
    av = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ak = np.array([0, 2, 4])
    bv = np.array([10.0, 20.0, 30.0])
    bk = np.array([1, 2, 5])
    cv = np.array([[7.0, 8.0, 9.0]])
    ck = np.array([3])
    # Every output has a value from index 3 on; one row per index 3, 4, 5.
    k = np.array([3, 4, 5])
    want = [
        (np.array([[3.0, 4.0], [5.0, 6.0], [5.0, 6.0]]), k),
        (np.array([20.0, 20.0, 30.0]), k),
        (np.array([[7.0, 8.0, 9.0]] * 3), k),
    ]
    got = p((av, ak), (bv, bk), (cv, ck))
    _same(got, want)
    events = sorted([("a", int(i), r) for i, r in zip(ak, av)] +
                    [("b", int(i), float(r)) for i, r in zip(bk, bv)] +
                    [("c", int(i), r) for i, r in zip(ck, cv)], key=lambda e: e[1])
    _same(_live(p, events), want)


def test_independent_outputs_are_unchanged():
    a = Input("a")
    p = Pipeline([a], [a, Select(columns=[0])(a)], align_outputs=False)
    (v0, _), (v1, _) = p(X3)
    np.testing.assert_array_equal(v0, X3)
    np.testing.assert_array_equal(v1, X3[:, 0])


# ---------------------------------------------------------------------------
# G10: Filter widths
# ---------------------------------------------------------------------------

def _filter():
    d, m = Input("d"), Input("m")
    return Pipeline([d, m], [Filter()(d, m)])


@pytest.mark.parametrize("wide", ["data", "mask"])
def test_filter_refuses_a_wide_stream_in_batch(wide):
    one = np.array([1.0, 0.0, 1.0])
    two = np.ones((3, 2))
    data, mask = (two, one) if wide == "data" else (one, two)
    with pytest.raises(ValueError, match=f"Filter takes a 1-column {wide} stream"):
        _filter()(data, mask)
    with pytest.raises(ValueError, match=f"Filter takes a 1-column {wide} stream"):
        Filter()(data, mask)


@pytest.mark.parametrize("name", ["d", "m"])
def test_filter_refuses_a_wide_live_event_before_any_node_sees_it(name):
    p = _filter()
    s = p.live()
    with pytest.raises(ValueError, match="Filter takes a 1-column"):
        s.push(name, 0, [1.0, 1.0])
    # Nothing was pushed or locked: the session runs as a fresh one.
    for k, (x, m) in enumerate([(5.0, 1.0), (6.0, 0.0), (7.0, 2.0)]):
        s.push("d", k, x)
        s.push("m", k, m)
    s.flush()
    _same([s.result()], [p(np.array([5.0, 6.0, 7.0]), np.array([1.0, 0.0, 2.0]))])


def test_filter_of_one_selected_column_runs():
    d, m = Input("d"), Input("m")
    p = Pipeline([d, m], [Filter()(Select(columns=[1])(d), m)])
    v, k = p(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.0, 1.0]))
    np.testing.assert_array_equal(v, [4.0])
    np.testing.assert_array_equal(k, [1])


# ---------------------------------------------------------------------------
# G11: every node raises ValueError for a width it cannot take
# ---------------------------------------------------------------------------

def _graphs():
    a, b = Input("a"), Input("b")
    return {
        "operator": (Pipeline([a], [RollingMean(2)(a)]), "operator takes 1 column"),
        "select": (Pipeline([a], [Select(columns=[2])(a)]), "select needs column 2"),
        "reducer resample": (Pipeline([a], [Resample(every=2, agg=ExpandingMax())(a)]),
                             "reducer resample takes 1 column"),
        "plan resample": (Pipeline([a], [Resample(every=2, agg="ohlcv")(a)]), "plan takes 2"),
        "cumulative resample": (Pipeline([a, b], [Resample(a, b, threshold=5, agg="last")]),
                                "cumulative resample takes"),
        "clock max_age": (Pipeline([a, b], [Resample(a, b, clock=True, agg="last", fill="carry", max_age=3)]),
                          "max_age takes 1 value column"),
        "filter": (Pipeline([a, b], [Filter()(a, b)]), "Filter takes a 1-column data"),
    }


@pytest.mark.parametrize("node", list(_graphs()))
def test_width_mismatch_is_a_value_error_in_batch_and_live(node):
    p, msg = _graphs()[node]
    wide = np.arange(8.0).reshape(4, 2) if node != "plan resample" else np.arange(12.0).reshape(4, 3)
    feeds = [(wide, np.arange(4))]
    if len(p.inputs) == 2:
        feeds.append((np.ones(4), np.arange(4)))
    with pytest.raises(ValueError, match=msg):
        p(*feeds)
    s = p.live()
    if len(p.inputs) == 2:
        s.push("b", 0, 1.0)        # so the graph knows every width but a's
    with pytest.raises(ValueError, match=msg):
        s.push("a", 0, wide[0])


# Past the pre-check. An empty feed leaves its input's width unknown, so the
# pre-check cannot see the width of a CombineLatest(emit="on_any") over it; that
# node emits rows of its compile-time width (one column per input) at run time,
# and the node downstream refuses them itself. These reach the node-level
# std::invalid_argument checks (2.8.0 raised RuntimeError there, or gave a wrong
# result for Filter and ohlcv_bars).

_X = np.array([1.0, 2.0, 3.0, 4.0])
_K = np.arange(4)
_E = (np.array([]), np.array([], dtype=np.int64))


def _node_cases():
    from screamer import CombineLatest
    from screamer.dag import make_operator_node
    from screamer.streams import Resample as _R
    a, b, c, d, m = (Input(n) for n in "abcdm")
    on_any = lambda *xs: CombineLatest(emit="on_any")(*xs)   # noqa: E731
    cumulative = make_operator_node(_R, (on_any(a, b, c),), {
        "threshold": 5, "agg": "last", "every": None, "count": None,
        "origin": 0, "label": "left", "fill": "skip"})
    return {
        "operator": ([a, b], [RollingMean(2)(on_any(a, b))], 0,
                     "an operator takes 1 column"),
        "select": ([a, b], [Select(columns=[5])(on_any(a, b))], 0,
                   "select needs column 5, but its input has 2"),
        "filter": ([a, b, m], [Filter()(on_any(a, b), m)], 1,
                   "Filter takes a 1-column data stream, but its data has 2"),
        "reducer resample": ([a, b], [Resample(every=2, agg=ExpandingMax())(on_any(a, b))], 0,
                             "a reducer resample takes 1 column, but its input has 2"),
        "plan resample": ([a, b, c], [Resample(every=2, agg="ohlcv")(on_any(a, b, c))], 0,
                          "a resample plan takes 2 column"),
        "ohlcv_bars": ([a, b, c, d], [Resample(every=2, agg="ohlcv_bars")(on_any(a, b, c, d))], 0,
                       "at least 5 columns., but its input has 4"),
        "clock max_age": ([a, b, m], [Resample(on_any(a, b), m, clock=True, agg="last",
                                               fill="carry", max_age=3)], 1,
                          "max_age takes 1 value column, but its input has 2"),
        "clock ohlcv_bars": ([a, b, c, d, m], [Resample(on_any(a, b, c, d), m, clock=True,
                                                        agg="ohlcv_bars")], 1,
                             "at least 5 columns., but its input has 4"),
        "cumulative resample": ([a, b, c], [cumulative], 0,
                                "cumulative resample takes \\[value, driver\\] \\(2 columns\\), "
                                "but its input has 3"),
    }


@pytest.mark.parametrize("node", list(_node_cases()))
def test_node_level_width_check_is_a_value_error(node):
    inputs, outputs, n_full, msg = _node_cases()[node]
    p = Pipeline(inputs, outputs)
    # Input a and the last n_full inputs carry data; the others are empty.
    feeds = [(_X, _K)] + [_E] * (len(inputs) - 1 - n_full) + [(_X, _K)] * n_full
    with pytest.raises(ValueError, match=msg):
        p(*feeds)


def test_graph_builder_and_engine_counts_are_value_errors():
    from screamer import Sub
    gb = _b._GraphBuilder()
    i = gb.add_input()
    with pytest.raises(ValueError, match="Filter takes 2 inputs"):
        gb.add_filter([i])
    a = Input("a")
    with pytest.raises(ValueError, match="reducer of 1 input, got one of 2"):
        Pipeline([a], [Resample(every=2, agg=Sub())(a)])
    p = Pipeline([a], [RollingMean(2)(a)])
    with pytest.raises(ValueError, match="expected 1 input arrays, got 0"):
        p._cg.run_batch([])


def test_functor_batch_width_is_a_value_error():
    from screamer import Sub
    with pytest.raises(ValueError, match="takes 2 column"):
        _b._run_functor_batch(Sub(), np.arange(3, dtype=np.int64), np.ones(3))


# ---------------------------------------------------------------------------
# G14: ohlcv_bars takes at least 5 columns on every path
# ---------------------------------------------------------------------------

def _bars(w, n=6):
    return np.arange(n * w, dtype=float).reshape(n, w), np.arange(n)


@pytest.mark.parametrize("w", [1, 2, 3, 4])
def test_ohlcv_bars_needs_five_columns_everywhere(w):
    v, k = _bars(w)
    a, c = Input("a"), Input("c")
    msg = "at least 5 columns" if w > 1 else "ohlcv_bars"
    with pytest.raises(ValueError, match=msg):
        resample((v if w > 1 else v[:, 0], k), every=2, agg="ohlcv_bars")
    with pytest.raises(ValueError, match="at least 5 columns"):
        Pipeline([a], [Resample(every=2, agg="ohlcv_bars")(a)])((v, k))
    with pytest.raises(ValueError, match="at least 5 columns"):
        Pipeline([a, c], [Resample(a, c, clock=True, agg="ohlcv_bars")])((v, k), (np.ones(2), np.array([1, 3])))
    with pytest.raises(ValueError, match="at least 5 columns"):
        Pipeline([a], [Resample(every=2, agg="ohlcv_bars")(a)]).live().push("a", 0, v[0])


def test_ohlcv_bars_with_a_cumulative_driver_raises():
    a, b = Input("a"), Input("b")
    with pytest.raises(ValueError, match="ohlcv_bars"):
        Resample(a, b, threshold=5, agg="ohlcv_bars")


@pytest.mark.parametrize("w", [5, 6])
def test_ohlcv_bars_with_five_or_more_columns_agree(w):
    v, k = _bars(w)
    a = Input("a")
    graph = Pipeline([a], [Resample(every=2, agg="ohlcv_bars")(a)])((v, k))
    eager = resample((v, k), every=2, agg="ohlcv_bars")
    _same([graph], [eager])
    want = np.array([[v[i, 0], v[i:i + 2, 1].max(), v[i:i + 2, 2].min(), v[i + 1, 3],
                      *v[i:i + 2, 4:].sum(axis=0)] for i in (0, 2, 4)])
    np.testing.assert_array_equal(graph[0], want)


# ---------------------------------------------------------------------------
# G17: batch merge of int and float indexes
# ---------------------------------------------------------------------------

AI = (np.array([1.0, 2.0, 3.0]), np.array([0, 2, 4]))
BF = (np.array([10.0, 20.0, 30.0]), np.array([0.0, 2.0, 3.0]))       # whole floats
BFR = (np.array([10.0, 20.0, 30.0]), np.array([0.5, 2.0, 3.5]))      # fractional


def _it(s):
    return zip(s[0].tolist(), s[1].tolist())


@pytest.mark.parametrize("pair", [(AI, BF), (BF, AI), (BFR, AI)], ids=["int,float", "float,int", "frac,int"])
def test_batch_merge_of_int_and_float_index_matches_the_iterator(pair):
    a, b = pair
    vals, src, idx = merge(a, b)
    it = list(merge(_it(a), _it(b)))
    np.testing.assert_array_equal(vals, [e[0] for e in it])
    np.testing.assert_array_equal(idx, [e[1] for e in it])
    np.testing.assert_array_equal(src, [e[2] for e in it])
    # The first index sets the kind, as in the iterator form.
    assert idx.dtype == (np.int64 if isinstance(it[0][1], int) else np.float64)
    m_vals, m_src, m_idx = Merge()(a, b)
    np.testing.assert_array_equal(m_vals, vals)
    np.testing.assert_array_equal(m_idx, idx)


def test_batch_merge_hand_built_order():
    vals, src, idx = merge(AI, BF)
    np.testing.assert_array_equal(vals, [1.0, 10.0, 2.0, 20.0, 30.0, 3.0])
    np.testing.assert_array_equal(src, [0, 1, 0, 1, 1, 0])
    np.testing.assert_array_equal(idx, [0, 0, 2, 2, 3, 4])


@pytest.mark.parametrize("pair", [(AI, BF), (BF, AI)], ids=["int,float", "float,int"])
def test_batch_combine_latest_of_int_and_float_index_matches_the_iterator(pair):
    a, b = pair
    rows, idx = combine_latest(a, b)
    it = list(combine_latest(_it(a), _it(b)))
    np.testing.assert_array_equal(rows, [r for r, _ in it])
    np.testing.assert_array_equal(idx, [i for _, i in it])


def test_fractional_float_after_int_raises_in_batch_and_iterator():
    with pytest.raises(TypeError, match="merge: the index of input 1: index must be integer-valued"):
        merge(AI, BFR)
    with pytest.raises(TypeError, match="integer-valued"):
        list(merge(_it(AI), _it(BFR)))


def test_empty_first_stream_does_not_set_the_kind():
    empty = (np.array([]), np.array([], dtype=np.int64))
    vals, src, idx = merge(empty, BFR)
    np.testing.assert_array_equal(idx, BFR[1])


# ---------------------------------------------------------------------------
# Q2: the float-index iterator merge applies the index kind rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [True, np.bool_(True), decimal.Decimal("2.5"),
                                 fractions.Fraction(5, 2), None, "3", b"3"],
                         ids=["bool", "np.bool_", "Decimal", "Fraction", "None", "str", "bytes"])
def test_float_index_iterator_merge_refuses_a_wrong_kind(bad):
    with pytest.raises(TypeError, match=r"index must be a number \(an int or a float\), got "):
        list(merge(zip([1.0, 2.0], [0.5, bad]), zip([3.0], [1.5])))


def test_float_index_iterator_merge_takes_numbers():
    out = list(merge(zip([1.0, 2.0, 4.0], [0.5, np.int64(2), np.float32(2.5)]),
                     zip([3.0], [1])))
    assert [e[1] for e in out] == [0.5, 1.0, 2.0, 2.5]
    with pytest.raises(TypeError, match="one kind and unit"):
        list(merge(zip([1.0, 2.0], [0.5, np.datetime64(2, "s")]), zip([3.0], [1.5])))


def test_float_index_iterator_merge_rounds_a_big_int_as_batch_does():
    big = 2**53 + 1
    it = list(merge(zip([1.0, 2.0], [0.5, big]), zip([3.0], [1.5])))
    _, _, idx = merge((np.array([1.0, 2.0]), [0.5, big]), (np.array([3.0]), np.array([1.5])))
    assert [e[1] for e in it] == list(idx)
    assert it[-1][1] == float(big)
    with pytest.raises(TypeError, match="index does not fit a float64, got an int of 401 digits"):
        list(merge(zip([1.0, 2.0], [0.5, -10**400]), zip([3.0], [1.5])))


@pytest.mark.parametrize("first", [AI, BFR], ids=["int first", "float first"])
def test_batch_merge_refuses_a_bool_index_like_the_iterator(first):
    bools = (np.array([1.0, 2.0]), np.array([False, True]))
    with pytest.raises(TypeError, match="index must be a number"):
        merge(first, bools)
    with pytest.raises(TypeError, match="index must be a number"):
        list(merge(_it(first), _it(bools)))
