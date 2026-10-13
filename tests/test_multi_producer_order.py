"""An operator or Dropna with several producers takes their events in index
order (final review finding 1).

Each producer is a separate stream; the node takes them as one stream sorted by
index, events at one index in the order of the node's inputs. Batch, lazy and a
live session whose inputs run ahead of each other give the same rows.
"""
import numpy as np
import pytest

import screamer as sc
from screamer import Input, Pipeline
from screamer.dag import make_operator_node


def feeds2():
    return ((np.array([[1.0, 2.0], [1.0, 1.0]]), np.array([10, 12])),
            (np.array([[3.0, 4.0]]), np.array([5])))


def test_live_out_of_order_operator_matches_batch():
    a, b = Input("a"), Input("b")
    p = Pipeline([a, b], [sc.Add()(a, b)])
    want = p(*feeds2())
    s = p.live()
    s.push("a", 10, [1.0, 2.0])
    s.push("b", 5, [3.0, 4.0])                   # b runs behind a
    s.push("a", 12, [1.0, 1.0])
    s.flush()
    got = s.result()
    assert list(want[1]) == list(got[1]) == [5, 10, 12]
    assert want[0].tobytes() == got[0].tobytes()


def test_no_lookahead_into_an_aligned_row():
    a, b = Input("a"), Input("b")
    p = Pipeline([a, b], [sc.Add()(a, b), sc.CombineLatest()(a, b)])
    want = p(*feeds2())
    s = p.live()
    s.push("a", 10, [1.0, 2.0])
    s.push("b", 5, [3.0, 4.0])
    s.push("a", 12, [1.0, 1.0])
    s.flush()
    got = s.result()
    for (gv, gk), (wv, wk) in zip(got, want):
        assert list(gk) == list(wk) == [10, 12]
        assert gv.tobytes() == wv.tobytes()


def test_multi_producer_dropna_live_out_of_order_matches_batch():
    a, b = Input("a"), Input("b")
    p = Pipeline([a, b], [make_operator_node(sc.Dropna, (a, b), {"how": "any"})])
    fa = (np.array([1.0, np.nan, 3.0]), np.array([1, 4, 9]))
    fb = (np.array([10.0, 20.0]), np.array([2, 9]))
    want = p(fa, fb)
    s = p.live()
    for nm, k, v in [("a", 1, 1.0), ("a", 4, np.nan), ("a", 9, 3.0), ("b", 2, 10.0), ("b", 9, 20.0)]:
        s.push(nm, k, v)
    s.flush()
    got = s.result()
    assert list(want[1]) == list(got[1]) == [1, 2, 9, 9]
    assert list(want[0]) == list(got[0]) == [1.0, 10.0, 3.0, 20.0]
    lazy = list(p(iter([(1.0, 1), (np.nan, 4), (3.0, 9)]), iter([(10.0, 2), (20.0, 9)])))
    assert [r[1] for r in lazy] == [1, 2, 9, 9]


def test_events_at_one_index_go_in_input_order():
    # The second input of the operator is the first input of the Pipeline.
    a, b = Input("a"), Input("b")
    p = Pipeline([a, b], [sc.Sub()(b, a)])
    fa = (np.array([[1.0, 0.0]]), np.array([5]))
    fb = (np.array([[9.0, 0.0]]), np.array([5]))
    v, k = p(fa, fb)
    assert list(k) == [5, 5]
    assert list(v) == [9.0, 1.0]                 # b's event first: it is input 0 of Sub
    s = p.live()
    s.push("a", 5, [1.0, 0.0]).push("b", 5, [9.0, 0.0])
    s.flush()
    lv, lk = s.result()
    assert list(lv) == [9.0, 1.0]


def test_a_stalled_producer_raises_at_max_pending():
    a, b = Input("a"), Input("b")
    p = Pipeline([a, b], [sc.Add()(a, b)], max_pending=3)
    s = p.live()
    s.push("b", 0, [0.0, 0.0])
    with pytest.raises(RuntimeError, match="reorder buffer overflowed"):
        for t in range(1, 10):
            s.push("b", t, [1.0, 1.0])
            s.push("b", t, [1.0, 1.0])

