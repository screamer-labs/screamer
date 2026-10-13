"""G18: an empty result keeps the output's width when it is known.

The width comes from the graph (an operator's outputs, a Select, a reducer
Resample) or from the input (an empty (0, 3) array is three columns wide), in
batch and in live drains, aligned or not. Only a width-1 output is (0,). A
live session before its first event does not know an input's width, so an
output that passes that width through falls back to one column.
"""
import numpy as np
import pytest

import screamer as s
from screamer import Input, Pipeline

E3 = np.zeros((0, 3))
NO_INDEX = np.zeros(0, dtype=np.int64)


def shape(result):
    return result[0].shape


def test_empty_wide_batch_keeps_its_width():
    a = Input("a")
    assert shape(Pipeline([a], [a])(E3)) == (0, 3)
    assert shape(Pipeline([a], [a])(np.zeros(0))) == (0,)
    assert shape(Pipeline([a], [a])(np.zeros((0, 0)))) == (0, 0)
    assert shape(Pipeline([a], [s.Dropna()(a)])(E3)) == (0, 3)
    assert shape(Pipeline([a], [s.Delay(1)(a)])((E3, NO_INDEX))) == (0, 3)
    assert shape(Pipeline([a], [s.Resample(count=2, agg="ohlc")(a)])(E3)) == (0, 12)
    assert shape(Pipeline([a], [s.Resample(count=2, agg="ohlc")(a)])(np.zeros(0))) == (0, 4)
    b = Input("b")
    assert shape(Pipeline([a, b], [s.CombineLatest()(a, b)])(E3, np.zeros(0))) == (0, 4)


def test_empty_multi_output_batch_keeps_each_width():
    a, b = Input("a"), Input("b")
    for align in (True, False):
        p = Pipeline([a, b], [a, s.RollingMean(2)(b)], align_outputs=align)
        (va, _), (vb, _) = p(E3, np.zeros(0))
        assert (va.shape, vb.shape) == ((0, 3), (0,))


def test_graph_widths_are_known_before_any_event():
    a = Input("a")
    for node, width in ((lambda x: s.BollingerBands(3)(x), 3),
                        (lambda x: s.Select([0, 2])(x), 2),
                        (lambda x: s.Resample(count=2, agg=s.BollingerBands(3))(x), 3)):
        p = Pipeline([a], [node(a)])
        assert shape(p.live().result()) == (0, width)
        assert shape(p(np.zeros(0) if width == 3 else E3)) == (0, width)
    # A passthrough of an input whose width is not known yet: one column.
    assert shape(Pipeline([a], [a]).live().result()) == (0,)


@pytest.mark.parametrize("align", [True, False])
def test_an_empty_live_drain_keeps_the_learned_width(align):
    a, b = Input("a"), Input("b")
    p = Pipeline([a, b], [a, s.RollingMean(2)(b)], align_outputs=align)
    live = p.live()
    live.push("a", 0, [1.0, 2.0, 3.0])
    live.push("b", 0, 1.0)
    live.push("a", 1, [1.0, 2.0, 3.0])
    live.push("b", 1, 2.0)
    first = live.result()
    assert first[0][0].shape[1:] == (3,)
    (va, _), (vb, _) = live.result()                    # nothing new: empty
    assert (va.shape, vb.shape) == ((0, 3), (0,))


def test_a_known_width_reaches_the_aligned_join_before_any_row():
    # The join is laid out (output 0 settled rows) while output 1 has no row
    # and its input's width is not known. Once it is, the empty drains show
    # it, and the first row has it.
    a, b = Input("a"), Input("b")
    p = Pipeline([a, b], [s.RollingMean(1)(a), s.Resample(count=2)(b)])
    live = p.live()
    live.push("a", 0, 1.0)
    live.advance(5)
    (_, _), (vb, _) = live.result()
    assert vb.shape == (0,)
    live.push("b", 6, [5.0, 6.0])
    (_, _), (vb, _) = live.result()
    assert vb.shape == (0, 2)
    live.push("b", 7, [7.0, 8.0])
    live.flush()
    (_, _), (vb, ib) = live.result()
    assert vb.tolist() == [[7.0, 8.0]] and list(ib) == [6]


def test_eager_stream_operators_keep_an_empty_wide_width():
    feed = (E3, NO_INDEX)
    assert s.Resample(count=2, agg="ohlc")(E3)[0].shape == (0, 3, 4)
    assert s.Resample(count=2)(E3)[0].shape == (0, 3)
    assert s.Resample(freq=2, agg="ohlc")(feed)[0].shape == (0, 3, 4)
    assert s.Resample(freq=2, fill="carry")(feed)[0].shape == (0, 3)
    assert s.Resample(freq=2)((np.zeros((0, 3, 2)), NO_INDEX))[0].shape == (0, 3, 2)
    assert s.Resample(feed, (np.zeros(0), NO_INDEX), clock=True)[0].shape == (0, 3)
    assert s.Delay(1)(feed)[0].shape == (0, 3)
    assert s.Dropna()(feed)[0].shape == (0, 3)
    assert s.Select([0, 1])(feed)[0].shape == (0, 2)
    assert s.CombineLatest()(feed, (np.zeros(0), NO_INDEX))[0].shape == (0, 4)
