"""flush() ends a live session's input (final review finding 2).

flush() releases every row of the joins (CombineLatest, Filter, the output
alignment), so an event pushed after it could only come out of order: a
push() or advance() after flush() raises ValueError, and a new live()
session starts over. A repeated flush() and result() are allowed.
"""
import numpy as np
import pytest

import screamer as sc
from screamer import Input, Pipeline

MSG = r"flush\(\) ended this session's input; start a new live\(\) session"


def pipe():
    a, b = Input("a"), Input("b")
    return Pipeline([a, b], [sc.CombineLatest()(a, b)])


def test_push_and_advance_after_flush_raise():
    p = pipe()
    s = p.live()
    s.push("a", 1, 1.0).push("b", 1, 2.0)
    s.flush()
    s.flush()                                    # a repeated flush does nothing
    v, k = s.result()
    assert list(k) == [1]
    with pytest.raises(ValueError, match=MSG):
        s.push("a", 3, 1.0)
    with pytest.raises(ValueError, match=MSG):
        s.push("a", 3, [1.0])
    with pytest.raises(ValueError, match=MSG):
        s.advance(5)
    v, k = s.result()                            # nothing new, and no error
    assert len(k) == 0


def test_a_new_session_after_flush_gives_the_batch_rows():
    p = pipe()
    s = p.live()
    s.push("a", 1, 1.0).push("b", 1, 2.0)
    s.flush()
    s = p.live()
    for nm, t, v in [("a", 3, 1.0), ("b", 2, 2.0), ("a", 4, 3.0), ("b", 5, 4.0)]:
        s.push(nm, t, v)
    s.flush()
    v, k = s.result()
    bv, bk = p((np.array([1.0, 3.0]), np.array([3, 4])), (np.array([2.0, 4.0]), np.array([2, 5])))
    assert list(k) == list(bk) == [3, 4, 5]
    assert v.tobytes() == bv.tobytes()
