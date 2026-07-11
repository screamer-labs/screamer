"""Tests for the lazy pull driver: dag(generators) -> lazy iterator."""

import numpy as np
from screamer import Input, Dag, RollingMean, Sub
from screamer.streams import CombineLatest


def _spread_dag():
    a, b = Input("a"), Input("b")
    return Dag(inputs=[a, b], outputs=[RollingMean(3)(Sub()(CombineLatest()(a, b)))]), a, b


def test_dag_lazy_equals_batch_single_output():
    dag, a, b = _spread_dag()
    va, ia = np.array([10.0, 20.0, 30.0, 40.0, 50.0]), np.array([1, 2, 3, 4, 5])
    vb, ib = np.array([1.0, 2.0, 3.0, 4.0, 5.0]),     np.array([1, 2, 3, 4, 5])

    batch_v, batch_i = dag((va, ia), (vb, ib))          # arrays -> batch

    # generators (lazy iterators) of (value, index) events -> lazy iterator out
    ga = ((float(v), int(k)) for v, k in zip(va, ia))
    gb = ((float(v), int(k)) for v, k in zip(vb, ib))
    out = dag(ga, gb)
    assert hasattr(out, "__next__") and not isinstance(out, tuple)
    rows = list(out)                                     # [(value, index), ...]
    got_v = np.array([r[0] for r in rows])
    got_i = np.array([r[1] for r in rows])
    np.testing.assert_allclose(got_v, np.asarray(batch_v).reshape(-1), equal_nan=True)
    np.testing.assert_array_equal(got_i, np.asarray(batch_i).reshape(-1))


def test_dag_lazy_is_lazy():
    """The driver must pull input events one at a time, not eagerly."""
    from screamer import CumSum
    pulled = []

    def spy(vals):
        for i, v in enumerate(vals):
            pulled.append(v)
            yield (v, i)

    x = Input("x")
    dag = Dag(inputs=[x], outputs=[CumSum()(x)])
    it = dag(spy([1.0, 2.0, 3.0]))
    assert pulled == []            # nothing consumed before first next()
    first = next(it)
    assert pulled == [1.0]         # exactly one input event consumed
    assert first == (1.0, 0)


def test_dag_lazy_equals_batch_multi_output():
    """Multi-output lazy rows must match the batch result column by column."""
    a, b = Input("a"), Input("b")
    spread = Sub()(CombineLatest()(a, b))
    dag = Dag(inputs=[a, b], outputs=[spread, RollingMean(2)(spread)])  # 2 outputs
    va, ia = np.array([10.0, 20.0, 30.0]), np.array([1, 2, 3])
    vb, ib = np.array([1.0, 2.0, 3.0]),   np.array([1, 2, 3])
    batch = dag((va, ia), (vb, ib))                     # tuple of (values, index) pairs
    ga = ((float(v), int(k)) for v, k in zip(va, ia))
    gb = ((float(v), int(k)) for v, k in zip(vb, ib))
    rows = list(dag(ga, gb))                             # rows of (col0, col1, index)
    # Compare against the batch oracle column by column and index by index.
    # batch is a tuple of (values, index) pairs, one per output (co-indexed).
    exp_col0 = np.asarray(batch[0][0]).reshape(-1)       # output 0 (spread)
    exp_col1 = np.asarray(batch[1][0]).reshape(-1)       # output 1 (RollingMean of spread)
    exp_idx = np.asarray(batch[0][1]).reshape(-1)
    assert len(rows) == len(exp_col0)
    np.testing.assert_allclose([r[0] for r in rows], exp_col0, equal_nan=True)
    np.testing.assert_allclose([r[1] for r in rows], exp_col1, equal_nan=True)
    np.testing.assert_array_equal([r[-1] for r in rows], exp_idx)


def test_dag_lazy_align_outputs_false_multi_raises():
    """align_outputs=False multi-output cannot be represented as lazy rows; the
    driver must fail fast rather than silently diverge from batch."""
    import pytest
    from screamer import CumSum, Lag
    x = Input("x")
    s = CumSum()(x)
    dag = Dag(inputs=[x], outputs=[s, Lag(1)(s)], align_outputs=False)  # 2 outputs
    g = ((float(v), int(k)) for v, k in zip([1.0, 2.0, 3.0], [0, 1, 2]))
    with pytest.raises(NotImplementedError):
        list(dag(g))


def test_dag_mixed_lazy_and_concrete_feeds_raises():
    """A mix of a lazy generator and a concrete array feed is ambiguous; raise."""
    import pytest
    a, b = Input("a"), Input("b")
    dag = Dag(inputs=[a, b], outputs=[Sub()(CombineLatest()(a, b))])
    gen_a = ((float(v), int(k)) for v, k in zip([1.0, 2.0], [0, 1]))
    with pytest.raises(TypeError):
        dag(gen_a, (np.array([1.0, 2.0]), np.array([0, 1])))


def test_dag_batch_on_concrete_feed():
    """Rule A: ndarray and list feeds route to the batch path, not lazy."""
    from screamer import CumSum
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[CumSum()(x)])
    # ndarray feed -> batch result (tuple), not a lazy iterator
    arr_out = dag(np.array([1.0, 2.0, 3.0]))
    assert isinstance(arr_out, tuple) and not hasattr(arr_out, "__next__")
    # list feed -> batch result (tuple), not a lazy iterator
    list_out = dag([1.0, 2.0, 3.0])
    assert isinstance(list_out, tuple) and not hasattr(list_out, "__next__")


def test_dag_fractional_index_raises():
    """The engine is int64-indexed; a fractional index must fail loudly (batch and
    lazy) rather than be silently floored. Integer-valued floats (2.0) pass."""
    import pytest
    from screamer import CumSum
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[CumSum()(x)])
    with pytest.raises(TypeError):
        dag((np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.5, 2.0])))      # batch, fractional
    with pytest.raises(TypeError):
        list(dag(((float(v), 0.5 * v) for v in range(3))))               # lazy, fractional
    # integer-valued float indices are lossless and must NOT raise
    v_batch, k_batch = dag((np.array([1.0, 2.0]), np.array([0.0, 1.0])))
    np.testing.assert_array_equal(k_batch, [0, 1])
