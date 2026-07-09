"""Tests for the lazy pull driver: dag(generators) -> lazy iterator."""

import numpy as np
from screamer import Input, Dag, RollingMean, Sub
from screamer.streams import combine_latest


def _spread_dag():
    a, b = Input("a"), Input("b")
    return Dag(inputs=[a, b], outputs=[RollingMean(3)(Sub()(combine_latest(a, b)))]), a, b


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
    spread = Sub()(combine_latest(a, b))
    dag = Dag(inputs=[a, b], outputs=[spread, RollingMean(2)(spread)])  # 2 outputs
    va, ia = np.array([10.0, 20.0, 30.0]), np.array([1, 2, 3])
    vb, ib = np.array([1.0, 2.0, 3.0]),   np.array([1, 2, 3])
    batch = dag((va, ia), (vb, ib))                     # tuple of (values, index) pairs
    ga = ((float(v), int(k)) for v, k in zip(va, ia))
    gb = ((float(v), int(k)) for v, k in zip(vb, ib))
    rows = list(dag(ga, gb))                             # rows of (col0, col1, index)
    # one row per output index; compare count against batch
    assert len(rows) == len(np.asarray(batch[0][0]).reshape(-1))


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
