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
