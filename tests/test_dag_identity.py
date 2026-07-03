import numpy as np
import pytest
from screamer import RollingMean, Diff, Sub, Add, Input, Dag, combine_latest
from tests._dag_oracle import run_oracle


def _row(v):
    v = np.ascontiguousarray(v, dtype=np.float64)
    return np.arange(v.size, dtype=np.int64), v


def _series(size, seed):
    rng = np.random.default_rng(seed)
    k = np.sort(rng.integers(0, size * 3, size=size)).astype(np.int64)
    return k, rng.standard_normal(size)


def _chain():
    x = Input("x")
    return Dag(inputs=[x], outputs=[Diff(1)(RollingMean(5)(x))]), [_row(np.random.default_rng(0).standard_normal(150))]


def _fanout():
    x = Input("x")
    s = RollingMean(5)(x)
    return Dag(inputs=[x], outputs=[Diff(1)(s), RollingMean(3)(s)]), [_row(np.random.default_rng(1).standard_normal(150))]


def _combine():
    a, b = Input("a"), Input("b")
    z = RollingMean(4)(Sub()(combine_latest(a, b)))
    return Dag(inputs=[a, b], outputs=[z]), [_series(120, 2), _series(120, 3)]


def _to_pairs(result):
    """Normalize dag output to a list of (keys, values) pairs.

    Single-output dags return (keys, values); multi-output dags return a tuple
    of (keys, values) pairs. Distinguish by checking the first element type.
    """
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], np.ndarray):
        return [result]
    return list(result)


@pytest.mark.parametrize("factory", [_chain, _fanout, _combine])
def test_batch_equals_oracle(factory):
    dag, feeds = factory()
    got = _to_pairs(dag(*feeds))
    exp = _to_pairs(run_oracle(dag, {nm: f for nm, f in zip(dag._names, feeds)}))
    for (gk, gv), (ek, ev) in zip(got, exp):
        np.testing.assert_array_equal(gk, ek)
        np.testing.assert_array_equal(gv, ev)


@pytest.mark.parametrize("factory", [_chain, _fanout, _combine])
def test_stream_equals_batch(factory):
    dag, feeds = factory()
    batch = _to_pairs(dag(*feeds))
    stream = _to_pairs(dag.stream(*feeds))
    for (bk, bv), (sk, sv) in zip(batch, stream):
        np.testing.assert_array_equal(bk, sk)
        np.testing.assert_array_equal(bv, sv)
