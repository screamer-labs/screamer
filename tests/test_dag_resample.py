import numpy as np
import pytest

from screamer import Input, Dag
from screamer.streams import resample, combine_latest


AGGS = ["first", "last", "min", "max", "sum", "count", "mean"]


def _stream_1d(dag, *feeds):
    bk, bv = dag(*feeds)
    sk, sv = dag.stream(*feeds)
    return (bk, bv), (sk, sv)


@pytest.mark.parametrize("agg", AGGS)
def test_resample_by_key_batch_stream_oracle(agg):
    keys = np.array([0, 3, 10, 12, 20, 25], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, width=10, agg=agg)])
    (bk, bv), (sk, sv) = _stream_1d(dag, (keys, vals))
    ek, ev = resample(keys, vals, width=10, agg=agg)   # eager oracle
    np.testing.assert_array_equal(bk, ek)
    np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek)
    np.testing.assert_array_equal(sv.reshape(-1), ev)


@pytest.mark.parametrize("agg", AGGS)
def test_resample_by_key_nan_in_graph(agg):
    # Exercises the C++ ResampleNode's isnan2 NaN-ignore path (batch and stream
    # share the node, so only the eager-oracle comparison catches a NaN bug).
    # bucket [0,10) has a NaN + a finite; bucket [10,20) is all-NaN (emits NaN/0).
    keys = np.array([0, 1, 2, 10, 11], dtype=np.int64)
    vals = np.array([np.nan, 4.0, np.nan, np.nan, np.nan])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, width=10, agg=agg)])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    ek, ev = resample(keys, vals, width=10, agg=agg)   # eager oracle
    # assert_array_equal treats NaN==NaN as equal (not allclose)
    np.testing.assert_array_equal(bk, ek)
    np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek)
    np.testing.assert_array_equal(sv.reshape(-1), ev)


def test_resample_by_key_right_label_and_origin():
    keys = np.array([2, 7, 12, 19], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, width=5, origin=2, agg="sum", label="right")])
    (bk, bv), (sk, sv) = _stream_1d(dag, (keys, vals))
    ek, ev = resample(keys, vals, width=5, origin=2, agg="sum", label="right")
    np.testing.assert_array_equal(bk, ek); np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek); np.testing.assert_array_equal(sv.reshape(-1), ev)


def test_resample_ohlc_width4_in_graph():
    keys = np.array([0, 1, 2, 10, 11], dtype=np.int64)
    vals = np.array([4.0, 2.0, 8.0, 6.0, 7.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, width=10, agg="ohlc")])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    ek, ev = resample(keys, vals, width=10, agg="ohlc")
    assert bv.shape[1] == 4
    np.testing.assert_array_equal(bv, ev)
    np.testing.assert_array_equal(sv, ev)
    np.testing.assert_array_equal(bk, ek)


def test_resample_trailing_bucket_flush_batch_equals_stream():
    # the last bucket [20,30) is partial (only key 25); it must emit in BOTH modes
    keys = np.array([0, 10, 25], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, width=10, agg="last")])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    np.testing.assert_array_equal(bk, [0, 10, 20])
    np.testing.assert_array_equal(bv.reshape(-1), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(sk, bk)
    np.testing.assert_array_equal(sv, bv)


def test_resample_flush_through_combine_latest():
    # resample downstream of a functor fed by combine_latest: flush must reach
    # the resample THROUGH the combine_latest port (the Port::flush fix).
    from screamer import Sub
    ak = np.array([0, 10, 25], dtype=np.int64); av = np.array([5.0, 6.0, 7.0])
    bk = np.array([0, 10, 25], dtype=np.int64); bv = np.array([1.0, 1.0, 1.0])
    a, b = Input("a"), Input("b")
    dag = Dag(inputs=[a, b], outputs=[resample(Sub()(combine_latest(a, b)), width=10, agg="last")])
    rbk, rbv = dag((ak, av), (bk, bv))
    rsk, rsv = dag.stream((ak, av), (bk, bv))
    # batch == stream is the key guarantee (trailing bucket present in both)
    np.testing.assert_array_equal(rbk, rsk)
    np.testing.assert_array_equal(rbv, rsv)
    # the [20,30) bucket (key 25) must be present -> 3 buckets
    assert len(rbk) == 3


def test_resample_feeds_functor():
    from screamer import RollingMean
    keys = np.array([0, 1, 10, 11, 20], dtype=np.int64)
    vals = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[RollingMean(2)(resample(x, width=10, agg="mean"))])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    np.testing.assert_array_equal(bk, sk)
    np.testing.assert_array_equal(bv, sv)


def test_existing_stream_dag_unaffected_by_flush():
    # a non-resample streaming dag must be byte-identical to before (flush no-op)
    from screamer import RollingMean
    keys = np.array([1, 2, 3, 4], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[RollingMean(2)(x)])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    np.testing.assert_array_equal(bk, sk)
    np.testing.assert_array_equal(bv, sv)


def test_resample_by_key_negative_keys_graph():
    keys = np.array([-15, -5, 5, 12], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, width=10, agg="last")])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    ek, ev = resample(keys, vals, width=10, agg="last")   # eager oracle
    np.testing.assert_array_equal(bk, ek); np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek); np.testing.assert_array_equal(sv.reshape(-1), ev)


@pytest.mark.parametrize("agg", AGGS)
def test_resample_by_count_batch_stream_oracle(agg):
    keys = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, count=2, agg=agg)])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    ek, ev = resample(keys, vals, count=2, agg=agg)   # eager oracle (incl. trailing partial)
    np.testing.assert_array_equal(bk, ek)
    np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek)
    np.testing.assert_array_equal(sv.reshape(-1), ev)


def test_resample_by_count_right_label_in_graph():
    keys = np.array([10, 20, 30, 40], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, count=2, agg="last", label="right")])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    ek, ev = resample(keys, vals, count=2, agg="last", label="right")
    np.testing.assert_array_equal(bk, ek); np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek); np.testing.assert_array_equal(sv.reshape(-1), ev)


def test_resample_by_count_ohlc_in_graph():
    keys = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    vals = np.array([4.0, 2.0, 8.0, 6.0, 7.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, count=2, agg="ohlc")])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    ek, ev = resample(keys, vals, count=2, agg="ohlc")
    assert bv.shape[1] == 4
    np.testing.assert_array_equal(bv, ev)
    np.testing.assert_array_equal(sv, ev)
    np.testing.assert_array_equal(bk, ek)
