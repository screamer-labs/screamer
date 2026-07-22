import numpy as np
import pytest
from screamer import Delay


def test_delay_shifts_index_leaves_values():
    vals = np.array([1.0, 2.0, 3.0])
    idx = np.array([0, 7, 14], dtype=np.int64)
    v, i = Delay(5)(vals, idx)
    np.testing.assert_array_equal(v, vals)           # values unchanged
    np.testing.assert_array_equal(i, [5, 12, 19])    # index + duration


def test_delay_requires_explicit_index():
    with pytest.raises(TypeError):
        Delay(5)(np.array([1.0, 2.0, 3.0]))          # no index -> error


def test_delay_on_regular_grid_matches_index_shift():
    vals = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64) * 100         # 100-unit grid
    v, i = Delay(300)(vals, idx)                       # 3-step delay
    np.testing.assert_array_equal(i, idx + 300)
    np.testing.assert_array_equal(v, vals)


def test_delay_irregular_feed_worked_example():
    # 7s-spaced feed, 5s delay: value at t=7 becomes current at t=12
    vals = np.array([10.0, 11.0, 12.0])
    idx = np.array([0, 7, 14], dtype=np.int64)
    v, i = Delay(5)(vals, idx)
    np.testing.assert_array_equal(list(zip(i.tolist(), v.tolist())),
                                  [(5, 10.0), (12, 11.0), (19, 12.0)])


def test_delay_batch_equals_live():
    from screamer.dag import Input, Pipeline
    rng = np.random.default_rng(0)
    n = 200
    vals = rng.standard_normal(n)
    idx = np.cumsum(rng.integers(1, 9, size=n)).astype(np.int64)   # irregular, increasing
    batch_v, batch_i = Delay(4)((vals, idx))

    x = Input("x")
    pipe = Pipeline([x], [Delay(4)(x)])
    sess = pipe.live()
    for t, val in zip(idx.tolist(), vals.tolist()):
        sess.push("x", int(t), float(val))
    sess.flush()
    live_v, live_i = sess.result()
    # compare as (index, value) sets, sorted by index (drain order is not guaranteed)
    order_b = np.argsort(batch_i, kind="stable")
    order_l = np.argsort(live_i, kind="stable")
    np.testing.assert_array_equal(live_i[order_l], batch_i[order_b])
    np.testing.assert_allclose(live_v[order_l], batch_v[order_b])
