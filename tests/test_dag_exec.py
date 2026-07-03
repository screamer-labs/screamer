import numpy as np
import pytest
from screamer import RollingMean, Diff
from screamer.dag import Input, Dag


def _row(vals):
    v = np.asarray(vals, dtype=np.float64)
    return (np.arange(v.size, dtype=np.int64), v)


def test_single_output_equals_handwritten():
    x = Input("x")
    y = Diff(1)(RollingMean(3)(x))
    dag = Dag(inputs=[x], outputs=[y])
    data = _row(np.arange(20.0))
    (keys, vals) = dag(data)
    exp = Diff(1)(RollingMean(3)(data[1]))
    np.testing.assert_array_equal(vals, exp)
    np.testing.assert_array_equal(keys, data[0])


def test_multi_output_returns_tuple():
    x = Input("x")
    a = RollingMean(3)(x)
    b = Diff(1)(x)
    dag = Dag(inputs=[x], outputs=[a, b])
    data = _row(np.arange(30.0))
    out = dag(data)
    assert isinstance(out, tuple) and len(out) == 2
    np.testing.assert_array_equal(out[0][1], RollingMean(3)(data[1]))
    np.testing.assert_array_equal(out[1][1], Diff(1)(data[1]))


def test_keyword_call_by_input_name():
    x = Input("x")
    y = RollingMean(2)(x)
    dag = Dag(inputs=[x], outputs=[y])
    data = _row(np.arange(10.0))
    (_, v_pos) = dag(data)
    (_, v_kw) = dag(x=data)
    np.testing.assert_array_equal(v_pos, v_kw)


def test_wrong_arg_count_raises():
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[RollingMean(2)(x)])
    with pytest.raises((TypeError, ValueError)):
        dag(_row([1.0]), _row([2.0]))       # 2 args for 1 input


def test_undeclared_input_raises():
    x, extra = Input("x"), Input("extra")
    y = RollingMean(2)(x)                  # does not use `extra`
    with pytest.raises(ValueError):
        Dag(inputs=[x, extra], outputs=[y])  # `extra` declared but unused
