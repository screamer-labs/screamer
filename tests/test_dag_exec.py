import numpy as np
import pytest
from screamer import RollingMean, Diff, Sub, Add
from screamer.dag import Input, Dag


def _row(vals):
    v = np.asarray(vals, dtype=np.float64)
    return (v, np.arange(v.size, dtype=np.int64))   # (values, index) - values-first


def test_single_output_equals_handwritten():
    x = Input("x")
    y = Diff(1)(RollingMean(3)(x))
    dag = Dag(inputs=[x], outputs=[y])
    data = _row(np.arange(20.0))
    (vals, keys) = dag(data)              # values-first result
    exp = Diff(1)(RollingMean(3)(data[0]))   # data[0] is values
    np.testing.assert_array_equal(vals, exp)
    np.testing.assert_array_equal(keys, data[1])  # data[1] is index


def test_multi_output_returns_tuple():
    x = Input("x")
    a = RollingMean(3)(x)
    b = Diff(1)(x)
    # align_outputs=False: verify executor computes correct values without alignment.
    dag = Dag(inputs=[x], outputs=[a, b], align_outputs=False)
    data = _row(np.arange(30.0))
    out = dag(data)
    assert isinstance(out, tuple) and len(out) == 2
    np.testing.assert_array_equal(out[0][0], RollingMean(3)(data[0]))   # out[i][0]=values, data[0]=values
    np.testing.assert_array_equal(out[1][0], Diff(1)(data[0]))


def test_keyword_call_by_input_name():
    x = Input("x")
    y = RollingMean(2)(x)
    dag = Dag(inputs=[x], outputs=[y])
    data = _row(np.arange(10.0))
    (v_pos, _) = dag(data)          # values-first; v_pos = computed values
    (v_kw, _) = dag(x=data)
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


def test_fanout_wiring_correct():
    # Verify that a shared intermediate correctly feeds both consumers with the
    # same data (fan-out data-flow correctness). A stateful functor evaluated
    # twice (or reset between uses) would produce wrong values.
    x = Input("x")
    shared = RollingMean(3)(x)          # one node, two consumers
    d = Diff(1)(shared)
    m = RollingMean(2)(shared)
    dag = Dag(inputs=[x], outputs=[d, m])
    data = _row(np.arange(15.0))
    out = dag(data)
    assert isinstance(out, tuple) and len(out) == 2
    rm3 = RollingMean(3)(data[0])            # data[0] is values
    np.testing.assert_array_equal(out[0][0], Diff(1)(rm3))     # out[i][0] is values
    np.testing.assert_array_equal(out[1][0], RollingMean(2)(rm3))


def test_align_outputs_default_coindexes_different_branches():
    from screamer import combine_latest
    a = Input("a"); b = Input("b"); c = Input("c")
    # Two branches over different input pairs -> naturally different key sets.
    ab = Sub()(combine_latest(a, b))   # keys = union(a, b)
    ac = Add()(combine_latest(a, c))   # keys = union(a, c)
    dag = Dag(inputs=[a, b, c], outputs=[ab, ac], align_outputs=True)
    ka = (np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 2, 3, 4], dtype=np.int64))   # (values, index)
    kb = (np.array([10.0, 20.0]), np.array([1, 2], dtype=np.int64))
    kc = (np.array([30.0, 40.0]), np.array([3, 4], dtype=np.int64))
    out = dag(ka, kb, kc)
    assert len(out) == 2
    assert out[0][0].shape == out[1][0].shape        # co-indexed, equal length (out[i][0]=values)

    # align_outputs=False leaves the branches at their natural (differing) lengths
    dag2 = Dag(inputs=[a, b, c], outputs=[ab, ac], align_outputs=False)
    out2 = dag2(ka, kb, kc)
    assert out2[0][0].shape != out2[1][0].shape


def test_reused_functor_instance_raises():
    from screamer import RollingMean
    from screamer.dag import Input, Dag
    x = Input("x")
    f = RollingMean(3)
    a = f(x)          # same instance...
    b = f(a)          # ...used in two nodes
    with pytest.raises(ValueError):
        Dag(inputs=[x], outputs=[b])
