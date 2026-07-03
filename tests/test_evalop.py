import numpy as np
from screamer import RollingMean
from screamer import screamer_bindings as _b


def test_screamerbase_arity():
    f = RollingMean(3)
    assert f.num_inputs == 1
    assert f.num_outputs == 1


def test_eval_matches_process_scalar():
    f = RollingMean(3)
    g = RollingMean(3)
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    # eval one value at a time == process_scalar one value at a time
    got = [_b._eval_op(f, [x])[0] for x in xs]
    exp = [g.process_scalar(x) for x in xs]
    np.testing.assert_array_equal(got, exp)
