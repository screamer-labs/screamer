import numpy as np
from screamer import RollingMean
from screamer import RollingCorr, Cart2Polar, BollingerBands
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


def test_functorbase_arity():
    assert RollingCorr(20).num_inputs == 2 and RollingCorr(20).num_outputs == 1
    assert Cart2Polar().num_inputs == 2 and Cart2Polar().num_outputs == 2
    assert BollingerBands(20).num_inputs == 1 and BollingerBands(20).num_outputs == 3


def test_eval_matches_call_2in_1out():
    f = RollingCorr(10)
    g = RollingCorr(10)
    xs = np.random.default_rng(0).standard_normal(30)
    ys = np.random.default_rng(1).standard_normal(30)
    got = [_b._eval_op(f, [x, y])[0] for x, y in zip(xs, ys)]
    exp = [g(x, y) for x, y in zip(xs, ys)]
    np.testing.assert_array_equal(got, exp)


def test_eval_matches_call_2in_2out():
    f = Cart2Polar()
    out = _b._eval_op(f, [3.0, 4.0])
    exp = Cart2Polar()(3.0, 4.0)               # tuple (r, theta)
    np.testing.assert_array_equal(out, list(exp))


def test_eval_matches_call_1in_3out():
    f = BollingerBands(10)
    g = BollingerBands(10)
    xs = np.random.default_rng(3).standard_normal(30)
    got = [tuple(_b._eval_op(f, [x])) for x in xs]
    exp = [tuple(g(x)) for x in xs]           # each call returns a 3-tuple
    np.testing.assert_array_equal(got, exp)
