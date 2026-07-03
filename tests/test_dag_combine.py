import numpy as np
import pytest
from screamer import combine_latest, Sub
from screamer import screamer_bindings as _b



@pytest.mark.parametrize("when_all", [True, False])
def test_combine_latest_node_matches_batch(when_all):
    rng = np.random.default_rng(0)
    a_k = np.sort(rng.integers(0, 500, size=120)).astype(np.int64)
    a_v = rng.standard_normal(120)
    b_k = np.sort(rng.integers(0, 500, size=90)).astype(np.int64)
    b_v = rng.standard_normal(90)

    out_k, out_a = _b._run_combine_latest_batch([a_k, b_k], [a_v, b_v], when_all)
    exp_k, exp_a = combine_latest((a_k, a_v), (b_k, b_v),
                                  emit="when_all" if when_all else "on_any")
    np.testing.assert_array_equal(out_k, exp_k)
    np.testing.assert_array_equal(out_a, exp_a)      # NaN==NaN under assert_array_equal


def test_combine_latest_node_three_inputs():
    rng = np.random.default_rng(1)
    series = []
    for _ in range(3):
        k = np.sort(rng.integers(0, 300, size=60)).astype(np.int64)
        v = rng.standard_normal(60)
        series.append((k, v))
    out_k, out_a = _b._run_combine_latest_batch([s[0] for s in series],
                                                [s[1] for s in series], True)
    exp_k, exp_a = combine_latest(*series)
    np.testing.assert_array_equal(out_k, exp_k)
    np.testing.assert_array_equal(out_a, exp_a)


def test_combine_then_sub_matches_eager():
    # align a,b then a C++ Sub over the width-2 frame == eager Sub()(combine_latest)
    rng = np.random.default_rng(2)
    a_k = np.sort(rng.integers(0, 500, size=100)).astype(np.int64)
    a_v = rng.standard_normal(100)
    b_k = np.sort(rng.integers(0, 500, size=100)).astype(np.int64)
    b_v = rng.standard_normal(100)

    out_k, spread = _b._run_combine_then_sub_batch([a_k, b_k], [a_v, b_v], True)
    exp_k, aligned = combine_latest((a_k, a_v), (b_k, b_v))     # when_all
    np.testing.assert_array_equal(out_k, exp_k)
    np.testing.assert_array_equal(spread.reshape(-1), aligned[:, 0] - aligned[:, 1])


def test_broadcast_fans_out():
    # A width-2 aligned frame delivered to two collectors is identical.
    rng = np.random.default_rng(3)
    a_k = np.sort(rng.integers(0, 200, size=50)).astype(np.int64)
    a_v = rng.standard_normal(50)
    b_k = np.sort(rng.integers(0, 200, size=50)).astype(np.int64)
    b_v = rng.standard_normal(50)
    (k1, a1), (k2, a2) = _b._run_combine_latest_fanout([a_k, b_k], [a_v, b_v], True)
    np.testing.assert_array_equal(k1, k2)
    np.testing.assert_array_equal(a1, a2)
