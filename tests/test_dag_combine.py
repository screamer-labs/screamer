import numpy as np
import pytest
from screamer import combine_latest
from screamer import screamer_bindings as _b


def _keys_vals(pairs):
    k = np.array([p[0] for p in pairs], dtype=np.int64)
    v = np.array([p[1] for p in pairs], dtype=np.float64)
    return k, v


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
