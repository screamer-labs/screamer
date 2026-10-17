import math

import numpy as np
import pytest

from devtools.baselines.RollingPermutationEntropy import RollingPermutationEntropy_numpy
from screamer import RollingPermutationEntropy
from tests.regime_helpers import assert_regimes_agree


def test_monotone_and_constant_series_are_exactly_zero():
    up = np.asarray(RollingPermutationEntropy(10, order=3)(np.arange(50.0)))
    assert (up[11:] == 0.0).all()
    flat = np.asarray(RollingPermutationEntropy(10, order=4)(np.full(50, 7.0)))
    assert (flat[12:] == 0.0).all()


def test_order_two_alternating():
    # patterns up, down, up, down, up from step 1; a window of 3 holds 2:1 of them
    out = np.asarray(RollingPermutationEntropy(3, order=2)(np.array([1.0, 2, 1, 2, 1, 2])))
    assert np.isnan(out[:3]).all()
    h = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)) / math.log(2)
    np.testing.assert_allclose(out[3:], [h, h, h], rtol=1e-12)


def test_ties_rank_by_appearance():
    # (2, 1, 1) and (1, 1, 2) are two different patterns: entropy log 2 / log 3!
    out = np.asarray(RollingPermutationEntropy(2, order=3)(np.array([2.0, 1.0, 1.0, 2.0])))
    np.testing.assert_allclose(out[3], math.log(2) / math.log(6), rtol=1e-12)


def test_white_noise_is_near_one():
    x = np.random.default_rng(0).standard_normal(5000)
    out = np.asarray(RollingPermutationEntropy(1000, order=4)(x))
    assert np.nanmin(out) > 0.97


def test_expanding_and_zero():
    x = np.array([1.0, 2.0, 1.0, 2.0])
    exp = np.asarray(RollingPermutationEntropy(3, order=2, start_policy="expanding")(x))
    assert math.isnan(exp[0])
    np.testing.assert_allclose(exp[1:], [0.0, 1.0, -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)) / math.log(2)],
                               rtol=1e-12)
    zero = np.asarray(RollingPermutationEntropy(3, order=2, start_policy="zero")(x))
    np.testing.assert_allclose(zero, RollingPermutationEntropy_numpy(3, 2, 1, "zero")(x), rtol=1e-12)


@pytest.mark.parametrize("order,delay", [(2, 1), (3, 1), (3, 2), (5, 1), (7, 1)])
@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_matches_baseline(order, delay, start_policy):
    rng = np.random.default_rng(order * 10 + delay)
    x = np.round(rng.standard_normal(400).cumsum(), 1)        # rounding makes ties
    x[[11, 200]] = np.nan
    got = np.asarray(RollingPermutationEntropy(40, order=order, delay=delay, start_policy=start_policy)(x))
    want = RollingPermutationEntropy_numpy(40, order, delay, start_policy)(x)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_long_stream_has_no_drift():
    x = np.random.default_rng(9).standard_normal(100_000)
    got = np.asarray(RollingPermutationEntropy(50, order=3)(x))[-200:]
    want = RollingPermutationEntropy_numpy(50, 3, 1, "strict")(x[-400:])[-200:]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_regimes_agree():
    x = np.random.default_rng(1).standard_normal(300)
    x[5] = np.nan
    assert_regimes_agree(lambda: RollingPermutationEntropy(30, order=4, delay=2), x)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        RollingPermutationEntropy(10, order=1)
    with pytest.raises(ValueError, match="order"):
        RollingPermutationEntropy(10, order=8)
    with pytest.raises(ValueError):
        RollingPermutationEntropy(10, delay=0)
    with pytest.raises(ValueError):
        RollingPermutationEntropy(1)
