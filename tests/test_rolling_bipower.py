import math

import numpy as np
import pytest

from devtools.baselines.RollingBipowerVar import RollingBipowerVar_numpy
from devtools.baselines.RollingJumpShare import RollingJumpShare_numpy
from screamer import RollingBipowerVar, RollingJumpShare
from tests.regime_helpers import assert_regimes_agree


def test_bipower_hand_example():
    out = np.asarray(RollingBipowerVar(3)(np.array([1.0, -2.0, 3.0, -1.0])))
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [6 * math.pi, 6.75 * math.pi], rtol=1e-12)


def test_jump_share_hand_example():
    out = np.asarray(RollingJumpShare(4)(np.array([1.0, 1.0, 1.0, 10.0])))
    np.testing.assert_allclose(out[3], 1 - 8 * math.pi / 103, rtol=1e-12)
    smooth = np.asarray(RollingJumpShare(3)(np.array([1.0, -2.0, 3.0])))
    assert smooth[2] == 0.0                      # BV above RV: no jump


def test_jump_share_of_zero_returns_is_zero():
    out = np.asarray(RollingJumpShare(5)(np.zeros(10)))
    np.testing.assert_array_equal(out[4:], 0.0)


def test_expanding_needs_two_returns_and_zero_pairs_with_zero():
    exp = np.asarray(RollingBipowerVar(3, start_policy="expanding")(np.array([2.0, 3.0])))
    assert math.isnan(exp[0])
    np.testing.assert_allclose(exp[1], math.pi / 2 * 6 * 2, rtol=1e-12)
    zero = np.asarray(RollingBipowerVar(3, start_policy="zero")(np.array([2.0, 3.0])))
    np.testing.assert_allclose(zero, [0.0, math.pi / 2 * 6 * 1.5], rtol=1e-12)


def test_bipower_estimates_the_variance_without_jumps():
    r = np.random.default_rng(0).standard_normal(200_000) * 0.01
    bv = np.asarray(RollingBipowerVar(1000)(r))
    rv = np.convolve(r * r, np.ones(1000), mode="valid")
    assert abs(np.nanmean(bv[999:]) / rv.mean() - 1.0) < 0.01


@pytest.mark.parametrize("cls,ref", [(RollingBipowerVar, RollingBipowerVar_numpy),
                                     (RollingJumpShare, RollingJumpShare_numpy)])
@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_matches_baseline(cls, ref, start_policy):
    rng = np.random.default_rng(5)
    r = rng.standard_normal(500) * 0.01
    r[rng.random(500) < 0.02] *= 15                # jumps
    r[[9, 250]] = np.nan
    got = np.asarray(cls(30, start_policy=start_policy)(r))
    np.testing.assert_allclose(got, ref(30, start_policy)(r), rtol=1e-9, atol=1e-15, equal_nan=True)


@pytest.mark.parametrize("cls", [RollingBipowerVar, RollingJumpShare])
def test_regimes_agree(cls):
    r = np.random.default_rng(6).standard_normal(300)
    r[40] = np.nan
    assert_regimes_agree(lambda: cls(25), r)
